# H20 Top-K 累积优化与 Cluster 实测

## 1. 公平测试契约

- GPU：NVIDIA H20，CUDA 架构 `sm_90a`
- 输入与输出值：BF16
- 输出索引：INT64
- Batch：6
- K：512
- 输出：完整且正确的无序 Top-K values 和 indices
- N：64K、128K、256K、512K、1M、2M、4M
- 计时：20 次预热，50 次逐样本 CUDA Event 测量，报告 median/p90
- 指标：`batch * N * sizeof(BF16) / median latency`
- 禁止：CPU wall time、插桩、Perfetto、Nsys 和 Ncu

每个实现都验证输出 shape/dtype、索引范围、每行索引唯一、`values == input.gather(indices)`，并将完整 Top-K 值多重集合与 PyTorch reference 比较。压力测试覆盖多 seed、normal、uniform、全相等、大量重复值、升序、降序、batch 1/2/3/6、512/4096/8192 边界和最高 4M 宽度。

性能图：[assets/topk_h20_stages.png](assets/topk_h20_stages.png)

原始数据：[assets/topk_h20_stages.json](assets/topk_h20_stages.json)。JSON 包含 77 个点的全部 CUDA Event 样本、median、p90、capability、canary 状态和 cluster 选择过程。

## 2. 累积阶段

| 阶段 | 相比前一阶段只增加的主要机制 |
|---|---|
| V0 Full sort | `torch.sort` 全排序后取前 K |
| V1 Partial select | 改为 `torch.topk(sorted=True)`，避免全量排序 |
| V2 Unsorted Top-K | 改为 `sorted=False`，不再排序 K 个输出 |
| V3A Scan/filter + atomic | 专用 BF16 Scan-Filter-Compact；shared atomic 分配候选槽；串行 TMA；固定重构阈值 |
| V3B Ballot compaction | 只把 shared atomic 分配替换为 ballot/popcount/prefix |
| V3C TMA pipeline | 只把当前轮搬运改为多缓冲 TMA 预取流水 |
| V3D Adaptive threshold | 只对长行提前重构，使阈值更快升高 |
| V3E Adaptive dispatch | 加入生产版按 wave 选择 256/512 threads、4096/8192 元素每轮和 TMA 深度 |
| V3F Cluster Cx | C 个 CTA 分担一行，DSM 汇总 local Top-K，CTA0 再做 global Top-K |

V3A 必须同时包含精确初始阈值、scan/filter 和 radix-select：如果没有这个基础不变量，最坏情况下需要保存 O(N) 个候选，无法在固定 shared memory 中公平实现宽行版本。

## 3. V0、V1、V2 删除了什么工作

### V0：完整排序

```python
values, indices = torch.sort(x, dim=1, descending=True)
values = values[:, :k]
indices = indices[:, :k]
```

V0 计算全部 N 个元素的完整顺序，然后只保留 K 个，近似工作量为 `O(N log N)`。

### V1：部分选择

```python
values, indices = torch.topk(x, k, dim=1, sorted=True)
```

V1 只寻找最大的 K 个元素，再排序这 K 个输出，删除了后 `N-K` 个元素之间无意义的排序。

### V2：无序 Top-K

```python
values, indices = torch.topk(x, k, dim=1, sorted=False)
```

V2 仍返回完全正确的 Top-K 集合，但不保证内部顺序，因此可以省去最终输出排序。V2 与全部 V3 阶段具有相同输出契约。

## 4. V3 的共同算法：Scan-Filter-Compact

### 4.1 CTA、warp 与线程分工

normal V3 让一个 CTA 处理一行，每行独立维护 survivor、候选和阈值，不需要跨 CTA 同步。

每个线程一轮检查 16 个 BF16；一个 32-thread warp 覆盖 512 个元素：

- 256 threads：8 个 warp，每轮 4096 个元素。
- 512 threads：16 个 warp，每轮 8192 个元素。

输入通过 TMA 搬入 shared memory，线程并行比较、压缩候选，并在需要时执行 radix-select。

### 4.2 阈值不是猜测值

内核先从初始窗口精确选出 K 个 survivor：

```text
survivors = TopK(已经扫描的数据)
threshold = min(survivors)
```

关键不变量是：survivor 始终是已扫描区域的完整 Top-K。数据集合扩大时，第 K 大值只会保持或升高，所以旧阈值只可能偏低，不可能过高：

```text
value > threshold  -> 可能进入最终 Top-K，保留
value < threshold  -> 已有至少 K 个更大值，安全丢弃
value = threshold  -> 相同值任选，但最终必须精确补足 K 个
```

因此不会出现“阈值过高导致最后不足 K 个，再向下寻找”。旧阈值偏低只会放进额外候选，影响性能而不影响正确性。

### 4.3 候选为什么需要写入位置

每个线程对自己的 16 个值生成 16-bit `hit_mask`，`popc(hit_mask)` 是该线程命中的候选数量。所有线程随后并行写同一个 shared-memory candidate array，所以必须得到互不重叠的临时槽位；这个位置不是原始输入 index，而是紧凑候选数组中的写入位置。

V3A 使用 CTA shared counter。每个活跃线程用一次 `atomicAdd_block` 领取一段连续槽位，再把自己的命中元素写进去。它不是每个候选执行一次 global atomic。

V3B 改用两级前缀和：

1. 每个 warp 汇总自己的命中总数。
2. shared memory 中的 warp totals 给出前面 warp 占用的槽位数。
3. warp 内将每线程 0–16 的命中数拆成 5 个 bit。
4. 对每个 bit 执行一次 `ballot`。
5. 当前 lane 对自己之前的置位执行 `popc`，加权还原 exclusive prefix。

最终：

```text
dst_slot = 旧候选数量
         + 前面 warp 的命中总数
         + 当前 warp 内前面线程的命中总数
```

每个线程再用 `ffs` 依次取出 `hit_mask` 中的命中项，连续写入自己的槽位区间。

### 4.4 Radix-select 如何保持恰好 K 个

候选达到触发规模后，CTA 对“旧 survivor + 新 candidates”重新选择：

```text
new_survivors = TopK(old_survivors + candidates)
new_threshold = min(new_survivors)
```

这里不是完整排序，而是两级 8-bit radix-select：

1. 把 BF16 bit pattern 映射为可按无符号整数比较的顺序。
2. 统计高 8 bit 的 256 个桶，定位第 K 大所在桶。
3. 只对该桶统计低 8 bit，得到精确 pivot。
4. 保留全部 `> pivot` 的元素。
5. 计算 `eq_quota = K - count_gt`，从 `== pivot` 中补足恰好 K 个。

survivor 使用双缓冲区，重构后交换读写角色。

## 5. V3A→V3E 的受控差异

### V3A：atomic + serial TMA + fixed threshold

V3A 是专用算法基础版。它已有精确初始 Top-K、阈值过滤和 radix-select，但 TMA 每轮发起后立即等待，没有 lookahead；候选使用 shared atomic 分配；累计约 4096 个候选后才重构。

### V3B：ballot compaction

V3B 只替换候选槽位分配。实测几何平均为 V3A 的 `0.94x`，在 4M 也为 `0.94x`。这说明当前 H20、K=512、batch=6 矩阵下，ballot/prefix 的额外指令并未被避免 shared atomic 的收益抵消。该结论来自受控 CUDA Event 对比，不依赖硬件计数器。

### V3C：多缓冲 TMA pipeline

V3C 保留 ballot 和固定重构阈值，只加入 D4 环形 TMA 缓冲：当前轮比较与候选压缩时提前搬运后续轮次。它相对 V3B 的完整矩阵几何平均为 `1.51x`，4M 为 `2.26x`，是 normal 消融中最大的单步收益。

### V3D：adaptive threshold

V3D 只改变重构触发时机：长行约累计 1024 个候选就提前 radix-select，而不是始终等待约 4096 个。更早得到较高阈值后，后续轮次通常写入更少候选。实测几何平均为 V3C 的 `1.01x`，4M 为 `1.01x`。

### V3E：生产版 adaptive dispatch

V3E 根据 batch 与 SM 数得到 wave 数，并选择：

- 单 wave：512 threads、每轮 8192 元素、较深 TMA pipeline。
- 多 wave：256 threads、每轮 4096 元素、面向更高 CTA occupancy 的配置。

本次 batch=6 属于低 wave 场景。V3E 相对 V3D 的几何平均为 `1.22x`，4M 为 `1.36x`。

## 6. V3F：H20 thread-block cluster

### 6.1 多 CTA 如何处理同一行

cluster Cx 为每行启动 C 个 CTA：

1. rank 0 负责行尾的非置换区域。
2. 可置换前缀按访问顺序切成 C 个连续范围。
3. 每个 CTA 独立执行与 normal V3 相同的扫描，得到 local Top-K。
4. 每个 CTA 通过 Distributed Shared Memory 异步写入 rank 0 的 shared memory。
5. rank 0 收齐最多 `C * K` 个 local candidates，再做一次 global radix-select。
6. 只有 rank 0 写最终 global output。

local Top-K 的并集一定包含 global Top-K：如果一个元素连自己分片的前 K 都进不了，它前面至少已有 K 个更大元素，因此不可能进入整行 Top-K。

### 6.2 DSM 生命周期

实现同时把 DSM 数据地址和 completion mbarrier 地址映射到 rank 0。barrier 初始化后执行 cluster-scoped rendezvous；rank 0 先注册 expected transaction bytes，所有 rank 再发起 remote async store。rank 0 等 value gather 后可开始 pivot 计算，再等待 pair gather。最终输出完成后所有 CTA 再 rendezvous，避免非零 rank 在 remote store 或 rank 0 消费 DSM scratch 前退出。

shortcut、正常路径和 NaN 路径遵守同一 CTA 生命周期协议。

### 6.3 H20 capability 与 canary

| Cluster | Dynamic shared memory | Registers/thread | Max potential cluster | Max active clusters | 结果 |
|---:|---:|---:|---:|---:|---|
| C2 | 93,184 B | 96 | 8 | 78 | 通过 |
| C4 | 109,568 B | 96 | 8 | 32 | 通过 |
| C8 | 142,336 B | 96 | 8 | 7 | 通过 |

三个内核均为零 spill。每个 size 都在独立 Python 进程中通过 exact occupancy query、首次 launch、边界/重复值正确性和 50 次重复 DSM launch。H20 对精确内核报告的最大 potential cluster size 是 8，因此本项目不构建 C16。

## 7. 实测结果

### 7.1 完整矩阵几何平均

| 阶段 | 几何平均有效带宽 | 相比前一 normal 阶段 |
|---|---:|---:|
| V0 | 20.7 GB/s | — |
| V1 | 49.7 GB/s | 2.40x |
| V2 | 56.6 GB/s | 1.14x |
| V3A | 58.6 GB/s | 1.04x |
| V3B | 54.9 GB/s | 0.94x |
| V3C | 82.9 GB/s | 1.51x |
| V3D | 84.0 GB/s | 1.01x |
| V3E | 102.7 GB/s | 1.22x |

Cluster 候选都直接与 V3E 比较：

| Cluster | 几何平均有效带宽 | 相对 V3E |
|---:|---:|---:|
| C2 | 114.6 GB/s | 1.12x |
| C4 | 157.5 GB/s | 1.53x |
| C8 | 194.8 GB/s | 1.90x |

按“完整宽度矩阵几何平均最高”选择固定 C8；图中仍以透明虚线保留 C2/C4，不逐点选择不同 cluster size。

### 7.2 4M 宽度的逐阶段结果

| 阶段 | Median | 有效带宽 | 相比前一阶段 |
|---|---:|---:|---:|
| V0 | 1410.69 us | 35.7 GB/s | — |
| V1 | 470.03 us | 107.1 GB/s | 3.00x |
| V2 | 452.90 us | 111.1 GB/s | 1.04x |
| V3A | 797.97 us | 63.1 GB/s | 0.57x |
| V3B | 846.29 us | 59.5 GB/s | 0.94x |
| V3C | 373.66 us | 134.7 GB/s | 2.26x |
| V3D | 371.46 us | 135.5 GB/s | 1.01x |
| V3E | 272.42 us | 184.8 GB/s | 1.36x |
| V3F C2 | 234.88 us | 214.3 GB/s | 1.16x vs V3E |
| V3F C4 | 115.76 us | 434.8 GB/s | 2.35x vs V3E |
| V3F C8 | 70.18 us | 717.2 GB/s | 3.88x vs V3E |

Cluster 在小宽度主要受固定启动、同步和最终 merge 开销影响：64K 时 C8 相对 V3E 约 `1.05x`。随着 N 增大，单行可并行扫描工作增加，C8 的收益扩大到 4M 的 `3.88x`。

## 8. HBM 屋顶线

图中的有效输入带宽定义为：

```text
effective GB/s = batch * N * sizeof(BF16) / median CUDA Event latency
```

H20 HBM 理论峰值按 `4000 GB/s` 绘制。它假设输入只读一次，是统一的乐观屋顶线，不是硬件计数器测得的实际 HBM 流量。

在 `batch=6, N=4M`：

- V3E：184.8 GB/s，峰值的 4.62%。
- V3F C8：717.2 GB/s，峰值的 17.93%。
- C8 相对 V3E：3.88x。

剩余差距包含低 batch 下可用并行度、候选处理、TMA/DSM 同步、radix-select 和输出写回等成本；由于本实验没有使用硬件计数器，不进一步虚构具体 stall 或流量归因。

## 9. 关键代码

- 编译期消融策略：[csrc/cuda_kernels/config.h](csrc/cuda_kernels/config.h)
- 共享 scan/filter/radix 实现：[csrc/cuda_kernels/common_parts.cuh](csrc/cuda_kernels/common_parts.cuh)
- normal V3：[csrc/cuda_kernels/v3/topk_select.cuh](csrc/cuda_kernels/v3/topk_select.cuh)
- H20 cluster V3：[csrc/cuda_kernels/v3_cluster/topk_select.cuh](csrc/cuda_kernels/v3_cluster/topk_select.cuh)
- 教学 variant 分派：[csrc/api.cpp](csrc/api.cpp)
- 正确性、preflight 与 CUDA Event benchmark：[tests/learn_topk.py](tests/learn_topk.py)
- 单坐标图：[tests/plot_topk_stages.py](tests/plot_topk_stages.py)
