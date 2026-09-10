# H20 Top-K 逐步优化说明

## 1. 测试设置

- GPU：NVIDIA H20，CUDA 架构 `sm_90a`
- 数据类型：BF16
- Batch：256
- K：512
- 输入尺寸 N：1024、4096、16384、65536、131072、262144
- 计时：CUDA Event，20 次预热，50 次测量，报告中位延迟
- 正确性：使用 `randn` 输入，逐行完整比较 Top-K 值集合，并验证返回值与索引一致

性能图：

[assets/topk_h20_stages.png](assets/topk_h20_stages.png)

原始数据：

[assets/topk_h20_stages.json](assets/topk_h20_stages.json)

## 2. V0：完整排序

实现：

```python
values, indices = torch.sort(x, dim=1, descending=True)
values = values[:, :k]
indices = indices[:, :k]
```

V0 先对每一行的全部 N 个元素降序排序，再截取前 K 个结果。

### 特点

- 计算了 Top-K 不需要的后 `N-K` 个元素之间的完整顺序。
- 近似计算复杂度为 `O(N log N)`。
- 中间结果包含全部 N 个排序值和索引，读写流量较大。
- 优点是实现简单，并且输出天然有序，适合作为基础版本。

### 性能

当 `N=262144` 时，中位延迟为 `4372.21 us`，定义为 `1.00x` 基线。

## 3. V1：部分选择

实现：

```python
values, indices = torch.topk(x, k, dim=1, sorted=True)
```

V1 不再排序全部 N 个元素，只寻找最大的 K 个元素，然后对这 K 个输出排序。

### 相比 V0 的变化

- 删除了后 `N-K` 个元素之间无意义的排序工作。
- 主要工作从“全量排序”变成“Top-K 选择 + K 个结果排序”。
- 当 `K << N` 时，选择算法明显优于全量排序。

### 性能特点

- `N=262144`：`915.36 us`，相对 V0 为 `4.78x`。
- 在 `N=1024` 和 `N=4096` 等小输入上，调度和选择逻辑的固定开销可能超过节省的排序工作，因此不一定比 V0 快。
- 输入越大，避免全量排序带来的收益越明显。

## 4. V2：跳过 Top-K 输出排序

实现：

```python
values, indices = torch.topk(x, k, dim=1, sorted=False)
```

V2 仍然返回完全正确的 Top-K 值和索引，但不保证这 K 个结果内部有序。

### 相比 V1 的变化

- V1 的输出契约要求 `values[:, 0] >= values[:, 1] >= ...`。
- V2 只要求返回的集合是 Top-K，不要求集合内部顺序。
- 因此可以省去或简化最终 K 个候选的排序工作。

### 性能特点

- `N=262144`：`897.44 us`，相对 V0 为 `4.87x`。
- 相比 V1 仅提升约 `1.02x`。
- 原因是 K 只有 512；大输入下，扫描 N 个输入元素和执行 Top-K 选择才是主要开销，最终排序 512 个结果占比很小。

V2 与 V3 都返回值、INT64 索引和无序 Top-K，因此二者具有一致的输出契约，是最直接的公平对比。

## 5. V3：DeepSelect

实现：

```python
values, indices = deep_select.topk(
    x,
    k,
    sorted=False,
    indices_type=torch.int64,
    return_value=True,
    abort_when_nan_found=False,
)
```

DeepSelect 的核心思想是 Scan-Filter-Compact，即“扫描、过滤、压缩”。它避免让全部 N 个元素进入昂贵的选择或排序过程。

### 5.1 CTA、warp 与线程如何分工

normal-v3 路径让一个 CTA 独立处理一行，因此每一行都有自己的候选集合和阈值，不需要跨 CTA 同步。

每个线程一轮读取并检查 16 个 BF16；一个 warp 的 32 个线程因此覆盖 512 个元素。内核根据 batch、K 和 GPU wave 数选择 256 或 512 个线程：

- 256 threads：8 个 warp，每轮扫描 4096 个元素。
- 512 threads：16 个 warp，每轮扫描 8192 个元素。

后续轮次使用 TMA 和 shared-memory 环形缓冲区流水搬运数据，使下一轮读取尽量与当前轮比较、压缩重叠。

### 5.2 动态阈值不是猜测值

内核先从初始窗口中精确选出 K 个 survivor。阈值是这些 survivor 中最小的值，也就是当前已扫描数据的第 K 大值：

```text
survivors = TopK(已经扫描的数据)
threshold = min(survivors)
```

这个定义给出了关键不变量：内核始终保存已经扫描区域的完整 Top-K。即使后面的元素全部没有超过阈值，已有 survivor 也仍然提供完整的 K 个结果。

对于尚未扫描完整行时，当前第 K 大值相对最终第 K 大值只可能偏低，不可能偏高。随着数据集合扩大，第 K 大值只会保持或升高。因此旧阈值最多让额外候选通过，不会错误丢弃真正的最终 Top-K：

```text
value > threshold  -> 可能替换当前 survivor，保留为候选
value < threshold  -> 前面已经至少有 K 个更大元素，可以安全丢弃
value = threshold  -> 已有 survivor 足以占满 K；相同值之间允许任选
```

当 pivot 存在重复值时，radix-select 会先统计 `> pivot` 的数量，再设置 `eq_quota = K - count_gt`，只从 `== pivot` 的元素中补足所需数量，保证 survivor 始终恰好有 K 个。

### 5.3 每个线程先产生局部命中掩码

线程把自己负责的 16 个值与阈值比较，生成一个 16-bit `hit_mask`：

```text
values:    1.2  3.1  0.5  2.8 ...
threshold: 2.0
hit_mask:   0    1    0    1  ...
```

`popc(hit_mask)` 给出该线程需要写入的候选数量。接下来必须为所有命中元素分配连续且互不冲突的 shared-memory 位置。

### 5.4 Ballot 与两级前缀和如何压缩候选

首先，每个 warp 使用 `__reduce_add_sync` 汇总 32 个线程的命中数量。lane 0 将 warp 总数写入 `smem.warp_cnt[warp_id]`，CTA 同步后，每个 warp 就能计算前面所有 warp 占用了多少位置。

warp 内还需要计算每个线程之前有多少命中。普通实现可以用 `shfl_up` 按 1、2、4、8、16 五个距离执行前缀扫描；DeepSelect 利用“每线程最多命中 16 个”这一约束，将命中数量拆成 5 个二进制位，并对每一位执行一次 ballot：

```text
B[k] = ballot((thread_hit_count >> k) & 1)
lane_prefix += popc(B[k] 中位于当前 lane 之前的位) * 2^k
```

五个 ballot 彼此没有逐级依赖。组合五个位的结果后，`lane_prefix` 就等于当前 warp 中前面所有线程的命中总数。

每个线程的最终写入起点为：

```text
dst_slot = 之前轮次的候选数
         + 前面 warp 的命中数
         + 当前 warp 内前面线程的命中数
```

线程再通过 `ffs` 依次取出 `hit_mask` 中的置位，将自己的候选连续写入 `dst_slot` 开始的位置。这样所有候选自然形成无空洞数组，不需要为每个元素执行全局 atomic，也不会发生写入冲突。

### 5.5 候选重构与阈值更新

阈值在任何时刻都是正确的；什么时候更新阈值只影响性能。候选积累到触发规模后，CTA 对下面的集合重新选择：

```text
新的 survivors = TopK(旧 survivors + 新 candidates)
新的 threshold = min(新的 survivors)
```

普通情况约积累 4096 个候选后重构；长行会提前到约 1024 个候选，避免使用偏低的旧阈值收集过多元素。

重构不是完整排序，而是两级 8-bit radix-select：

1. 将 BF16 bit pattern 转换为可按无符号整数比较的顺序。
2. 并行统计高 8 bit 的 256 个直方桶，定位第 K 大值所在桶。
3. 只对目标高位桶统计低 8 bit，得到精确 pivot。
4. 保留全部 `> pivot` 的元素，再按 `eq_quota` 补足 `== pivot` 的元素。
5. 将恰好 K 个 survivor 写入另一个 shared-memory buffer，并交换双缓冲区。

因此重构触发得晚，只会增加候选写入和下一次 radix-select 的工作量，不影响正确性；它不会出现“阈值过高导致最后不足 K 个，再向下寻找”的回退流程。

### 5.6 扫描结束

扫描完成后，如果仍有未重构候选，内核再执行一次相同的 radix-select。最终 survivor buffer 已经包含整行精确的无序 Top-K，随后写回 BF16 values 和 INT64 indices。

### 5.7 为什么提升最大

V0→V1 和 V1→V2 主要减少排序工作；V2→V3 则改变了进入重型选择过程的数据量：

```text
V2：扫描全部 N，并由通用 Top-K 路径处理选择
V3：并行扫描全部 N，但只让少量阈值候选进入压缩和重新选择
```

当 `N=262144` 时：

- V2：`897.44 us`
- V3：`90.94 us`
- V2→V3：`9.87x`
- V0→V3：`48.08x`

因此最大收益来自候选集缩减、warp 级无原子压缩以及 BF16/H20 专用实现，而不是单条 CUDA 指令或跳过一次小排序。

## 6. 理论硬件上限与当前差距

图中的绝对性能定义为：

```text
有效输入带宽 = batch * N * sizeof(BF16) / 中位延迟
```

该指标统计每个输入元素被读取一次所对应的数据率。它是统一比较不同算法的绝对吞吐指标，不是硬件计数器测得的实际 HBM 流量。

本机 H20 的 HBM3 标称峰值带宽约为 `4.0 TB/s = 4000 GB/s`。因此图中加入了 `4000 GB/s` 水平虚线作为理想硬件屋顶线。

对于 `batch=256, N=262144, BF16`：

```text
输入字节数       = 256 * 262144 * 2 = 134217728 Bytes
理论最低读取时间 = 134217728 / 4.0 TB/s = 33.55 us
DeepSelect 实测   = 90.94 us
有效输入带宽     = 1475.8 GB/s
峰值利用率       = 1475.8 / 4000 = 36.90%
距离理论上限     = 4000 / 1475.8 = 2.71x
```

DeepSelect 在不同尺寸下距离 HBM 屋顶线的情况如下：

| N | DeepSelect 有效输入带宽 | HBM 峰值利用率 | 距离理论上限 |
|---:|---:|---:|---:|
| 1024 | 21.0 GB/s | 0.53% | 190.06x |
| 4096 | 81.7 GB/s | 2.04% | 48.98x |
| 16384 | 197.8 GB/s | 4.95% | 20.22x |
| 65536 | 889.0 GB/s | 22.23% | 4.50x |
| 131072 | 1165.7 GB/s | 29.14% | 3.43x |
| 262144 | 1475.8 GB/s | 36.90% | 2.71x |

小尺寸距离屋顶线很远，主要因为 kernel launch、同步和固定选择开销无法被足够多的数据摊薄。尺寸增大后，DeepSelect 越来越接近带宽受限状态。

`4.0 TB/s` 是只读取输入一次的乐观上限。真实 Top-K 还需要阈值维护、候选写入与压缩、同步以及结果写回，因此不能把剩余差距全部视为可消除的软件损失。

## 7. 六个输入尺寸的实测结果

| N | V0 完整排序 | V1 部分选择 | V2 无序 Top-K | V3 DeepSelect | V3 有效输入带宽 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 41.12 us | 45.38 us | 32.56 us | 24.91 us | 21.0 GB/s |
| 4096 | 49.89 us | 68.83 us | 63.74 us | 25.68 us | 81.7 GB/s |
| 16384 | 303.55 us | 115.79 us | 102.34 us | 42.40 us | 197.8 GB/s |
| 65536 | 1125.70 us | 268.72 us | 254.11 us | 37.74 us | 889.0 GB/s |
| 131072 | 2214.45 us | 527.60 us | 509.86 us | 57.57 us | 1165.7 GB/s |
| 262144 | 4372.21 us | 915.36 us | 897.44 us | 90.94 us | 1475.8 GB/s |

DeepSelect 在六个输入尺寸上均为最快版本，并且 N 越大，相对优势和绝对带宽都越高。

## 8. H20 适配说明

本项目在 H20 上使用 `sm_90a` 构建：

- `setup.py` 支持通过 `DEEP_SELECT_CUDA_ARCHS=90a` 选择目标架构。
- SM90 使用 128-bit 全局加载和存储路径。
- 教学仓库仅保留 H20 使用的 normal v3 内核，不包含 SM100 cluster 路径。
- 所有性能数字均来自 H20 原生 CUDA 扩展实测，不是模拟结果。

关键代码：

- 版本定义与正确性：[tests/learn_topk.py](tests/learn_topk.py)
- 折线图生成：[tests/plot_topk_stages.py](tests/plot_topk_stages.py)
- DeepSelect CUDA 内核：[csrc/cuda_kernels/](csrc/cuda_kernels/)
- H20 分派逻辑：[csrc/api.cpp](csrc/api.cpp)

## 9. 结论

优化路线可以概括为：

```text
完整排序
  -> 只选择 Top-K
  -> 不排序 Top-K 输出
  -> 扫描时过滤并压缩候选集合
```

前三个版本逐步删除不必要的排序工作；DeepSelect 进一步减少进入选择过程的数据量，因此取得数量级最大的性能提升。
