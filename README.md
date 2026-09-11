# H20 BF16 Top-K 教学仓库

本仓库在 NVIDIA H20（`sm_90a`）上把 Top-K 拆成可累积的优化阶段，并用 CUDA Event 实测每一步。统一契约为二维 BF16 输入、`K=512`、BF16 values、INT64 indices、`sorted=False`。

## 优化阶段

- **V0**：`torch.sort` 完整排序。
- **V1**：`torch.topk(sorted=True)` 部分选择。
- **V2**：`torch.topk(sorted=False)` 跳过输出排序。
- **V3A**：Scan/filter/radix-select，shared atomic 分配候选槽位，串行 TMA，固定重构阈值。
- **V3B**：只把 atomic 候选压缩替换为 warp ballot/popcount/prefix。
- **V3C**：只加入多缓冲 TMA 预取流水。
- **V3D**：只加入长行提前重构和动态阈值更新。
- **V3E**：生产版按 GPU wave 数选择 256/512 threads、每轮 4096/8192 元素和 TMA 深度。
- **V3F**：C2/C4/C8 thread-block cluster；多个 CTA 扫描同一行，经 DSM 汇总到 CTA0 做最终 Top-K。

H20 对这些精确内核报告的最大可运行 cluster size 为 **8**，因此不构建 C16。C2/C4/C8 均通过独立进程 canary、完整值/索引校验和重复 DSM 同步测试。

## 实测结论

主图固定 `batch=6, K=512`，覆盖 `64K` 到 `4M` 宽度，每点 20 次预热、50 次 CUDA Event 测量。完整矩阵几何平均选择 **C8** 作为固定 V3F，不逐点拼接最优结果。

- V3A→V3B 的几何平均性能为 `0.94x`，说明本矩阵中 ballot 压缩没有单独带来收益。
- V3B→V3C 为 `1.51x`，多缓冲 TMA 流水是 normal 消融中最大的单步收益。
- V3C→V3D 为 `1.01x`，提前更新阈值的平均收益较小。
- V3D→V3E 为 `1.22x`，自适应 CTA 配置在低 batch 下明显有效。
- C8 相对 V3E 的完整矩阵几何平均为 `1.90x`；在 `N=4M` 时为 `3.88x`，达到 `717.2 GB/s`。

所有负收益均保留在图和 JSON 中。详细算法、资源数据和逐阶段表格见 [TOPK_OPTIMIZATION.md](TOPK_OPTIMIZATION.md)。

## 构建与运行

验证环境为 NVIDIA H20、CUDA 12.8、PyTorch 2.7：

```bash
./run_h20.sh build
./run_h20.sh benchmark
./run_h20.sh plot
```

也可直接构建：

```bash
DEEP_SELECT_CUDA_ARCHS=90a python3 setup.py build_ext --inplace
```

`deep_select.topk()` 保持生产入口；教学消融通过 `deep_select.benchmark_topk(..., variant=...)` 调用。`plot` 会更新包含原始 CUDA Event 样本的 [assets/topk_h20_stages.json](assets/topk_h20_stages.json) 和单坐标图 [assets/topk_h20_stages.png](assets/topk_h20_stages.png)。

## H20 绝对带宽

![H20 BF16 Top-K 绝对带宽](assets/topk_h20_stages.png)

纵轴是“输入字节数 / 中位延迟”的有效输入带宽。H20 HBM 理论峰值按 `4000 GB/s` 绘制，它是输入只读一次的理想屋顶线，不是硬件计数器测得的实际流量。

## 来源与许可证

CUDA 实现裁剪自上游 [DeepSeek-AI/DeepSelect](https://github.com/deepseek-ai/DeepSelect)。项目采用 [MIT License](LICENSE)；内含 CUTLASS 与 Kerutils 的许可证或来源说明。
