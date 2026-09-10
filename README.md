# H20 BF16 Top-K 教学仓库

本仓库用四个版本展示 Top-K 的逐步优化：

- **V0**：`torch.sort` 完整排序后截取前 K。
- **V1**：`torch.topk(..., sorted=True)` 部分选择并排序输出。
- **V2**：`torch.topk(..., sorted=False)` 跳过输出排序。
- **V3**：DeepSelect CUDA Scan-Filter-Compact 内核。

V3 仅支持二维、CUDA、连续的 BF16 输入和 `sorted=False`；benchmark 固定返回 BF16 values 与 INT64 indices。完整原理和数据见 [TOPK_OPTIMIZATION.md](TOPK_OPTIMIZATION.md)。

## 关键技术

V0 对全部 N 个元素排序，做了大量与 Top-K 无关的工作；V1 改为部分选择，只对最终 K 个结果排序；V2 放弃结果内部顺序，进一步省去末端排序。V3 DeepSelect 使用 Scan-Filter-Compact：分块扫描输入、维护动态 Top-K 阈值、只把超过阈值的元素写入候选缓冲区，并在候选过多时用 radix-select 压缩，最终只在小候选集上完成选择。

H20 路径使用 `sm_90a`、TMA 数据搬运、128-bit 全局访存以及按 K/批量分派的 256/512-thread CTA。`N=262144` 时实测有效输入带宽约 `1476 GB/s`，达到 H20 `4000 GB/s` HBM 理论峰值的 `36.9%`，距离只读一次输入的理想屋顶线约 `2.71x`。

## 环境

验证环境为 NVIDIA H20（`sm_90a`）、CUDA 12.8、PyTorch 2.7。构建需要可用的 CUDA Toolkit、C++20 编译器，以及与 CUDA 匹配的 PyTorch。

## 构建与运行

```bash
./run_h20.sh build
./run_h20.sh benchmark
./run_h20.sh plot
```

也可直接构建；`DEEP_SELECT_CUDA_ARCHS` 默认值为 `90a`：

```bash
DEEP_SELECT_CUDA_ARCHS=90a python3 setup.py build_ext --inplace
```

`plot` 会更新 [assets/topk_h20_stages.json](assets/topk_h20_stages.json) 和 [assets/topk_h20_stages.png](assets/topk_h20_stages.png)。

## H20 绝对带宽

![H20 BF16 Top-K 绝对带宽](assets/topk_h20_stages.png)

图中绝对性能为“输入字节数 / 中位延迟”的有效输入带宽。H20 HBM3 的理论上限按 **4.0 TB/s（4000 GB/s）** 绘制；它代表输入只读取一次的理想屋顶线，并非 Top-K 内核实际 HBM 流量。

## 来源与许可证

CUDA 实现裁剪自上游 [DeepSeek-AI/DeepSelect](https://github.com/deepseek-ai/DeepSelect)，仅保留 H20 BF16 normal v3 路径。项目采用 [MIT License](LICENSE)；内含 CUTLASS 与 Kerutils 的许可证或来源说明。
