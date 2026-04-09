---
title: 'GEMM 优化技术分析报告（Nsight Systems）'
date: 2026-03-20
permalink: /posts/2026/03/gemm-optimize-nsys-analysis/
tags:
  - CUDA
  - GEMM
  - Nsight Systems
  - CUTLASS
  - CuTe
---

本文基于本地项目 `cuda-basic/gemm_optimize` 目录下的多种 GEMM（通用矩阵乘法）实现整理，使用 **NVIDIA Nsight Systems (nsys)** 进行性能分析，总结逐步优化点与 nsys 实测结果。（与该项目内 `README.md` 一致。）

---

## 1. 实验环境与方法

- **矩阵规模**  
  - `naive_0`：M=4096, N=4096, K=1024  
  - `naive_1` / `naive_2` / `cutlass_gemm_basic` / `cutlass_gemm_pro` / `cute_gemm`：M=N=16384, K=4096（即 4096×4）
- **分析工具**：`nsys_easy` → 生成 `nsys_easy.nsys-rep`，再经 `cuda_gpu_sum.py` 得到 **CUDA GPU Summary (Kernels/MemOps)**。
- **指标**：各操作的 Time (%)、Total Time (ns)、Instances、Avg/Med/Min/Max、Category、Operation。

---

## 2. 各版本实现要点

| 版本 | 可执行文件 | 主要实现特点 |
|------|------------|----------------|
| **naive_0** | `naive_0.out` | 1D 网格（128×256），每线程用双重循环覆盖多行/多列；Unified Memory + prefetch；int 类型 |
| **naive_1** | `naive_1.out` | 2D 网格，32×32 block，每线程算一个 C 元素；无分块；Unified Memory；int，规模 4× |
| **naive_2** | `naive_2.out` | 在 naive_1 基础上增加 **共享内存分块**（TILE=32），沿 K 维度分块加载 A/B，减少全局内存访问 |
| **cutlass_basic** | `cutlass_gemm_basic.out` | CUTLASS 官方 float GEMM（MmaPipelined），Unified Memory |
| **cutlass_pro** | `cutlass_gemm.out` | CUTLASS **FP16** GEMM，**显式设备内存**（cudaMalloc + cudaMemcpy H2D/D2H） |
| **cute_gemm** | `cute_gemm.out` | **CuTe** 手写 `gemm_device`：float，CTA tile 128×128×8，`cp.async` + 共享内存；**Unified Memory**（cudaMallocManaged） |

---

## 3. Nsight Systems 性能结果汇总

### 3.1 naive_0（基线：1D 网格 + 多元素/线程）

- **CUDA Kernel** `multiply(int *, int *, int *, int, int, int)`  
  - Time: **98.3%**，Total: **2,185,563,573 ns**（约 2.19 s），Instances: 1  
- **Device-to-Host**：1.2%，27,271,056 ns，6144 次  
- **Host-to-Device**：0.4%，9,888,529 ns，260 次  

**结论**：几乎全部时间在内核上，计算与访存模式未优化，作为基线。

---

### 3.2 naive_1（2D 网格，每线程一元素）

- **CUDA Kernel** `multiply(...)`  
  - Time: **90.4%**，Total: **343,716,183 ns**（约 343.7 ms），Instances: 1  
- **Device-to-Host**：7.0%，26,442,941 ns，6144 次  
- **Host-to-Device**：2.6%，9,881,262 ns，256 次  

**结论**：2D 映射、规整 block 带来明显加速（相对 naive_0 在更大规模下内核时间从秒级降到数百 ms）；内存传输占比上升是因为内核总时间缩短。

---

### 3.3 naive_2（共享内存分块）

- **CUDA Kernel** `multiply(...)`  
  - Time: **88.1%**，Total: **271,443,209 ns**（约 271.4 ms），Instances: 1  
- **Device-to-Host**：8.7%，26,813,575 ns，6144 次  
- **Host-to-Device**：3.2%，9,881,812 ns，256 次  

**结论**：在相同规模下，相对 naive_1 内核时间由 **343.7 ms → 271.4 ms**（约 **21% 提升**），说明共享内存分块有效降低全局内存带宽压力。

---

### 3.4 cutlass_gemm_basic（CUTLASS float + Unified Memory）

- **CUTLASS GEMM Kernel**（MmaPipelined）  
  - Time: **92.5%**，Total: **252,263,796 ns**（约 252.3 ms），Instances: 1  
- **Unified Host-to-Device**：7.5%，20,343,421 ns，8700 次  
- **Unified Device-to-Host**：0.0%，2,270 ns，2 次  

**结论**：内核时间进一步降到 **252.3 ms**，优于 naive_2（271.4 ms），体现 CUTLASS 在 float 上的高效实现；Unified Memory 带来较多小粒度 H2D 传输（8700 次）。

---

### 3.5 cutlass_gemm_pro（CUTLASS FP16 + 显式设备内存）

- **Host-to-Device**  
  - Time: **53.9%**，Total: **39,563,880 ns**（约 39.56 ms），Instances: 2  
- **CUTLASS GEMM Kernel**（MmaPipelined）  
  - Time: **46.1%**，Total: **33,863,442 ns**（约 33.86 ms），Instances: 1  
- **Device-to-Host**：0.0%，1,184 ns，1 次  

**结论**：  
- 内核时间 **33.86 ms**，相对 cutlass_basic 的 252.3 ms 大幅缩短（约 **7.4×**），主要来自 **FP16 与 CUTLASS 对 Tensor Core/半精度的高效利用**。  
- 总 GPU 时间约 **73.4 ms**，其中 H2D 占 53.9%，说明在此配置下**数据传输成为主要瓶颈**；若与计算重叠或增大单次计算量，可进一步隐藏传输。

---

### 3.6 cute_gemm（CuTe tensor GEMM + Unified Memory）

- **CUDA Kernel** `gemm_device<…>`（CuTe 分块 + `cp.async` + `gemm`）  
  - Time: **89.3%**，Total: **244,221,854 ns**（约 244.2 ms），Instances: 1  
- **Unified Host-to-Device**：**10.7%**，**29,220,480 ns**，**16,948** 次  

**结论**：与同规模 float + Unified Memory 的 **cutlass_basic**（内核约 252.3 ms）相比，CuTe 版内核约 **244.2 ms**，略快约 **3%**；H2D 次数（16948）多于 cutlass_basic（8700），但总 H2D 时间（约 29.2 ms）高于 cutlass_basic（约 20.3 ms），说明统一内存迁移仍占一定比例，但计算内核已较优。

---

## 4. 内核时间与优化效果对比（同规模：M=N=16384, K=4096）

| 版本 | 内核总时间 (ns) | 内核占比 | 相对 naive_1 内核加速 |
|------|------------------|----------|------------------------|
| naive_1 | 343,716,183 | 90.4% | 1.0×（基准） |
| naive_2 | 271,443,209 | 88.1% | ≈1.27× |
| cutlass_basic | 252,263,796 | 92.5% | ≈1.36× |
| cute_gemm | 244,221,854 | 89.3% | ≈1.41× |
| cutlass_pro | 33,863,442 | 46.1% | ≈10.2×（FP16 + 显存） |

*注：naive_0 为更小规模，未列入上表同规模对比。*

---

## 5. 逐步优化点总结

1. **naive_0 → naive_1**  
   - 从 1D 网格 + 每线程多元素改为 **2D 网格、每线程一元素**，并增大问题规模；内核从秒级降到数百 ms，说明 2D 映射与规整 block 更利于利用 GPU。

2. **naive_1 → naive_2**  
   - 引入 **共享内存分块（TILE=32）**，沿 K 维度分块加载 A、B，用 `__syncthreads` 同步后计算，减少重复全局内存读取，内核时间约降 21%。

3. **naive_2 → cutlass_basic**  
   - 使用 **CUTLASS float GEMM（MmaPipelined）** 替代手写内核，在相同 float 规模下进一步缩短内核时间，体现库在流水线与内存层次上的优化。

4. **cutlass_basic ↔ cute_gemm（同规模 float + Unified Memory）**  
   - **CuTe** 版 `gemm_device`（128×128×8 tile、`cp.async`）在 nsys 中内核时间略低于 CUTLASS float 基本示例（约 **244 ms vs 252 ms**），说明在相同数据路径下手写 CuTe 分块已接近甚至略优于该 CUTLASS 配置；两者均受 Unified Memory 下大量小粒度 H2D 影响。

5. **cutlass_basic → cutlass_pro**  
   - **数据类型**：float → **FP16 (half_t)**，算力与带宽利用率提升。  
   - **内存管理**：Unified Memory → **显式 cudaMalloc + cudaMemcpy H2D/D2H**，减少统一内存的按需迁移与碎片化。  
   - 内核时间从 252 ms 降到 33.86 ms，整体 GPU 时间约 73 ms；此时 H2D 占比超过一半，后续可考虑 **异步传输 + 多 stream / 重叠** 以进一步优化端到端时间。

---

## 6. 结论

- **手写内核**：从 1D 多元素/线程 → 2D 一元素/线程 → 共享内存分块，每一步都带来可观测的内核加速。  
- **CUTLASS / CuTe**：CUTLASS float 已优于手写 naive tiled；**cute_gemm** 在同规模 float + Unified Memory 下内核时间略低于 `cutlass_gemm_basic`；FP16 + 显式设备内存使内核再提升约 7.4×，总 GPU 时间进入数十 ms 量级。  
- **Nsight Systems** 的 CUDA GPU Summary 清晰区分了 **CUDA_KERNEL** 与 **MEMORY_OPER**（H2D/D2H），便于定位是计算瓶颈还是传输瓶颈，并指导下一步优化（如重叠传输、增大 batch 或使用 Tensor Core 的更高阶 CUTLASS 配置）。

---

*报告中的时间与百分比均来自 Nsight Systems 的 cuda_gpu_sum 报告（nsys_easy 采集）。*
