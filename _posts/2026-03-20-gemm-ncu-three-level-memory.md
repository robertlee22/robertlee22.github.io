---
title: '用 Nsight Compute 驱动 GEMM 三级访存优化：从全局内存到寄存器'
date: 2026-03-20
permalink: /posts/2026/03/gemm-ncu-three-level-memory/
tags:
  - CUDA
  - GEMM
  - Nsight Compute
---

## 前言

GEMM（通用矩阵乘法）是 CUDA 性能优化的经典入门算子。多数教程的思路是"先讲原理，再写代码，最后看结果"。本文反其道而行——**以 Nsight Compute（NCU）的性能指标为出发点**，每一步优化都从 Profiling 数据中定位瓶颈，然后给出针对性的代码改进。

读完本文，你将掌握：

- NCU 中最关键的性能指标（Duration、Roofline、Memory Chart）的阅读与分析方法；
- GPU 三级存储层次（Global Memory → Shared Memory → Register）对应的三个优化版本；
- 每一步优化的核心原理、代码实现与实测加速比。

> 实验环境：RTX 2060 Mobile/Max-Q（Turing，SM 7.5）｜CUDA 12.6｜矩阵规模 M = N = 8192，K = 2048

---

## V1：朴素实现——建立 Baseline

从 NVIDIA 官方教程出发，最直觉的 GEMM kernel 如下——每个线程独立计算输出矩阵 C 的一个元素，沿 K 维度逐步累加：

```c++
// gemm_v1.cu
__global__ void multiply(float *A, float *B, float *C, int M, int K, int N){
    int col = threadIdx.x + blockDim.x* blockIdx.x ; 
    int row =  threadIdx.y + blockDim.y * blockIdx.y; 
    
    if(col < N && row < M){
        int sum = 0; 
       
        for(int k = 0; k<K; k++){
            sum += A[row* K + k] * B[k*N + col];
        }
        C[row* N + col] = sum; 
    }
}
```

> ⚠️ 注意：此处 `sum` 声明为 `int`，浮点乘积会被截断为整数后累加，是原始教程代码中的一个隐患。后续版本已修正为 `float`。

### NCU Profiling 结果

| 指标               | V1 数值 |
| ------------------ | ------- |
| Duration           | 690 ms  |
| Compute Throughput | 73%     |
| Memory Throughput  | 73%     |

乍看 73% 的吞吐率似乎不低，但这只反映了**硬件单元的忙碌程度**，并不等于有效计算效率。查看 Roofline 图能看到实际计算速度仅 **0.39 TFLOPS**，而 RTX 2060 的 FP32 理论峰值为 **4.36 TFLOPS**——当前仅达到峰值的 **8.9%**，提升空间巨大。

![image-20260412160725460](/images/md-img/image-20260412160725460.png)

### 瓶颈分析：全局内存成为性能杀手

GPU 的存储层级由快到慢依次为：**寄存器 → 共享内存 → L2 Cache → 全局内存（DRAM）**。在经典论文 FlashAttention（Dao et al., 2022）中，核心优化思路正是将数据尽可能搬运到更近的存储层级，减少对慢速内存的依赖。

查看 V1 的内存指标：

![image-20260412161414224](/images/md-img/image-20260412161414224.png)

**全局内存读请求高达 8.59 GB，而共享内存完全未使用。** 每个线程在 K 维度的循环中反复从 DRAM 读取 A 和 B 的元素，大量带宽被重复的全局内存访问消耗。优化方向明确：把数据搬到更近的存储层级。

---

## V2：Shared Memory Tiling——第一层访存优化

核心思路：线程块协作地将 A 和 B 的子块（tile）从全局内存搬运到共享内存，然后从共享内存读取数据做计算。沿 K 维度分块迭代，每次搬运一个 TILE × TILE 的子矩阵。

```c++
// gemm_v2.cu
__global__ void multiply(const float* A, const float* B, float* C, int m, int k, int n) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < n && row < m) {
        float sum = 0.f;

        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];

        for (int t = 0; t * TILE < k; ++t) {
            const int ax = t * TILE + threadIdx.x;
            const int by = t * TILE + threadIdx.y;

            As[threadIdx.y][threadIdx.x] =
                (row < m && ax < k) ? A[static_cast<size_t>(row) * k + ax] : 0.f;
            Bs[threadIdx.y][threadIdx.x] =
                (by < k && col < n) ? B[static_cast<size_t>(by) * n + col] : 0.f;

            __syncthreads();

            for (int kk = 0; kk < TILE; ++kk) {
                sum += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
            }
            __syncthreads();
        }
        C[static_cast<size_t>(row) * n + col] = sum;
    }
}
```

### NCU Profiling 结果

| 指标               | V2 数值 | V1 数值 | 变化     |
| ------------------ | ------- | ------- | -------- |
| Duration           | 506 ms  | 690 ms  | **1.36×** |
| Compute Throughput | 68.96%  | 73%     | —        |
| Memory Throughput  | 68.96%  | 73%     | —        |

耗时缩短至 506 ms，加速比 **1.36×**。再看内存指标——全局内存读请求**下降了 31×**，数据被搬运到了共享内存，Shared Memory 读取量为 5.64 GB：

![image-20260412162324590](/images/md-img/image-20260412162324590.png)

### Roofline 分析

Roofline 图显示计算速度提升至 **0.54 TFLOPS**，但仍仅为理论峰值的 **12%**。落点位于脊点（Ridge Point）左侧，说明**内存访问仍是主要瓶颈**——kernel 的算术强度不足，计算单元在等待数据。

![image-20260412162602598](/images/md-img/image-20260412162602598.png)

V2 中每个线程仅计算 C 的一个元素，从共享内存加载的 A、B 数据在线程间没有复用。要进一步提升算术强度，需要让每个线程计算 C 的多个元素——也就是将数据从共享内存提升到**寄存器**层级。

---

## V3：Register Tiling——第二层访存优化

### 设计思路

V3 的核心变化是**每个线程负责计算 C 的一个 TM × TN 子块**（本例中 TM = TN = 2），将 A、B 的元素显式加载到寄存器变量中，在寄存器层面完成乘加运算。

需要注意的是，线程的分工在两个阶段有所不同：

- **数据搬运阶段**：256 个线程协作搬运 BM × BK 的 A 子块和 BK × BN 的 B 子块，每线程负责多个元素（stride loop）；
- **计算阶段**：每个线程从共享内存读取 TM 个 A 元素和 TN 个 B 元素到寄存器，计算 TM × TN = 4 个 C 元素的部分和。

```c++
// gemm_v3.cu
__global__ void __launch_bounds__(THREADS, 2)
multiply(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
         const int m, const int k, const int n) {
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    const int tx = threadIdx.x % (BN / TN);
    const int ty = threadIdx.x / (BN / TN);

    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

    float creg[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            creg[i][j] = 0.f;
        }
    }

    for (int tile = 0; tile * BK < k; ++tile) {
        const int k0 = tile * BK;

        for (int idx = threadIdx.x; idx < BM * BK; idx += THREADS) {
            const int ar = idx / BK;
            const int ac = idx % BK;
            const int rr = blockIdx.y * BM + ar;
            const int cc = k0 + ac;
            As[ar][ac] = (rr < m && cc < k) ? A[static_cast<size_t>(rr) * k + cc] : 0.f;
        }
        for (int idx = threadIdx.x; idx < BK * BN; idx += THREADS) {
            const int br = idx / BN;
            const int bc = idx % BN;
            const int rr = k0 + br;
            const int cc = blockIdx.x * BN + bc;
            Bs[br][bc] = (rr < k && cc < n) ? B[static_cast<size_t>(rr) * n + cc] : 0.f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a0 = As[ty * TM + 0][kk];
            float a1 = As[ty * TM + 1][kk];
            float b0 = Bs[kk][tx * TN + 0];
            float b1 = Bs[kk][tx * TN + 1];
            creg[0][0] += a0 * b0;
            creg[0][1] += a0 * b1;
            creg[1][0] += a1 * b0;
            creg[1][1] += a1 * b1;
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = row0 + i;
            const int c = col0 + j;
            if (r < m && c < n) {
                C[static_cast<size_t>(r) * n + c] = creg[i][j];
            }
        }
    }
}
```

下图展示了数据搬运阶段的线程分工（左）与计算阶段每个线程负责的 C 子块（右）：


![image-20260413021736301](/images/md-img/image-20260413021736301.png)


![image-20260413022032166](/images/md-img/image-20260413022032166.png)

### NCU Profiling 结果

| 指标               | V3 数值 | V2 数值 | V1 数值 | V3 vs V1  |
| ------------------ | ------- | ------- | ------- | --------- |
| Duration           | 249 ms  | 506 ms  | 690 ms  | **2.77×** |
| Compute Throughput | 73.9%   | 68.96%  | 73%     | —         |
| Memory Throughput  | 73.9%   | 68.96%  | 73%     | —         |

V3 相比 V2 再次加速 **2.0×**，累计相对 V1 加速 **2.77×**。

### Roofline 分析

Roofline 图显示计算速度达到 **1.1 TFLOPS**，为理论峰值的 **24.4%**。相比 V1 的 0.39 TFLOPS，绝对性能提升了近 3 倍。但距离 4.36 TFLOPS 的硬件上限仍有很大空间。

![image-20260413022515969](/images/md-img/image-20260413022515969.png)

---

## 三版性能总览

| 版本 | 优化手段            | Duration | TFLOPS | 峰值占比 | vs V1   |
| ---- | ------------------- | -------- | ------ | -------- | ------- |
| V1   | 朴素实现            | 690 ms   | 0.39   | 8.9%     | 1.00×   |
| V2   | Shared Memory Tiling | 506 ms  | 0.54   | 12.4%    | 1.36×   |
| V3   | Register Tiling     | 249 ms   | 1.10   | 25.2%    | 2.77×   |

从全局内存到共享内存、再从共享内存到寄存器，每一层搬运距离的缩短都带来了可观的加速。这正是 GPU 访存优化的核心思路——**让数据尽可能靠近计算单元**。

---

## 后续方向

V3 达到了理论峰值的约 25%，仍有较大的优化空间。后续可探索的方向包括：

- **增大 tile 尺寸与线程计算量**：增大 TM、TN（如 4×4、8×8），进一步提升算术强度和寄存器复用率，但需注意寄存器压力对 occupancy 的影响；
- **向量化内存访问**：使用 `float4` 等向量类型进行全局内存到共享内存的搬运，提升内存带宽利用率；
- **双缓冲（Double Buffering）**：用两组共享内存交替使用，实现数据搬运与计算的流水线重叠，隐藏访存延迟；
- **Tensor Core 利用**：在支持的硬件上（SM 7.0+）引入 WMMA 或 MMA 指令，将峰值上限从 FP32 TFLOPS 提升到 Tensor 吞吐。

<!-- 实验环境：Windows 11 | RTX 2060 Mobile/Max-Q | CUDA 12.6 | VS 2022 | VSCode | Nsight Compute -->
