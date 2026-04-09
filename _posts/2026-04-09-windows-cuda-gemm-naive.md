---
title: 'Windows 下的朴素 GEMM 示例（CUDA）'
date: 2026-04-09
permalink: /posts/2026/04/windows-cuda-gemm-naive/
tags:
  - CUDA
  - GEMM
  - Windows
---

本目录包含在 **Windows + MSVC + NVCC** 上可直接编译的整数矩阵乘示例：`C = A × A`，其中 `A` 为 `(M,K)`、`B` 为 `(K,N)`、`C` 为 `(M,N)`。规模与校验逻辑在源文件中以 `constexpr` 固定。

## 源文件

| 文件 | 说明 |
|------|------|
| `win_naive_multiply_kernel_0.cu` | 早期网格/线程映射版本 |
| `win_naive_multiply_kernel_1.cu` | 中间优化版本 |
| `win_naive_multiply_kernel_2.cu` | **第二版**：32×32 共享内存分块，一线程一输出 |
| `win_naive_multiply_kernel_3.cu` | **第三版**：在第二版思路上进一步做块内与主机侧优化 |

具体编译命令可参考同目录下的 `win_compile.sh`（PowerShell 调用 `nvcc` 的示例行）。

---

## 第三版相对第二版的主要改进

### 1. 线程级寄存器分块（2×2 thread tile）

- **第二版**：每个线程负责 `C` 中 **一个** 元素；块大小为 `32×32` 个线程（1024 线程/块）。
- **第三版**：每个线程用寄存器累加 **2×2** 共四个输出（`TM=2`, `TN=2`），块内仅需 **256** 个线程即可覆盖同样的 32×32 输出瓦片。

效果：在相同 32×32 输出块下，**提高算术强度**（同一块 A/B 子区域被更多乘法复用），减轻对全局/共享带宽的相对压力，并为编译器提供更多 **指令级并行（ILP）** 空间。

### 2. 共享内存布局与 bank 冲突

- **第二版**：`As[TILE][TILE]`、`Bs[TILE][TILE]`，最内层对 `Bs[kk][threadIdx.x]` 的访问模式容易在部分 GPU 上产生 **共享内存 bank 冲突**。
- **第三版**：使用 **`[行][列+1]`** 的填充（如 `As[BM][BK+1]`、`Bs[BK][BN+1]`），使最内层 `kk` 循环中的列索引错开 bank，**降低 bank conflict**。

### 3. 同步与线程参与方式

- **第二版**：`__shared__` 与计算写在 `if (col < n && row < m)` 内部。当 `M`、`N` 不是块尺寸的整数倍时，块内部分线程可能不进入该分支，从而 **不参与 `__syncthreads()`**，存在 **死锁风险**。
- **第三版**：共享内存与整块协同加载、内层乘加均在 **所有线程统一路径** 上执行，仅在最后写回 `C` 时对越界下标做判断，**边界情况更安全**。

### 4. 编译器与占用提示

- 第三版为核函数使用 **`__launch_bounds__(THREADS, 2)`**，并对初始化、内层 `kk` 循环及写回使用 **`#pragma unroll`**，便于 NVCC/ptxas 做循环展开与寄存器权衡。
- 指针使用 **`__restrict__`**，明确无别名，利于加载与调度优化。

### 5. 主机侧：设备绑定与可选统一内存预取

- **第二版**：分配 `cudaMallocManaged` 后仅 `cudaDeviceSynchronize()`，依赖后续访问触发迁移。
- **第三版**：
  - 显式 **`cudaSetDevice(0)`** 绑定设备，避免未定义上下文下的行为歧义。
  - 在 **`cudaDevAttrConcurrentManagedAccess != 0`** 时，使用 CUDA 12+/13 所需的 **`cudaMemLocation`** 形式调用 **`cudaMemPrefetchAsync`**，把 `A/B/C` 预取到 GPU，可能减少核函数启动前的缺页与迁移开销。

**注意（Windows 常见情况）**：许多消费级 GeForce 在 Windows 上 **`concurrentManagedAccess` 为 0**，此时向设备 prefetch **不合法**，会返回 `cudaErrorInvalidDevice`。第三版在检测到该属性为 0 时 **自动跳过 prefetch**，仍与第二版一样依赖按需迁移，**保证可运行**。

### 6. 索引与类型细节

- 第三版在大跨度索引上使用 **`static_cast<size_t>`**，降低 `int` 乘法溢出风险（在更大规模矩阵时更有意义）。

---

## 编译提示

- 将 `win_compile.sh` 中的路径改为你本机的 **CUDA `nvcc`** 与 **MSVC `cl` 所在目录**（`-ccbin`）。
- 若使用 **CUDA 13**，`cudaMemPrefetchAsync` 必须使用 **`cudaMemLocation`** 形式；本仓库第三版已按此编写，并带有上述 **concurrentManagedAccess** 判断。
- 追求测速时可采用 **Release 风格**：例如 MSVC `/O2`、PTXAS `-O3`（参见 `win_compile.sh` 末尾 `naive_win_3_release.exe` 示例行）。

---

## 校验

程序在主机上检查是否满足 `C[i] == K`（在全 1 的 `A`、`B` 下成立）。成功时打印 `pass!`。
