---
title: 从用法到 CUDA kernel：PyTorch LayerNorm 是怎么跑起来的
date: 2026-04-13 09:00:00 +0800
published: true
---

最近在学 CUDA，想找一个真实的工程案例来练手。LayerNorm 挺合适的——它在 Transformer 里几乎无处不在，数学不复杂，但 GPU 上的实现藏着不少值得思考的东西。

这篇文章记录了我从"知道 LayerNorm 是什么"到"看懂 PyTorch 底层 kernel"的过程，希望对同样在入门 CUDA 的朋友有用。

---

## 先说说它解决了什么问题

训练深层网络时，每一层的输入分布会随着参数更新不断偏移。第 L 层的参数更新了，它的输出分布就变了，第 L+1 层拿到的输入就跟着变了。层数越深，这个效应越剧烈——前面任何一层的微小变动都会被后面放大。

这叫 Internal Covariate Shift，实际表现是训练不稳、收敛慢，激活值容易落入饱和区，进而加剧梯度消失。

LayerNorm 的思路很直接：**既然分布会漂，那就每次前向传播时强制把它拉回来**。

具体做法是对每个 token 的特征向量单独归一化，计算该向量自己的均值和方差，然后做标准化。这样不管上一层的输出分布怎么漂，当前层拿到的输入永远是均值 0、方差 1。

---

## 数学原理

$$
\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad
\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2
$$

$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i
$$

γ 和 β 是可学习参数，初始化为全 1 和全 0。它们的作用是还给网络表达自由——如果强制把所有层的激活都压成标准正态，网络的表达能力会受损。加了 γ 和 β，网络可以自己决定"喜欢"什么样的均值和方差。

ε 是个很小的正数（默认 1e-5），防止方差为零时除以零。

---

## PyTorch 里怎么用

```python
import torch
import torch.nn as nn

# batch_size=2, seq_len=4, d_model=8
x = torch.randn(2, 4, 8)

# normalized_shape=8 表示对最后一维（特征维）做归一化
ln = nn.LayerNorm(normalized_shape=8)
y = ln(x)
```

验证一下它真的在做归一化：

```python
token = y[0, 0]
print(token.mean())  # ≈ 0.0
print(token.std())   # ≈ 1.0
```

每个 token 的 8 个数，均值被拉到 0，标准差被拉到 1。

Transformer 里的典型用法是 Pre-LN，先归一化再做子层计算：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x
```

---

## 进入 CUDA kernel

PyTorch 的 CUDA 实现在 `aten/src/ATen/native/cuda/layer_norm_kernel.cu`。

推荐使用github的vscode在线浏览方式——Codespaces，如图：
![image-20260413160457888](/images/md-img/image-20260413160457888.png)

计算均值和方差的 kernel 是 `RowwiseMomentsCUDAKernel`，完整签名：

```cuda-cpp
template <typename T, typename T_ACC, bool rms_norm>
__global__ void RowwiseMomentsCUDAKernel(
    int64_t N, T_ACC eps,
    const T* X, T_ACC* mean, T_ACC* rstd)
```

模板参数 `rms_norm` 是个编译期开关，LayerNorm 和 RMSNorm 共用这一套 kernel，只在最后写结果时分叉。这样编译器会生成两份特化代码，没有任何运行时分支开销。

---

### Welford 的数据结构

看 kernel 里用的类型：

```cuda-cpp
using WelfordType = WelfordData<T_ACC, int64_t>;
using WelfordOp   = WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;
```

`WelfordData` 定义在 `aten/src/ATen/native/SharedReduceOps.h`，内部存四个字段：

```cpp
struct WelfordData {
    scalar_t mean;   // 滚动均值
    scalar_t m2;     // 滚动 M2，即 Σ(x - mean)²
    index_t  n;      // 已处理元素数
    scalar_t nf;     // n 的浮点版本，避免整数除法
};
```

`WelfordOps` 定义了三个操作，分别对应 Welford 算法的三个阶段：

**`reduce`**：单步递推，把一个新元素合并进当前统计量

**`combine`**：合并两个局部统计量

**`project`**：从最终统计量提取结果，返回 `(方差, 均值)` 对。

---

### stride loop：每个 thread 本地累积

kernel 里每个 block 处理一行，blockDim.x = 512：

```cuda-cpp
const int64_t i = blockIdx.x;
WelfordType val(0, 0, 0, 0);

for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
}
```

以 N=2049 为例，thread 0 处理 j=0, 512, 1024, 1536, 2048 共 5 个元素，其余 thread 各处理 4 个。每次调用 `welford_op.reduce` 就是 Welford 单步递推，把新元素的贡献滚动更新进 `val`。

这个阶段 512 个 thread 完全独立，没有任何通信，全部在寄存器里完成。

---

### BlockReduce：warp reduce + shared memory

stride loop 结束后，每个 thread 各自持有一段元素的局部 `WelfordData`。接下来要把 512 个局部值规约成 1 个：

```cuda-cpp
val = cuda_utils::BlockReduce(val, welford_op,
          WelfordType(0, 0, 0, 0), val_shared_ptr);
```

`BlockReduce` 的实现在 `aten/src/ATen/native/cuda/block_reduce.cuh`：

```cuda-cpp
template <typename T, class ReduceOp, typename B = Block1D>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;   // lane id，0~31
  const int wid = tid / C10_WARP_SIZE;   // warp id，0~15

  val = WarpReduce(val, op);             // 第一步：warp 内规约
  __syncthreads();
  if (lid == 0) shared[wid] = val;       // lane 0 写入 shared memory
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : identity_element;
  if (wid == 0) val = WarpReduce(val, op); // 第二步：warp 0 收尾
  return val;
}
```

**第一步：warp 内规约**

`WarpReduce` 内部是 `__shfl_xor_sync` 蝶形交换，5 轮后 warp 内每个 thread 都持有本 warp 所有元素的 Welford 合并结果。关键是这里调用的 `op` 就是 `welford_op`，也就是说每次 shfl 交换完之后用的是 Welford `combine` 公式来合并两个 `WelfordData`，而不是简单相加。

16 个 warp 同时并行做这件事，互不干扰，全程在寄存器里。

**中间的两次 `__syncthreads()`**

第一处在 warp reduce 之后、写 shared memory 之前。注释里写的是 `prevent races when BlockReduces are called in a row`——`shared` 是外部传进来的 `val_shared_ptr`，如果前一次 BlockReduce 还没写完就开始读，会出竞态。

第二处是等 16 个 lane 0 全部写完，warp 0 才能安全读取。

**第二步：warp 0 收尾**

```cuda-cpp
val = (tid < B::Warps()) ? shared[lid] : identity_element;
if (wid == 0) val = WarpReduce(val, op);
```

`B::Warps()` 在 512 线程的 block 里等于 16。warp 0 的前 16 个 lane 各从 `shared[lid]` 读一个 warp 的结果，其余 lane 赋值为 identity（`WelfordType(0,0,0,0)`，merge 时不影响结果）。然后再做一次 `WarpReduce`，thread 0 拿到全 block 的最终 `WelfordData`。

---

### 从最终统计量提取结果

BlockReduce 之后，thread 0 持有合并了 2049 个元素的完整 `WelfordData`，然后用 `project` 取出均值和方差：

```cuda-cpp
if (threadIdx.x == 0) {
    auto [m2, m1] = welford_op.project(val);

    if constexpr (!rms_norm) {
        mean[i] = m1;
        rstd[i] = c10::cuda::compat::rsqrt(m2 + eps);   // LayerNorm
    } else {
        rstd[i] = c10::cuda::compat::rsqrt(m2 + m1*m1 + eps);  // RMSNorm
    }
}
```

注意存的是 `rstd`（reciprocal std，倒数标准差），不是 σ。第二个归一化 kernel 里直接乘这个值，省掉一次除法。

