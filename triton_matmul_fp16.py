"""
Triton Ascend MatMul FP16 Kernel (with Autotune)
=================================================
矩阵乘法: C = A @ B
- M=2048, N=1024, K=1536
- dtype: float16 (fp16)
- 目标硬件: Ascend 910B (20 AI Cores)

使用固定核心数启动模式，每个核心循环处理多个输出块。
通过 autotune 自动搜索最优的 BLOCK_M / BLOCK_N / BLOCK_K 组合。
"""

import torch
import triton
import triton.language as tl

# Ascend 910B 固定 20 个 AI Core
NUM_CORES = 20


@triton.autotune(
    configs=[
        # 注意：不要对 num_warps / num_stages 等参数调优，Ascend 后端不支持
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256}),
    ],
    key=['M', 'N', 'K'],  # 当 M, N, K 变化时触发重新 autotune
)
@triton.jit
def matmul_fp16_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,   # 由 autotune configs 自动传入
    BLOCK_N: tl.constexpr,   # 由 autotune configs 自动传入
    BLOCK_K: tl.constexpr,   # 由 autotune configs 自动传入
):
    """
    矩阵乘法内核: C[M, N] = A[M, K] @ B[K, N]

    使用固定核心数启动，每个核心通过循环处理多个输出块。
    累加器使用 float32 以保证数值精度，最终转换回 float16 存储。
    """
    # 1. 获取程序 ID（核心 ID: 0 ~ num_cores-1）
    pid = tl.program_id(0)

    # 计算输出矩阵的块数
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # 2. 每个核心循环处理多个块
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # 计算当前块的 2D 索引
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        # 当前块在 M 和 N 维度上的起始偏移
        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # 3. 初始化累加器（使用 float32 保证精度）
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # 4. K 维度循环：逐块加载 A 和 B 并累加
        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)

            # 加载 A 块 [BLOCK_M, BLOCK_K]
            a_offsets = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)

            # 加载 B 块 [BLOCK_K, BLOCK_N]
            b_offsets = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)

            # 矩阵乘累加
            accumulator += tl.dot(a, b)

        # 5. 将结果从 float32 转换为 float16 并存储
        result = accumulator.to(tl.float16)

        c_offsets = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptr + c_offsets, result, mask=c_mask)


class ModelNew(torch.nn.Module):
    """
    使用 Triton 内核实现矩阵乘法的 PyTorch 模块。
    C[M, N] = A[M, K] @ B[K, N], dtype=float16
    """

    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        参数:
            a: 输入矩阵 A, shape [M, K], dtype=float16
            b: 输入矩阵 B, shape [K, N], dtype=float16

        返回:
            c: 输出矩阵 C, shape [M, N], dtype=float16
        """
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"矩阵维度不匹配: A 的列数 ({K}) != B 的行数 ({K2})"

        # 分配输出张量
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        # autotune 要求 grid 使用 lambda，meta 包含 configs 中的参数
        # 固定核心数启动: grid 始终为 (NUM_CORES,)
        grid = lambda meta: (NUM_CORES,)

        # 调用内核时不要传递 autotune configs 中的参数
        # （BLOCK_M, BLOCK_N, BLOCK_K 由 autotune 自动注入）
        matmul_fp16_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            num_cores=NUM_CORES,
            # 不要写: BLOCK_M=128  ← 错误！autotune 会自动传入
        )

        return c


def main():
    """
    主函数：演示 M=2048, N=1024, K=1536 的 fp16 矩阵乘法
    """
    # 矩阵维度
    M, N, K = 2048, 1024, 1536

    # 创建 fp16 输入矩阵
    device = "npu"  # Ascend NPU 设备
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)

    # 使用 Triton 内核计算
    model = ModelNew()
    c = model(a, b)

    # 使用 PyTorch 原生矩阵乘法验证
    c_ref = torch.matmul(a, b)

    # 检查结果正确性（fp16 精度下使用较宽松的容差）
    if torch.allclose(c, c_ref, atol=1e-1, rtol=1e-2):
        print("结果验证通过!")
    else:
        max_diff = (c - c_ref).abs().max().item()
        print(f"结果存在差异，最大绝对误差: {max_diff}")

    print(f"输入 A: shape={a.shape}, dtype={a.dtype}")
    print(f"输入 B: shape={b.shape}, dtype={b.dtype}")
    print(f"输出 C: shape={c.shape}, dtype={c.dtype}")


if __name__ == "__main__":
    main()
