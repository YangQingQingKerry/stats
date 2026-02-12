import torch
import torch.nn as nn
import triton
import triton.language as tl


# Ascend 910B2 has 24 AI Cores.
NUM_CORES_910B2 = 24


@triton.autotune(
    configs=[
        # Preferred baseline: fully utilizes L0C (128x256 fp32 = 128KB),
        # while keeping L0A/L0B within 64KB limits with BLOCK_K=128.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 4}, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 3}, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 6}, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_910b2_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    num_cores,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_blocks_m * num_blocks_n

    offs_k = tl.arange(0, BLOCK_K)

    for linear_block in range(pid, num_blocks, num_cores):
        block_m = linear_block // num_blocks_n
        block_n = linear_block % num_blocks_n

        # M>=N for this shape; row-grouped swizzle tends to improve A-tile reuse in cache.
        task_m, task_n = tl.swizzle2d(block_m, block_n, num_blocks_m, num_blocks_n, GROUP_M)

        offs_m = task_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = task_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Static-shape path: (2048,1024)x(1024,1536), all candidate tiles divide dims exactly.
        # So we keep the inner loop mask-free for lower overhead.
        for _ in range(0, K, BLOCK_K):
            a_tile = tl.load(a_ptrs)
            b_tile = tl.load(b_ptrs)
            tl.compile_hint(a_tile, "dot_pad_only_k")
            tl.compile_hint(b_tile, "dot_pad_only_k")
            acc = tl.dot(a_tile, b_tile, acc)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16))


def matmul_2048x1024_1024x1536_triton_ascend(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Fast path for A[2048,1024] @ B[1024,1536] on Ascend 910B2.
    Requirements for custom kernel path:
      - device: NPU
      - dtype: float16
      - contiguous tensors
      - exact static shapes
    Otherwise falls back to torch.matmul.
    """
    if (
        x.shape != (2048, 1024)
        or y.shape != (1024, 1536)
        or x.device.type != "npu"
        or y.device.type != "npu"
        or x.dtype != torch.float16
        or y.dtype != torch.float16
    ):
        return torch.matmul(x, y)

    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    c = torch.empty((2048, 1536), device=x.device, dtype=torch.float16)

    _matmul_910b2_kernel[(NUM_CORES_910B2,)](
        x,
        y,
        c,
        2048,
        1536,
        1024,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_CORES_910B2,
    )
    return c


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return matmul_2048x1024_1024x1536_triton_ascend(x, y)


def get_inputs():
    x = torch.randn(2048, 1024)
    y = torch.randn(1024, 1536)
    if hasattr(torch, "npu") and torch.npu.is_available():
        x = x.npu().half()
        y = y.npu().half()
    return [x, y]


def get_init_inputs():
    return []
