import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None  # type: ignore

import triton
import triton.language as tl


@dataclass(frozen=True)
class MatmulStaticShape:
    M: int = 2048
    K: int = 1024
    N: int = 1536


@triton.autotune(
    configs=[
        # Default-recommended for this shape: fewer K-iterations, good L0 fit.
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 4},
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 8},
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 4},
            num_stages=3,
        ),
        # More parallel blocks (smaller M/N tile) for load-balance.
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 8},
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_SIZE": 8},
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_2048x1536_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # NOTE(Ascend): fixed-core launch is required. grid=(NUM_CORES,)
    pid = tl.program_id(axis=0)  # 0..NUM_CORES-1
    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    num_blocks = num_blocks_m * num_blocks_n

    # Static-shape fast path: 2048/1536/1024 are multiples of 128/64/256 in tuned configs.
    # Keeping masks out helps the compiler generate pure CUBE code.
    tl.multiple_of(K, BLOCK_K)

    for block_idx in range(pid, num_blocks, NUM_CORES):
        block_m = block_idx // num_blocks_n
        block_n = block_idx % num_blocks_n

        # Swizzle2D grouping for better L2/L1 locality.
        # For this workload: M(2048) >= N(1536) -> row-major grouping is preferred.
        task_m, task_n = tl.swizzle2d(block_m, block_n, num_blocks_m, num_blocks_n, GROUP_SIZE)

        m_start = task_m * BLOCK_M
        n_start = task_n * BLOCK_N

        # Accumulator in FP32 (maps to L0C/register-level accumulation).
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Compute block: A[m_start:m_start+BM, :] @ B[:, n_start:n_start+BN]
        # A is row-major (contiguous in K), B is row-major (contiguous in N).
        # We keep pointer arithmetic in 2D to help the compiler.
        rm = m_start + tl.arange(0, BLOCK_M)
        rn = n_start + tl.arange(0, BLOCK_N)

        # K-loop
        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)

            a_ptrs = A_ptr + (rm[:, None] * K + rk[None, :])
            b_ptrs = B_ptr + (rk[:, None] * N + rn[None, :])

            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

            # Let the compiler generate dot with K-padding only when needed.
            tl.compile_hint(a, "dot_pad_only_k")
            tl.compile_hint(b, "dot_pad_only_k")

            acc = tl.dot(a, b, acc)

        c_ptrs = C_ptr + (rm[:, None] * N + rn[None, :])
        tl.store(c_ptrs, acc.to(tl.float16))


def matmul_2048x1536_triton_ascend(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_cores: int = 24,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Specialized matmul for x:(2048,1024) @ y:(1024,1536) on Ascend 910B2 via Triton.
    - Fixed-core launch: grid=(24,)
    - Swizzle2D grouped block mapping
    - FP32 accumulation, FP16 output
    """
    shape = MatmulStaticShape()
    assert x.ndim == 2 and y.ndim == 2
    assert tuple(x.shape) == (shape.M, shape.K), f"x must be {(shape.M, shape.K)}, got {tuple(x.shape)}"
    assert tuple(y.shape) == (shape.K, shape.N), f"y must be {(shape.K, shape.N)}, got {tuple(y.shape)}"
    assert x.dtype == torch.float16 and y.dtype == torch.float16, "this kernel is specialized for fp16 inputs"
    assert x.is_contiguous(), "x must be contiguous (row-major)"
    assert y.is_contiguous(), "y must be contiguous (row-major)"

    if out is None:
        out = torch.empty((shape.M, shape.N), device=x.device, dtype=torch.float16)
    else:
        assert tuple(out.shape) == (shape.M, shape.N)
        assert out.dtype == torch.float16

    grid = (num_cores,)
    matmul_2048x1536_kernel[grid](
        x, y, out,
        M=shape.M, N=shape.N, K=shape.K,
        NUM_CORES=num_cores,
    )
    return out


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return matmul_2048x1536_triton_ascend(x, y)


def get_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
    shape = MatmulStaticShape()
    x = torch.randn(shape.M, shape.K, device="npu" if torch_npu is not None else "cpu", dtype=torch.float16)
    y = torch.randn(shape.K, shape.N, device="npu" if torch_npu is not None else "cpu", dtype=torch.float16)
    return x, y


def _smoke_test():
    if torch_npu is None or not torch.npu.is_available():
        print("torch_npu not available; skip runtime test.")
        return
    x, y = get_inputs()
    ref = torch.matmul(x, y)
    out = matmul_2048x1536_triton_ascend(x, y)
    max_err = (ref - out).abs().max().item()
    print("max_abs_err:", max_err)


if __name__ == "__main__":
    # Optional: reduce autotune noise / compilation cache.
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    _smoke_test()

