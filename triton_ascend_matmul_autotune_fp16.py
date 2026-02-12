import torch
import triton
import triton.language as tl

M_FIXED = 2048
N_FIXED = 1024
K_FIXED = 1536
NUM_ASCEND_CORES = 20


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_autotune_kernel(
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
    NUM_CORES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_blocks = num_blocks_m * num_blocks_n

    for block_idx in tl.range(pid, total_blocks, NUM_CORES):
        block_m = block_idx // num_blocks_n
        block_n = block_idx % num_blocks_n

        offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_SIZE_K):
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def matmul_fp16_autotune_ascend(
    a: torch.Tensor,
    b: torch.Tensor,
    num_cores: int = NUM_ASCEND_CORES,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul_fp16_autotune_ascend only supports 2D matrices.")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("Input matrices must be fp16.")
    if a.device != b.device:
        raise ValueError("Input matrices must be on the same device.")

    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"K dimension mismatch: {k} vs {k2}.")

    c = torch.empty((m, n), device=a.device, dtype=torch.float16)

    # Ascend pattern: launch a fixed number of cores, each core iterates tiles.
    grid = lambda meta: (num_cores,)
    matmul_autotune_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_CORES=num_cores,
    )
    return c


class ModelNew(torch.nn.Module):
    def __init__(self, num_cores: int = NUM_ASCEND_CORES):
        super().__init__()
        self.num_cores = num_cores

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return matmul_fp16_autotune_ascend(a, b, num_cores=self.num_cores)


def run_fixed_shape_example() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is required. Please run with torch_npu installed.")

    torch.manual_seed(0)
    device = torch.device("npu")
    a = torch.randn((M_FIXED, K_FIXED), device=device, dtype=torch.float16)
    b = torch.randn((K_FIXED, N_FIXED), device=device, dtype=torch.float16)

    c = matmul_fp16_autotune_ascend(a, b, num_cores=NUM_ASCEND_CORES)
    ref = torch.matmul(a, b)
    max_abs_err = (c - ref).abs().max().item()

    print(f"A shape: {tuple(a.shape)}, dtype: {a.dtype}")
    print(f"B shape: {tuple(b.shape)}, dtype: {b.dtype}")
    print(f"C shape: {tuple(c.shape)}, dtype: {c.dtype}")
    print(f"max abs error vs torch.matmul: {max_abs_err:.6f}")


if __name__ == "__main__":
    run_fixed_shape_example()
