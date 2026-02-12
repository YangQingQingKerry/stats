import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256},
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256},
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256},
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128},
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256},
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
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
    num_cores: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the program (core) ID
    pid = tl.program_id(0)

    # Calculate total number of output blocks
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_SIZE_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # Each core loops over multiple blocks
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # Compute 2D block indices from the linear block index
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        # Row and column offsets for the current block
        offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Initialize the accumulator in float32 for numerical stability
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Loop over the K dimension in chunks of BLOCK_SIZE_K
        for k in range(0, K, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)

            # Load a block of A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
            a_offsets = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)

            # Load a block of B: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
            b_offsets = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)

            # Matrix multiply-accumulate
            accumulator += tl.dot(a, b)

        # Cast accumulator back to float16 for output
        c = accumulator.to(tl.float16)

        # Store the result block
        c_offsets = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptr + c_offsets, c, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Inner dimensions must match, got {K} and {K2}"

        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        # Ascend 910B has 20 AI Cores
        num_cores = 20

        # Use lambda for grid since autotune manages block size parameters
        grid = lambda meta: (num_cores,)

        # Launch kernel — do NOT pass BLOCK_SIZE_* here; autotune handles them
        matmul_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            num_cores=num_cores,
        )
        return c
