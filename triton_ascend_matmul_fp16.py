import torch
import triton
import triton.language as tl


M_SIZE = 2048
N_SIZE = 1024
K_SIZE = 1536

# Ascend 910B4: 20 AI Core
NUM_CORES = 20
BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 256


@triton.jit
def matmul_kernel_fp16_ascend(
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
    # 固定核心数启动：pid 是核心索引
    pid = tl.program_id(axis=0)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_blocks = num_blocks_m * num_blocks_n

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 每个核心跨步处理多个 block
    for block_idx in range(pid, num_blocks, num_cores):
        block_m = block_idx // num_blocks_n
        block_n = block_idx % num_blocks_n

        m_idx = block_m * BLOCK_SIZE_M + offs_m
        n_idx = block_n * BLOCK_SIZE_N + offs_n

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_idx = k_block * BLOCK_SIZE_K + offs_k

            a_ptrs = a_ptr + m_idx[:, None] * stride_am + k_idx[None, :] * stride_ak
            b_ptrs = b_ptr + k_idx[:, None] * stride_bk + n_idx[None, :] * stride_bn

            a_mask = (m_idx[:, None] < M) & (k_idx[None, :] < K)
            b_mask = (k_idx[:, None] < K) & (n_idx[None, :] < N)

            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            accumulator += tl.dot(a, b)

        c_ptrs = c_ptr + m_idx[:, None] * stride_cm + n_idx[None, :] * stride_cn
        c_mask = (m_idx[:, None] < M) & (n_idx[None, :] < N)
        tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)


def matmul_fp16_ascend(a: torch.Tensor, b: torch.Tensor, num_cores: int = NUM_CORES) -> torch.Tensor:
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("a and b must be fp16 tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape != (M_SIZE, K_SIZE) or b.shape != (K_SIZE, N_SIZE):
        raise ValueError(
            f"expect a.shape=({M_SIZE}, {K_SIZE}), b.shape=({K_SIZE}, {N_SIZE}); "
            f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
        )
    if a.device != b.device:
        raise ValueError("a and b must be on the same device")

    c = torch.empty((M_SIZE, N_SIZE), device=a.device, dtype=torch.float16)

    # 关键：grid 固定为核心数，而不是总 block 数
    matmul_kernel_fp16_ascend[(num_cores,)](
        a,
        b,
        c,
        M_SIZE,
        N_SIZE,
        K_SIZE,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        num_cores=num_cores,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )
    return c


class ModelNew(torch.nn.Module):
    def __init__(self, num_cores: int = NUM_CORES):
        super().__init__()
        self.num_cores = num_cores

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return matmul_fp16_ascend(a, b, num_cores=self.num_cores)


def _default_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("No accelerator device found. Need Ascend NPU or CUDA.")


if __name__ == "__main__":
    device = _default_device()
    a = torch.randn((M_SIZE, K_SIZE), device=device, dtype=torch.float16)
    b = torch.randn((K_SIZE, N_SIZE), device=device, dtype=torch.float16)

    model = ModelNew(num_cores=NUM_CORES)
    c = model(a, b)

    # 仅用于正确性检查
    ref = torch.matmul(a, b)
    max_abs_diff = (c - ref).abs().max().item()
    print(f"output shape: {tuple(c.shape)}, dtype: {c.dtype}, max_abs_diff={max_abs_diff:.6f}")
