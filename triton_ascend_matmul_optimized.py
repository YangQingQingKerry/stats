import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover
    torch_npu = None


def _get_num_cores(default: int = 24) -> int:
    """Best-effort query for Ascend AI Core count."""
    try:
        if not hasattr(torch, "npu"):
            return default
        dev = torch.npu.current_device()
        props = driver.active.utils.get_device_properties(dev)
        for key in ("ai_core_num", "num_aicore", "core_num", "aicore_num"):
            if key in props:
                value = int(props[key])
                if value > 0:
                    return value
    except Exception:
        return default
    return default


def _dtype_to_tl_store_id(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    raise TypeError(f"Unsupported output dtype: {dtype}")


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE": 4, "K_PIPE": 2}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 4, "K_PIPE": 2}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 4, "K_PIPE": 2}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE": 8, "K_PIPE": 4}
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE": 8, "K_PIPE": 2}
        ),
    ],
    key=["M", "N", "K", "DIRECTION", "OUT_DTYPE_ID"],
)
@triton.jit
def matmul_kernel_ascend_swizzle(
    A,
    B,
    C,
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
    DIRECTION,
    OUT_DTYPE_ID,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    K_PIPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_blocks_m * num_blocks_n

    for block_idx in range(pid, num_blocks, num_cores):
        block_m = block_idx // num_blocks_n
        block_n = block_idx % num_blocks_n

        if DIRECTION == 0:
            # M >= N: row-major grouping; better A-tile reuse.
            task_m_idx, task_n_idx = tl.swizzle2d(
                block_m, block_n, num_blocks_m, num_blocks_n, GROUP_SIZE
            )
        else:
            # M < N: column-major grouped mapping; better B-tile reuse.
            size_gn = GROUP_SIZE * num_blocks_m
            group_id = block_idx // size_gn
            off_n = group_id * GROUP_SIZE
            cur_group = tl.minimum(num_blocks_n - off_n, GROUP_SIZE)
            local_idx = block_idx % size_gn
            task_m_idx = local_idx // cur_group
            task_n_idx = off_n + (local_idx % cur_group)

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = task_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Two-level K loop for better load/compute pipelining opportunities.
        for k_outer in range(0, K, BLOCK_K * K_PIPE):
            for k_inner in range(0, K_PIPE):
                k_start = k_outer + k_inner * BLOCK_K
                offs_k = k_start + tl.arange(0, BLOCK_K)

                a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

                a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

                a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

                tl.compile_hint(a_tile, "dot_pad_only_k")
                tl.compile_hint(b_tile, "dot_pad_only_k")

                acc = tl.dot(a_tile, b_tile, acc)

        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        if OUT_DTYPE_ID == 0:
            tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
        elif OUT_DTYPE_ID == 1:
            tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
        else:
            tl.store(c_ptrs, acc, mask=c_mask)


def matmul_ascend_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    num_cores: int | None = None,
) -> torch.Tensor:
    """
    Triton Ascend MatMul: C[M, N] = A[M, K] @ B[K, N]
    - Fixed-core launch: grid=(num_cores,)
    - Swizzle2D grouped scheduling with shape-adaptive direction
    - FP32 accumulation, configurable output dtype
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {x.ndim}D and {y.ndim}D.")
    if x.shape[1] != y.shape[0]:
        raise ValueError(f"Shape mismatch: {x.shape} @ {y.shape}")

    # Keep a safe CPU fallback for local development environments.
    if x.device.type != "npu":
        return torch.matmul(x, y)

    if y.device != x.device:
        y = y.to(x.device)

    if out_dtype is None:
        out_dtype = x.dtype

    supported = (torch.float16, torch.bfloat16, torch.float32)
    if x.dtype not in supported or y.dtype not in supported:
        raise TypeError(
            f"Unsupported input dtypes: x={x.dtype}, y={y.dtype}. "
            f"Supported: {supported}"
        )
    if out_dtype not in supported:
        raise TypeError(f"Unsupported output dtype: {out_dtype}")

    # Contiguous row-major layout gives stable pointer arithmetic and bandwidth.
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    m, k = x.shape
    _, n = y.shape
    c = torch.empty((m, n), device=x.device, dtype=out_dtype)

    if num_cores is None:
        num_cores = _get_num_cores(default=24)

    direction = 1 if m < n else 0
    out_dtype_id = _dtype_to_tl_store_id(out_dtype)

    # Ascend best practice: fixed core-count launch, each core loops multiple blocks.
    grid = (num_cores,)
    matmul_kernel_ascend_swizzle[grid](
        x,
        y,
        c,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        c.stride(0),
        c.stride(1),
        num_cores,
        direction,
        out_dtype_id,
    )
    return c


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # x: (M, K), y: (K, N)
        return matmul_ascend_triton(x, y, out_dtype=x.dtype)


def get_inputs():
    # Benchmark shape from task description.
    x = torch.randn(2048, 1024, device="npu", dtype=torch.float16)
    y = torch.randn(1024, 1536, device="npu", dtype=torch.float16)
    return [x, y]


def get_init_inputs():
    return []

