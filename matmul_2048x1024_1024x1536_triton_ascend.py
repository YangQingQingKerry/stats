import torch
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


@triton.autotune(
    configs=[
        # Recommended: matches the optimized sketch.
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N_MACRO": 256,
                "BLOCK_N_INNER": 128,
                "BLOCK_K": 256,
                "GROUP_SIZE": 4,
            }
        ),
        # Lower L0 pressure on B path, more K-loop iterations.
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N_MACRO": 256,
                "BLOCK_N_INNER": 128,
                "BLOCK_K": 128,
                "GROUP_SIZE": 4,
            }
        ),
        # Baseline-compatible fallback.
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N_MACRO": 128,
                "BLOCK_N_INNER": 128,
                "BLOCK_K": 256,
                "GROUP_SIZE": 4,
            }
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_ascend_2048x1024_1024x1536_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N_MACRO: tl.constexpr,
    BLOCK_N_INNER: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N_MACRO)
    num_blocks = num_pid_m * num_pid_n

    # Fixed-core launch: each core handles block_id = pid + t * num_cores.
    for block_id in range(pid, num_blocks, num_cores):
        # 2D grouped swizzle on M-major groups.
        num_pid_in_group = GROUP_SIZE * num_pid_n
        group_id = block_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE)
        pid_in_group = block_id % num_pid_in_group

        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n0 = pid_n * BLOCK_N_MACRO + tl.arange(0, BLOCK_N_INNER)
        offs_k = tl.arange(0, BLOCK_K)

        c_acc_0 = tl.zeros((BLOCK_M, BLOCK_N_INNER), dtype=tl.float32)

        if BLOCK_N_MACRO > BLOCK_N_INNER:
            offs_n1 = pid_n * BLOCK_N_MACRO + BLOCK_N_INNER + tl.arange(0, BLOCK_N_INNER)
            c_acc_1 = tl.zeros((BLOCK_M, BLOCK_N_INNER), dtype=tl.float32)

        # K pipeline (compiler can overlap transfer and compute).
        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k

            a_ptrs = A + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
            b0_ptrs = B + k_idx[:, None] * stride_bk + offs_n0[None, :] * stride_bn

            a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
            b0_mask = (k_idx[:, None] < K) & (offs_n0[None, :] < N)

            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b0_tile = tl.load(b0_ptrs, mask=b0_mask, other=0.0)

            # Keep K-tail padding behavior stable on dot path.
            tl.compile_hint(a_tile, "dot_pad_only_k")
            tl.compile_hint(b0_tile, "dot_pad_only_k")
            c_acc_0 = tl.dot(a_tile, b0_tile, c_acc_0)

            if BLOCK_N_MACRO > BLOCK_N_INNER:
                b1_ptrs = B + k_idx[:, None] * stride_bk + offs_n1[None, :] * stride_bn
                b1_mask = (k_idx[:, None] < K) & (offs_n1[None, :] < N)
                b1_tile = tl.load(b1_ptrs, mask=b1_mask, other=0.0)
                tl.compile_hint(b1_tile, "dot_pad_only_k")
                c_acc_1 = tl.dot(a_tile, b1_tile, c_acc_1)

        c0_ptrs = C + offs_m[:, None] * stride_cm + offs_n0[None, :] * stride_cn
        c0_mask = (offs_m[:, None] < M) & (offs_n0[None, :] < N)
        tl.store(c0_ptrs, c_acc_0.to(tl.float16), mask=c0_mask)

        if BLOCK_N_MACRO > BLOCK_N_INNER:
            c1_ptrs = C + offs_m[:, None] * stride_cm + offs_n1[None, :] * stride_cn
            c1_mask = (offs_m[:, None] < M) & (offs_n1[None, :] < N)
            tl.store(c1_ptrs, c_acc_1.to(tl.float16), mask=c1_mask)


def matmul_ascend_optimized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and y.ndim == 2, "x and y must be 2D tensors"
    assert x.device.type == "npu" and y.device.type == "npu", "inputs must be on NPU device"
    m, k = x.shape
    k2, n = y.shape
    assert k == k2, f"matmul shape mismatch: {x.shape} @ {y.shape}"
    assert (m, k, n) == (2048, 1024, 1536), (
        "This kernel is specialized for shape (2048, 1024) x (1024, 1536), "
        f"got {x.shape} x {y.shape}."
    )

    # Shape-specialized path uses FP16 inputs and FP16 output with FP32 accumulate.
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    if y.dtype != torch.float16:
        y = y.to(torch.float16)

    x = x.contiguous()
    y = y.contiguous()
    c = torch.empty((m, n), device=x.device, dtype=torch.float16)

    num_cores = 24
    matmul_ascend_2048x1024_1024x1536_kernel[(num_cores,)](
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
    )
    return c


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return matmul_ascend_optimized(x, y)


def get_inputs():
    x = torch.randn((2048, 1024), dtype=torch.float16)
    y = torch.randn((1024, 1536), dtype=torch.float16)
    return [x, y]


def get_init_inputs():
    return []

