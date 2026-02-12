try:
    from ai_kernel_generator.utils.triton_autotune_patch import apply_triton_patches

    apply_triton_patches()
except ImportError:
    pass

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

M_SIZE = 2048
N_SIZE = 1024
K_SIZE = 1536
DEFAULT_NUM_CORES = 20

# 仅调 BLOCK 大小和 swizzle 分组；不调 num_warps/num_stages 等 Ascend 不支持参数
MATMUL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 8}),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_SIZE": 8}),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 8}),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 4}),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE": 4}),
]


def get_npu_num_cores(default_cores: int = DEFAULT_NUM_CORES) -> int:
    if hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.npu.current_device()
        props = driver.active.utils.get_device_properties(device)
        return int(props.get("num_aicore", default_cores))
    return default_cores


@triton.autotune(
    configs=MATMUL_AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_fp16_ascend_fast(
    mat_a,
    mat_b,
    mat_c,
    M,
    N,
    K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # 固定核心数启动：每个核心跨步处理多个任务块
    pid = tl.program_id(axis=0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_blocks_m * num_blocks_n

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    for block_idx in range(pid, num_blocks, num_cores):
        # 使用 swizzle2d 优化任务映射，降低热点访存冲突
        block_m, block_n = tl.swizzle2d(
            block_idx // num_blocks_n,
            block_idx % num_blocks_n,
            num_blocks_m,
            num_blocks_n,
            GROUP_SIZE,
        )

        m_idx = block_m * BLOCK_M + offs_m
        n_idx = block_n * BLOCK_N + offs_n

        # 固定 shape 下 tile 全整除，走无 mask 快路径
        a_ptrs = mat_a + (m_idx[:, None] * K + offs_k[None, :])
        b_ptrs = mat_b + (offs_k[:, None] * N + n_idx[None, :])

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            a_tile = tl.load(a_ptrs)
            b_tile = tl.load(b_ptrs)
            tl.compile_hint(a_tile, "dot_pad_only_k")
            tl.compile_hint(b_tile, "dot_pad_only_k")
            acc = tl.dot(a_tile, b_tile, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N

        c_ptrs = mat_c + (m_idx[:, None] * N + n_idx[None, :])
        tl.store(c_ptrs, acc.to(tl.float16))


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
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


def matmul_fp16_ascend_optimized(
    a: torch.Tensor,
    b: torch.Tensor,
    num_cores: int | None = None,
) -> torch.Tensor:
    _validate_inputs(a, b)
    # 连续内存在当前地址计算方式下开销最低
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    if num_cores is None:
        num_cores = get_npu_num_cores()

    c = torch.empty((M_SIZE, N_SIZE), device=a.device, dtype=torch.float16)
    grid = lambda meta: (num_cores,)

    # 关键：autotune 参数由 configs 自动传入
    matmul_kernel_fp16_ascend_fast[grid](
        a,
        b,
        c,
        M_SIZE,
        N_SIZE,
        K_SIZE,
        num_cores=num_cores,
    )
    return c


class ModelNew(torch.nn.Module):
    def __init__(self, num_cores: int | None = None):
        super().__init__()
        self.num_cores = get_npu_num_cores() if num_cores is None else num_cores

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return matmul_fp16_ascend_optimized(x, y, num_cores=self.num_cores)


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

    model = ModelNew()
    c = model(a, b)

    ref = torch.matmul(a, b)
    max_abs_diff = (c - ref).abs().max().item()
    print(f"output shape: {tuple(c.shape)}, dtype: {c.dtype}, max_abs_diff={max_abs_diff:.6f}")
