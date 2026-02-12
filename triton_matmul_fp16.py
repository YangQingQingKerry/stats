"""
Triton Ascend MatMul FP16 Kernel — 高性能优化版
================================================
矩阵乘法: C = A @ B
- M=2048, N=1024, K=1536
- dtype: float16 (fp16)
- 目标硬件: Ascend 910B

相比参考实现的优化点:
1. 丰富的 autotune 配置空间 (16 组) — 参考实现仅 1 组
2. GROUP_SIZE 同步纳入 autotune 搜索
3. swizzle2d 提升 L2 cache 局部性
4. tl.compile_hint("dot_pad_only_k") 指导 Ascend CUBE 单元对齐
5. 3-arg tl.dot(a, b, acc) 融合乘累加，避免额外 += 开销
6. K 循环内仅计算 K 相关偏移，M/N 偏移和掩码提升到循环外
7. 动态获取 NPU 核心数，适配不同硬件型号
8. forward 中保证输入 contiguous，避免非连续内存带来的性能损失
"""

try:
    from ai_kernel_generator.utils.triton_autotune_patch import apply_triton_patches
    apply_triton_patches()
except ImportError:
    pass

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


def get_npu_properties():
    """获取当前 NPU 设备属性（包括 AI Core 数量）"""
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


# ---------------------------------------------------------------------------
# Autotune 配置空间
# ---------------------------------------------------------------------------
# 参考实现仅有 1 组配置 (256,128,128,GROUP=4)。
# 这里提供 16 组覆盖不同 BLOCK_M/N/K 和 GROUP_SIZE 的组合，
# 让 autotune 在实际硬件上自动选出最优配置。
#
# 对 M=2048, N=1024, K=1536 的分析:
#   BLOCK=128 → M 方向 16 块, N 方向 8 块 = 128 块 (负载均衡好)
#   BLOCK=256 → M 方向 8 块,  N 方向 4 块 = 32 块  (单块计算量大)
#   BLOCK_K 越大 → K 循环迭代次数越少，计算密度越高
#   GROUP_SIZE 影响 swizzle2d 的分组粒度，决定 L2 cache 复用效率
#
# 注意：不要对 num_warps / num_stages / num_ctas 等调优，Ascend 后端不支持
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # --- 大块: 高计算密度，适合计算瓶颈场景 ---
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_SIZE": 2}),
        # --- 中等块: 平衡计算密度和负载均衡 ---
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 2}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 8}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_SIZE": 2}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE": 8}),
        # --- 小块: 最优负载均衡，适合核心数较多或矩阵较小的场景 ---
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE": 2}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 8}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE": 2}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_fp16_kernel(
    mat_a, mat_b, mat_c,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    高性能矩阵乘法内核: C[M, N] = A[M, K] @ B[K, N]

    优化策略:
    - 固定核心数启动 + 循环多块调度
    - swizzle2d 改善 L2 cache 命中率
    - compile_hint 指导 Ascend CUBE 单元 K 维度对齐
    - 3-arg tl.dot 融合乘累加
    - 循环不变量提升 (M/N 偏移和掩码在 K 循环外计算)
    """
    pid = tl.program_id(axis=0)

    # 计算输出矩阵的总块数
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # 每个核心循环处理多个输出块
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # ------------------------------------------------------------------
        # swizzle2d: 将线性 block_idx 映射为 2D (m, n) 索引
        # 按 GROUP_SIZE 分组，使相邻核心处理空间上相近的块，
        # 提升 A/B 在 L2 cache 中的复用率
        # ------------------------------------------------------------------
        raw_m = block_idx // NUM_BLOCKS_N
        raw_n = block_idx % NUM_BLOCKS_N
        task_m, task_n = tl.swizzle2d(
            raw_m, raw_n,
            NUM_BLOCKS_M, NUM_BLOCKS_N,
            GROUP_SIZE,
        )

        # 当前块在 M/N 维度上的元素偏移 (K 循环不变量，提升到外层)
        m_start = task_m * BLOCK_M
        n_start = task_n * BLOCK_N
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = n_start + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        # 预计算 M/N 方向的基地址偏移 (K 循环内不再重复计算)
        a_base = offs_m * K      # [BLOCK_M] — A 每行起始偏移
        b_base = offs_n           # [BLOCK_N] — B 每列起始偏移
        c_base = offs_m * N      # [BLOCK_M] — C 每行起始偏移

        # L0C 累加器 (float32 精度)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # ------------------------------------------------------------------
        # K 维度循环: 分块加载 A 和 B，在 CUBE 单元上做矩阵乘累加
        # ------------------------------------------------------------------
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            # 加载 A 块 [BLOCK_M, BLOCK_K] 到 L0A
            a_offset = a_base[:, None] + offs_k[None, :]
            a_mask = m_mask[:, None] & k_mask[None, :]
            a_tile = tl.load(mat_a + a_offset, mask=a_mask, other=0.0)
            tl.compile_hint(a_tile, "dot_pad_only_k")

            # 加载 B 块 [BLOCK_K, BLOCK_N] 到 L0B
            b_offset = (offs_k * N)[:, None] + b_base[None, :]
            b_mask = k_mask[:, None] & n_mask[None, :]
            b_tile = tl.load(mat_b + b_offset, mask=b_mask, other=0.0)
            tl.compile_hint(b_tile, "dot_pad_only_k")

            # CUBE 矩阵乘累加 (3-arg 形式: acc = dot(a, b) + acc)
            acc = tl.dot(a_tile, b_tile, acc)

        # ------------------------------------------------------------------
        # 存储结果: float32 -> float16 转换后写回全局内存
        # ------------------------------------------------------------------
        c_offset = c_base[:, None] + offs_n[None, :]
        c_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(mat_c + c_offset, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Args:
            x: 输入矩阵 A, shape [M, K], dtype=float16
            y: 输入矩阵 B, shape [K, N], dtype=float16
        Returns:
            mat_c: 输出矩阵 C, shape [M, N], dtype=float16
        """
        # 保证输入连续，避免非连续内存的跨步访问性能损失
        x = x.contiguous()
        y = y.contiguous()

        M, K = x.shape
        K2, N = y.shape
        assert K == K2, f"矩阵维度不匹配: {K} != {K2}"

        # 分配输出张量 (连续内存)
        mat_c = torch.empty((M, N), dtype=x.dtype, device=x.device)

        # 动态获取 NPU 核心数
        num_cores = get_npu_properties()["num_aicore"]

        # autotune 要求 grid 使用 lambda
        # 固定核心数启动: grid = (num_cores,)
        grid = lambda meta: (num_cores,)

        # 调用内核 — BLOCK_M/N/K 和 GROUP_SIZE 由 autotune 自动注入
        matmul_fp16_kernel[grid](
            x, y, mat_c,
            M, N, K, num_cores,
            # 不传递 autotune configs 中的参数
        )

        return mat_c
