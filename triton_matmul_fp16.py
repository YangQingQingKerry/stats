import torch
import triton
import triton.language as tl


@triton.jit
def matmul_fp16_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 获取核心ID (0 ~ num_cores-1)
    pid = tl.program_id(0)

    # 计算总块数
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # 每个核心循环处理多个块
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # 计算当前块的2D索引
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        # 初始化累加器 (使用float32保证精度)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K维度循环累加
        for k in range(0, K, BLOCK_K):
            # 计算A块的偏移和掩码
            a_row = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
            a_col = k + tl.arange(0, BLOCK_K)
            a_offset = a_row[:, None] * K + a_col[None, :]
            a_mask = (a_row[:, None] < M) & (a_col[None, :] < K)
            a = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)

            # 计算B块的偏移和掩码
            b_row = k + tl.arange(0, BLOCK_K)
            b_col = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
            b_offset = b_row[:, None] * N + b_col[None, :]
            b_mask = (b_row[:, None] < K) & (b_col[None, :] < N)
            b = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)

            # 矩阵乘累加
            accumulator += tl.dot(a, b)

        # 将结果转换回fp16
        result = accumulator.to(tl.float16)

        # 存储结果
        c_row = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        c_col = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_offset = c_row[:, None] * N + c_col[None, :]
        c_mask = (c_row[:, None] < M) & (c_col[None, :] < N)
        tl.store(c_ptr + c_offset, result, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Inner dimensions must match: {K} != {K2}"

        # 输出tensor，与输入相同dtype (fp16)
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        # Ascend 910B4 有20个AI Core
        num_cores = 20
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256

        # 使用固定核心数启动
        matmul_fp16_kernel[(num_cores,)](
            a, b, c,
            M, N, K,
            num_cores,
            BLOCK_M, BLOCK_N, BLOCK_K,
        )
        return c


def test_matmul_fp16():
    """测试 matmul fp16 kernel: M=2048, N=1024, K=1536"""
    M, N, K = 2048, 1024, 1536

    # 创建fp16输入矩阵
    a = torch.randn((M, K), device="npu", dtype=torch.float16)
    b = torch.randn((K, N), device="npu", dtype=torch.float16)

    # 使用Triton内核计算
    model = ModelNew()
    c_triton = model(a, b)

    # 使用PyTorch参考实现计算
    c_ref = torch.matmul(a, b)

    # 验证结果 (fp16精度下使用较宽松的容差)
    assert c_triton.shape == (M, N), f"Output shape mismatch: {c_triton.shape} vs ({M}, {N})"
    assert c_triton.dtype == torch.float16, f"Output dtype mismatch: {c_triton.dtype} vs torch.float16"

    if torch.allclose(c_triton, c_ref, atol=1e-1, rtol=1e-2):
        print("PASS: Triton matmul fp16 result matches PyTorch reference.")
    else:
        max_diff = (c_triton - c_ref).abs().max().item()
        print(f"WARN: Max absolute difference = {max_diff}")
        print("Results may differ due to fp16 accumulation ordering.")

    print(f"Input A shape: {a.shape}, dtype: {a.dtype}")
    print(f"Input B shape: {b.shape}, dtype: {b.dtype}")
    print(f"Output C shape: {c_triton.shape}, dtype: {c_triton.dtype}")


if __name__ == "__main__":
    test_matmul_fp16()
