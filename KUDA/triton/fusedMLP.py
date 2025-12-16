import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['o_ptr'],
)
@triton.jit
def fused_mlp_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, bias_ptr, o_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // grid_n
    pid_n = pid % grid_n
    
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (off_m[:, None] * stride_am + off_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (off_k[:, None] * stride_bk + off_n[None, :] * stride_bn)

    accum = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    num_blocks_k = tl.cdiv(K, BLOCK_SIZE_K)
    for k_block in range(num_blocks_k):
        k_offset = k_block * BLOCK_SIZE_K
        k_offsets = k_offset + off_k
        
        # Masks for a: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_mask_m = (off_m[:, None] < M)
        a_mask_k = (k_offsets[None, :] < K)
        a_mask = a_mask_m & a_mask_k
        
        # Masks for b: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_mask_k = (k_offsets[:, None] < K)
        b_mask_n = (off_n[None, :] < N)
        b_mask = b_mask_k & b_mask_n
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accum += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    bias = tl.load(bias_ptr + off_n, mask=off_n < N, other=0.0)
    accum += bias[None, :]

    x = accum
    exp_term = tl.exp(-1.702 * x)
    accum = x / (1.0 + exp_term)
    
    o_ptrs = o_ptr + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn
    tl.store(
        o_ptrs,
        accum,
        mask=(off_m[:, None] < M) & (off_n[None, :] < N),
    )

def fused_mlp(A, B, bias):
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert bias.is_contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    Y = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    fused_mlp_kernel[grid](
        A, B, bias, Y,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        Y.stride(0), Y.stride(1),
    )

    return Y

def torch_mlp(A, B, bias):
    x = A @ B + bias
    exp_term = torch.exp(-1.702 * x)
    return x / (1.0 + exp_term)

A = torch.randn(128, 1024, device='cuda', dtype=torch.float16)
B = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
bias = torch.randn(4096, device='cuda', dtype=torch.float16)

out_triton = fused_mlp(A, B, bias)
out_torch = torch_mlp(A, B, bias)

torch.testing.assert_close(out_triton, out_torch, rtol=1e-2, atol=1e-2)
print("✅ Correctness check passed\n")

# Benchmark setup
import time

def benchmark_fn(fn, args, num_warmup=10, num_iter=100):
    """Benchmark a function with warmup iterations."""
    # Warmup
    for _ in range(num_warmup):
        _ = fn(*args)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iter):
        _ = fn(*args)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / num_iter * 1000
    return avg_time_ms

def compute_gflops(M, N, K, time_ms):
    """Compute GFLOPS: 2*M*N*K operations (multiply-add) / time"""
    ops = 2 * M * N * K  # 2 for multiply-add
    gflops = (ops / 1e9) / (time_ms / 1000)
    return gflops

# Test data
x = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
w = torch.randn(1024, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, device="cuda", dtype=torch.float16)

M, K = x.shape
_, N = w.shape

print("=" * 80)
print(f"Benchmarking MLP: M={M}, K={K}, N={N}")
print("=" * 80)
print()

# 1. Triton fused MLP (direct)
print("1. Triton Fused MLP (direct)...")
triton_time = benchmark_fn(fused_mlp, (x, w, b))
triton_gflops = compute_gflops(M, N, K, triton_time)
print(f"   Time: {triton_time:.3f} ms")
print(f"   GFLOPS: {triton_gflops:.2f}")
print()

# 2. Torch compiled Triton fused MLP
print("2. Torch Compiled Triton Fused MLP...")
compiled_triton_mlp = torch.compile(fused_mlp, mode="reduce-overhead")
compiled_triton_time = benchmark_fn(compiled_triton_mlp, (x, w, b))
compiled_triton_gflops = compute_gflops(M, N, K, compiled_triton_time)
print(f"   Time: {compiled_triton_time:.3f} ms")
print(f"   GFLOPS: {compiled_triton_gflops:.2f}")
print(f"   Speedup vs direct: {triton_time / compiled_triton_time:.2f}x")
print()

# 3. PyTorch reference (eager)
print("3. PyTorch Reference (eager)...")
torch_time = benchmark_fn(torch_mlp, (x, w, b))
torch_gflops = compute_gflops(M, N, K, torch_time)
print(f"   Time: {torch_time:.3f} ms")
print(f"   GFLOPS: {torch_gflops:.2f}")
print(f"   Speedup vs Triton: {torch_time / triton_time:.2f}x")
print()

# 4. Torch compiled PyTorch reference
print("4. Torch Compiled PyTorch Reference...")
compiled_torch_mlp = torch.compile(torch_mlp, mode="reduce-overhead")
compiled_torch_time = benchmark_fn(compiled_torch_mlp, (x, w, b))
compiled_torch_gflops = compute_gflops(M, N, K, compiled_torch_time)
print(f"   Time: {compiled_torch_time:.3f} ms")
print(f"   GFLOPS: {compiled_torch_gflops:.2f}")
print(f"   Speedup vs Triton: {compiled_torch_time / triton_time:.2f}x")
print()


print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Method':<40} {'Time (ms)':<15} {'GFLOPS':<15} {'Speedup':<15}")
print("-" * 80)
print(f"{'Triton Fused (direct)':<40} {triton_time:<15.3f} {triton_gflops:<15.2f} {'1.00x':<15}")
print(f"{'Triton Fused (compiled)':<40} {compiled_triton_time:<15.3f} {compiled_triton_gflops:<15.2f} {triton_time/compiled_triton_time:<15.2f}x")
print(f"{'PyTorch (eager)':<40} {torch_time:<15.3f} {torch_gflops:<15.2f} {torch_time/triton_time:<15.2f}x")
print(f"{'PyTorch (compiled)':<40} {compiled_torch_time:<15.3f} {compiled_torch_gflops:<15.2f} {compiled_torch_time/triton_time:<15.2f}x")
print("=" * 80)

