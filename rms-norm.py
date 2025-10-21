# Modified from https://triton-lang.org/main/_downloads/935c0dd0fbeb4b2e69588471cbb2d4b2/05-layer-norm.py
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import triton
import triton.language as tl

DEVICE = 'cuda'


@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Rrms,  # pointer to the 1/rms
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute sum of squares
    _sum_of_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x, 0.)
        _sum_of_squares += x * x
    var = tl.sum(_sum_of_squares, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    # Write rrms
    tl.store(Rrms + row, rrms)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rrms
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _rms_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                           DY,  # pointer to the output gradient
                           DW,  # pointer to the partial sum of weights gradient
                           X,  # pointer to the input
                           W,  # pointer to the weights
                           Rrms,  # pointer to the 1/rms
                           Lock,  # pointer to the lock
                           stride,  # how much to increase the pointer when moving by 1 row
                           N,  # number of columns in X
                           GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rrms = tl.load(Rrms + row)
    # Compute dx
    xhat = x * rrms
    g = w * dy
    xhat = tl.where(mask, xhat, 0.)
    g = tl.where(mask, g, 0.)
    dx = rrms * g - (1 / N) * rrms * rrms * rrms * tl.sum(g * x, axis=0) * x

    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw
    partial_dw = (dy * xhat).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)

    # need a barrier to ensure all threads finished before
    # releasing the lock
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _rms_norm_bwd_dw(DW,  # pointer to the partial sum of weights gradient
                     FINAL_DW,  # pointer to the weights gradient
                     M,  # GROUP_SIZE_M
                     N,  # number of columns
                     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)


class RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        rrms = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _rms_norm_fwd_fused[(M, )](  #
            x_arg, y, weight, rrms,  #
            x_arg.stride(0), N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(x, weight, rrms)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, rrms = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _rms_norm_bwd_dx_fused[(M, )](  #
            dx, dy, _dw, x, w, rrms, locks,  #
            x_arg.stride(0), N,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
        # accumulate partial sums in separate kernel
        _rms_norm_bwd_dw[grid](
            _dw, dw, min(GROUP_SIZE_M, M), N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, None


rms_norm = RMSNorm.apply

def pytorch_rms_norm(x, weight, eps=1e-5):
    """PyTorch reference implementation of RMS Norm."""
    return F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)


# Compiled version for performance comparison
pytorch_rms_norm_compiled = torch.compile(pytorch_rms_norm)


def test_rms_norm(M, N, dtype=torch.float16, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = rms_norm(x, w_shape, weight, eps)
    y_ref = pytorch_rms_norm(x, weight, eps).to(dtype)
    y_compiled = pytorch_rms_norm_compiled(x, weight, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri = [_.grad.clone() for _ in [x, weight]]
    x.grad, weight.grad = None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref = [_.grad.clone() for _ in [x, weight]]
    x.grad, weight.grad = None, None
    # backward pass (torch compiled)
    y_compiled.backward(dy, retain_graph=True)
    dx_compiled, dw_compiled = [_.grad.clone() for _ in [x, weight]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
    assert torch.allclose(y_tri, y_compiled, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_compiled, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_compiled, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch', 'torch_compiled'],
        line_names=['Triton', 'Torch', 'TorchCompiled'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='rms-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_rms_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return rms_norm(x, w_shape, weight, eps)  # noqa: F811, E704

        if provider == "torch":
            return pytorch_rms_norm(x, weight, eps)  # noqa: F811, E704

        if provider == "torch_compiled":
            return pytorch_rms_norm_compiled(x, weight, eps)  # noqa: F811, E704

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        def backward_fn():
            y = y_fwd()
            if provider == "torch_compiled":
                y.backward(dy, create_graph=False, retain_graph=False)  # type: ignore
            else:
                y.backward(dy, retain_graph=True)  # type: ignore
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(backward_fn, quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


test_rms_norm(1151, 8192, dtype=torch.float16)
bench_rms_norm.run(save_path='.', print_data=True)
