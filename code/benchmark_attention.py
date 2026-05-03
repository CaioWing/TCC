#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from tcc_kernels import attention_eager, attention_sdpa, triton_flash_attention


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    dtype: str
    latency_ms: float
    tflops: float
    max_abs_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 768, 1024])
    parser.add_argument(
        "--seq-range",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Use an inclusive token sweep, for example: --seq-range 32 1024 32.",
    )
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64])
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--skip-sdpa", action="store_true")
    parser.add_argument("--skip-triton", action="store_true")
    args = parser.parse_args()
    if args.seq_range:
        start, stop, step = args.seq_range
        if start <= 0 or stop <= 0 or step <= 0:
            parser.error("--seq-range values must be positive")
        if start > stop:
            parser.error("--seq-range START must be <= STOP")
        args.seq_lens = list(range(start, stop + 1, step))
    return args


def main() -> None:
    args = parse_args()
    require_cuda()
    torch.manual_seed(args.seed)

    dtype = parse_dtype(args.dtype)
    backends: list[tuple[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]] = [
        ("eager", attention_eager),
    ]
    if not args.skip_sdpa:
        backends.append(("sdpa", attention_sdpa))
    if not args.skip_triton:
        backends.append(("triton", triton_flash_attention))

    print_environment()
    results: list[BenchmarkResult] = []
    for seq_len in args.seq_lens:
        for head_dim in args.head_dims:
            q, k, v = make_inputs(args.batch, args.heads, seq_len, head_dim, dtype)
            reference = attention_eager(q, k, v)
            for backend_name, backend_fn in backends:
                result = run_case(
                    backend_name,
                    backend_fn,
                    q,
                    k,
                    v,
                    reference,
                    args.dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                results.append(result)
                print_result(result)

    if args.csv:
        write_csv(args.csv, results)


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA não está disponível. Execute em uma máquina com GPU NVIDIA e driver ativo.")


def parse_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def make_inputs(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    return q, k, v


def run_case(
    backend_name: str,
    backend_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    reference: torch.Tensor,
    dtype_name: str,
    *,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    output = backend_fn(q, k, v)
    torch.cuda.synchronize()
    max_abs_error = (output.float() - reference.float()).abs().max().item()
    latency_ms = cuda_time_ms(lambda: backend_fn(q, k, v), warmup=warmup, iters=iters)
    batch, heads, seq_len, head_dim = q.shape
    return BenchmarkResult(
        backend=backend_name,
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=dtype_name,
        latency_ms=latency_ms,
        tflops=attention_tflops(batch, heads, seq_len, head_dim, latency_ms),
        max_abs_error=max_abs_error,
    )


def cuda_time_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def attention_tflops(batch: int, heads: int, seq_len: int, head_dim: int, latency_ms: float) -> float:
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    seconds = latency_ms / 1000.0
    return flops / seconds / 1e12


def print_environment() -> None:
    props = torch.cuda.get_device_properties(0)
    print("Ambiente:")
    print(f"  Python: {platform.python_version()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  SMs: {props.multi_processor_count}")
    print()


def print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.backend:>6} | "
        f"B={result.batch} H={result.heads} N={result.seq_len} D={result.head_dim} "
        f"{result.dtype:>4} | "
        f"{result.latency_ms:8.4f} ms | "
        f"{result.tflops:8.2f} TFLOP/s | "
        f"erro max {result.max_abs_error:.3e}"
    )


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BenchmarkResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)
    print(f"\nCSV salvo em: {path}")


if __name__ == "__main__":
    main()
