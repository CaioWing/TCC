#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from tcc_kernels import attention_eager, attention_sdpa, triton_flash_attention


METRICS = [
    "dram__bytes.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "gpu__time_duration.sum",
]
BACKENDS = {
    "eager": attention_eager,
    "sdpa": attention_sdpa,
    "triton": triton_flash_attention,
}


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
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--backends", choices=sorted(BACKENDS), nargs="+", default=sorted(BACKENDS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-control", choices=["all", "none"], default="all")
    parser.add_argument("--clock-control", choices=["base", "none"], default="none")
    parser.add_argument(
        "--replay-mode",
        choices=["kernel", "range"],
        default="kernel",
        help="Nsight Compute replay mode. Kernel replay is more robust for PyTorch library kernels.",
    )
    parser.add_argument(
        "--ncu-path",
        type=Path,
        help="Optional path to the ncu binary when Nsight Compute is not on PATH.",
    )
    parser.add_argument("--csv", type=Path, default=Path("data/attention_roofline_nsight.csv"))
    parser.add_argument("--raw-csv", type=Path, default=Path("data/attention_roofline_nsight_raw.csv"))
    parser.add_argument("--output-prefix", type=Path, default=Path("results/nsight/attention_"))
    parser.add_argument(
        "--split-configs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Profile each backend/sequence pair in a separate Nsight report before merging CSVs.",
    )
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
    require_ncu(args.ncu_path)

    configs = build_configs(args)
    if args.split_configs and "NSPY_NCU_PROFILE" not in os.environ and len(configs) > 1:
        run_split_configs(args, configs)
        return

    import nsight

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    @nsight.analyze.kernel(
        configs=configs,
        runs=args.runs,
        metrics=METRICS,
        derive_metric=derive_roofline_metrics,
        combine_kernel_metrics=lambda x, y: x + y,
        replay_mode=args.replay_mode,
        cache_control=args.cache_control,
        clock_control=args.clock_control,
        output="progress",
        output_csv=True,
        output_prefix=str(args.output_prefix),
    )
    def profile_attention(
        backend: str,
        seq_len: int,
        batch: int,
        heads: int,
        head_dim: int,
        dtype_name: str,
        warmup: int,
        seed: int,
    ) -> None:
        torch.manual_seed(seed)
        dtype = parse_dtype(dtype_name)
        q, k, v = make_inputs(batch, heads, seq_len, head_dim, dtype)
        backend_fn = BACKENDS[backend]

        for _ in range(warmup):
            backend_fn(q, k, v)
        torch.cuda.synchronize()

        with nsight.annotate(f"{backend}_N{seq_len}"):
            backend_fn(q, k, v)
        torch.cuda.synchronize()

    try:
        results = profile_attention()
    except Exception as exc:
        handle_profiler_error(exc)
        raise SystemExit(1) from None
    df = results.to_dataframe()
    df.to_csv(args.raw_csv, index=False)
    write_roofline_csv(df, args.csv)
    print(f"Raw Nsight data saved to: {args.raw_csv}")
    print(f"Roofline data saved to: {args.csv}")


def build_configs(args: argparse.Namespace) -> list[tuple[str, int, int, int, int, str, int, int]]:
    return [
        (backend, seq_len, args.batch, args.heads, args.head_dim, args.dtype, args.warmup, args.seed)
        for seq_len in args.seq_lens
        for backend in args.backends
    ]


def run_split_configs(
    args: argparse.Namespace,
    configs: list[tuple[str, int, int, int, int, str, int, int]],
) -> None:
    import pandas as pd

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = args.csv.parent / f"{args.csv.stem}_parts"
    raw_parts_dir = args.raw_csv.parent / f"{args.raw_csv.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    raw_parts_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    roofline_parts: list[Path] = []
    raw_parts: list[Path] = []
    for backend, seq_len, batch, heads, head_dim, dtype_name, warmup, seed in configs:
        part_name = f"{backend}_N{seq_len}"
        part_csv = parts_dir / f"{part_name}.csv"
        part_raw_csv = raw_parts_dir / f"{part_name}.csv"
        output_prefix = args.output_prefix.parent / f"{args.output_prefix.name}{part_name}_"
        cmd = [
            sys.executable,
            str(script_path),
            "--batch",
            str(batch),
            "--heads",
            str(heads),
            "--seq-lens",
            str(seq_len),
            "--head-dim",
            str(head_dim),
            "--dtype",
            dtype_name,
            "--backends",
            backend,
            "--runs",
            str(args.runs),
            "--warmup",
            str(warmup),
            "--seed",
            str(seed),
            "--cache-control",
            args.cache_control,
            "--clock-control",
            args.clock_control,
            "--replay-mode",
            args.replay_mode,
            "--csv",
            str(part_csv),
            "--raw-csv",
            str(part_raw_csv),
            "--output-prefix",
            str(output_prefix),
            "--no-split-configs",
        ]
        if args.ncu_path is not None:
            cmd.extend(["--ncu-path", str(args.ncu_path)])

        print(f"\nProfiling {part_name}...", flush=True)
        subprocess.run(cmd, check=True)
        roofline_parts.append(part_csv)
        raw_parts.append(part_raw_csv)

    pd.concat((pd.read_csv(path) for path in roofline_parts), ignore_index=True).to_csv(args.csv, index=False)
    pd.concat((pd.read_csv(path) for path in raw_parts), ignore_index=True).to_csv(args.raw_csv, index=False)
    print(f"\nRaw Nsight data saved to: {args.raw_csv}")
    print(f"Roofline data saved to: {args.csv}")


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA não está disponível. Execute em uma máquina com GPU NVIDIA ativa.")


def require_ncu(ncu_path: Path | None) -> None:
    ncu_path = ncu_path or find_ncu()
    if ncu_path is not None:
        resolved_path = ncu_path.expanduser().resolve()
        if not resolved_path.exists():
            raise SystemExit(f"ncu não encontrado em: {resolved_path}")
        os.environ["PATH"] = f"{resolved_path.parent}{os.pathsep}{os.environ['PATH']}"

    if shutil.which("ncu"):
        return
    raise SystemExit(
        "NVIDIA Nsight Compute CLI (ncu) não está no PATH. "
        "O nsight-python depende do ncu 2022.4.0+ instalado no sistema. "
        "Use --ncu-path /caminho/para/ncu se ele estiver instalado fora do PATH."
    )


def find_ncu() -> Path | None:
    candidates = sorted(Path("/opt/nvidia/nsight-compute").glob("*/ncu"), reverse=True)
    return candidates[0] if candidates else None


def handle_profiler_error(exc: Exception) -> None:
    message = str(exc)
    if "ERR_NVGPUCTRPERM" not in message and "Performance Counters" not in message:
        return

    print(
        "\nProfiling bloqueado pelos GPU Performance Counters da NVIDIA.\n"
        "No Windows host, abra NVIDIA Control Panel como administrador, habilite "
        "Desktop > Enable Developer Settings e em Developer > Manage GPU Performance "
        "Counters escolha Allow access to the GPU performance counters to all users. "
        "Depois execute `wsl --shutdown` no PowerShell e rode este script novamente.\n"
    )


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


def derive_roofline_metrics(
    dram_bytes: float,
    dram_read_bytes: float,
    dram_write_bytes: float,
    gpu_time_ns: float,
    backend: str,
    seq_len: int,
    batch: int,
    heads: int,
    head_dim: int,
    dtype_name: str,
    warmup: int,
    seed: int,
) -> dict[str, float]:
    del backend, dtype_name, warmup, seed, dram_read_bytes, dram_write_bytes
    flops = attention_flops(batch, heads, seq_len, head_dim)
    seconds = gpu_time_ns * 1e-9
    return {
        "attention_flops": flops,
        "dram_bytes": dram_bytes,
        "arithmetic_intensity_flop_per_byte": flops / dram_bytes if dram_bytes else float("nan"),
        "effective_tflops": flops / seconds / 1e12 if seconds else float("nan"),
        "dram_bandwidth_gb_s": dram_bytes / seconds / 1e9 if seconds else float("nan"),
    }


def attention_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> float:
    return float(4 * batch * heads * seq_len * seq_len * head_dim)


def write_roofline_csv(df, path: Path) -> None:
    required_columns = {"Metric", "AvgValue"}
    if not required_columns.issubset(df.columns):
        df.to_csv(path, index=False)
        return

    index_columns = [
        column
        for column in ("Annotation", "backend", "seq_len", "batch", "heads", "head_dim", "dtype_name")
        if column in df.columns
    ]
    wide = (
        df.pivot_table(index=index_columns, columns="Metric", values="AvgValue", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide.to_csv(path, index=False)


if __name__ == "__main__":
    main()
