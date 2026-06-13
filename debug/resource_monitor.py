"""
Per-core CPU + GPU utilization monitor (no sudo required).

CPU: psutil per-core percentages (P-cores and E-cores listed separately).
GPU: AGXAccelerator PerformanceStatistics via ioreg ("Device Utilization %").
Memory: process RSS-equivalents are unreliable for MPS, so we track
system-level: used memory, swap, and GPU "In use system memory" from ioreg.

Usage:
    python debug/resource_monitor.py [--interval 2] [--out usage.csv]
Stop with Ctrl-C (or kill); prints a summary and leaves the CSV.
"""

import argparse
import csv
import re
import signal
import subprocess
import sys
import time

try:
    import psutil
except ImportError:
    sys.exit("pip install psutil first")


def gpu_stats():
    """Parse AGXAccelerator PerformanceStatistics from ioreg."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return {}
    stats = {}
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    if m:
        stats["gpu_util_pct"] = int(m.group(1))
    m = re.search(r'"Renderer Utilization %"=(\d+)', out)
    if m:
        stats["gpu_renderer_pct"] = int(m.group(1))
    m = re.search(r'"In use system memory"=(\d+)', out)
    if m:
        stats["gpu_mem_gb"] = round(int(m.group(1)) / 2**30, 2)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--out", default="usage.csv")
    args = ap.parse_args()

    ncores = psutil.cpu_count()
    rows = []
    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))

    fields = (["t", "gpu_util_pct", "gpu_renderer_pct", "gpu_mem_gb",
               "ram_used_gb", "swap_used_gb"] + [f"cpu{i}" for i in range(ncores)])
    f = open(args.out, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    t0 = time.time()
    psutil.cpu_percent(percpu=True)  # prime
    try:
        while not stop:
            time.sleep(args.interval)
            row = {"t": round(time.time() - t0, 1)}
            row.update(gpu_stats())
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            row["ram_used_gb"] = round(vm.used / 2**30, 1)
            row["swap_used_gb"] = round(sw.used / 2**30, 1)
            for i, pct in enumerate(psutil.cpu_percent(percpu=True)):
                row[f"cpu{i}"] = pct
            rows.append(row)
            writer.writerow(row)
            f.flush()
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        if rows:
            n = len(rows)
            gpu = [r.get("gpu_util_pct", 0) for r in rows]
            print(f"\n=== {n} samples over {rows[-1]['t']:.0f}s -> {args.out} ===")
            print(f"GPU util: mean {sum(gpu)/n:.0f}%  max {max(gpu)}%")
            for i in range(ncores):
                c = [r[f"cpu{i}"] for r in rows]
                bar = "#" * int(sum(c) / n / 5)
                print(f"cpu{i:2d}: mean {sum(c)/n:5.1f}%  max {max(c):5.1f}%  {bar}")
            sw = [r["swap_used_gb"] for r in rows]
            ram = [r["ram_used_gb"] for r in rows]
            print(f"RAM used: max {max(ram):.1f} GB | swap: max {max(sw):.1f} GB")


if __name__ == "__main__":
    main()
