#!/usr/bin/env python3
"""Gather ATT stats_*.csv files into a single opcode-count CSV.

Expects a layout of one subdirectory per benchmark:

    <root>/
        BE_SP_INT_ADD/
            stats_kernel0.csv       <- found at any depth
            disasm.txt              <- optional, used when the CSV has no
                                       instruction column
        LDS_U_8/
            .../stats_*.csv

Writes one row per (benchmark, opcode) with the summed hitcount and that
opcode's share of the benchmark's traced instructions.

Usage:
    ./stats_to_opcode_counts.py <root> [-o opcode_counts.csv] [--top 10]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

STATS_GLOB = "stats*.csv"
DISASM_NAME = "disasm.txt"

# rocprofv3 column names have drifted across releases; accept any spelling.
ADDRESS_COLUMNS = ("Vaddr", "Addr", "Address", "Offset")
HITCOUNT_COLUMNS = ("Hitcount", "Hit_Count", "Count")
INSTRUCTION_COLUMNS = ("Instruction", "Inst", "Disassembly")

# llvm-objdump: "\t3f30: 0a 20 04 68 \tv_add_u32_e32 v16, s2, v0"
DISASM_LINE = re.compile(r"^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]{2}\s+)+\s*(\S+)")

UNRESOLVED = re.compile(r"^<0x[0-9a-f]+>$")


def first_present(row: dict, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def parse_address(raw: str) -> int | None:
    text = raw.strip()
    for base in (16, 10) if text.lower().startswith("0x") else (10, 16):
        try:
            return int(text, base)
        except ValueError:
            continue
    return None


def load_disassembly(path: Path, offset: int = 0) -> dict[int, str]:
    """Map instruction address -> mnemonic, shifted by `offset`."""
    table: dict[int, str] = {}
    if not path.is_file():
        return table
    with path.open(errors="replace") as handle:
        for line in handle:
            match = DISASM_LINE.match(line)
            if match:
                table[int(match.group(1), 16) + offset] = match.group(2)
    return table


def opcode_for(row: dict, address: int | None, disasm: dict[int, str]) -> str:
    text = first_present(row, INSTRUCTION_COLUMNS)
    if text:
        return text.split()[0]
    if address is not None and address in disasm:
        return disasm[address]
    return f"<0x{address:x}>" if address is not None else "<unknown>"


def read_benchmark(directory: Path, offset: int) -> tuple[dict[str, float], int]:
    """Return (opcode -> hitcount, number of stats files read)."""
    disasm = load_disassembly(directory / DISASM_NAME, offset)
    counts: dict[str, float] = defaultdict(float)
    files = 0

    for path in sorted(directory.rglob(STATS_GLOB)):
        files += 1
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                raw_hits = first_present(row, HITCOUNT_COLUMNS)
                if raw_hits is None:
                    continue
                try:
                    hits = float(raw_hits)
                except ValueError:
                    continue
                raw_addr = first_present(row, ADDRESS_COLUMNS)
                address = parse_address(raw_addr) if raw_addr else None
                counts[opcode_for(row, address, disasm)] += hits

    return counts, files


def collect(root: Path, offset: int) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        counts, files = read_benchmark(directory, offset)
        if files == 0:
            continue
        if not counts:
            print(f"warning: {directory.name}: stats files had no usable rows",
                  file=sys.stderr)
            continue
        results[directory.name] = counts
    return results


def unresolved_share(counts: dict[str, float]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    bad = sum(v for k, v in counts.items() if UNRESOLVED.match(k))
    return bad / total


def write_csv(results: dict[str, dict[str, float]], output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "opcode", "hitcount", "fraction"])
        for benchmark in sorted(results):
            counts = results[benchmark]
            total = sum(counts.values()) or 1.0
            for opcode, hits in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                writer.writerow(
                    [benchmark, opcode, f"{hits:.0f}", f"{hits / total:.6f}"]
                )


def print_summary(results: dict[str, dict[str, float]], top: int) -> None:
    for benchmark in sorted(results):
        counts = results[benchmark]
        total = sum(counts.values())
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        width = max((len(op) for op, _ in ranked), default=8)

        print(f"{benchmark}  ({total:,.0f} traced instructions, "
              f"{len(counts)} distinct opcodes)")
        for opcode, hits in ranked:
            print(f"    {opcode.ljust(width)}  {hits:14,.0f}  "
                  f"{hits / total if total else 0:6.1%}")
        if len(counts) > top:
            print(f"    ... {len(counts) - top} more")
        print()


def warn_unresolved(results: dict[str, dict[str, float]], threshold: float) -> None:
    offenders = [
        (name, share)
        for name, counts in results.items()
        if (share := unresolved_share(counts)) > threshold
    ]
    if not offenders:
        return
    print("warning: addresses could not be mapped to opcodes for:", file=sys.stderr)
    for name, share in sorted(offenders, key=lambda kv: -kv[1]):
        print(f"  {name}: {share:.0%} of hitcount unresolved", file=sys.stderr)
    print(
        "  the stats CSV has no instruction column and disasm.txt is missing,\n"
        "  targets the wrong architecture, or uses a different address base\n"
        "  (try --offset to shift the disassembly addresses).",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="directory of per-benchmark subdirs")
    parser.add_argument("-o", "--output", type=Path,
                        help="output CSV (default: <root>/opcode_counts.csv)")
    parser.add_argument("--top", type=int, default=10,
                        help="opcodes to print per benchmark (default: 10)")
    parser.add_argument("--offset", type=lambda s: int(s, 0), default=0,
                        help="add this to disassembly addresses before joining")
    parser.add_argument("--quiet", action="store_true", help="skip the console summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.is_dir():
        print(f"error: no such directory: {args.root}", file=sys.stderr)
        return 1

    results = collect(args.root, args.offset)
    if not results:
        print(f"error: found no {STATS_GLOB} in any subdirectory of {args.root}",
              file=sys.stderr)
        return 1

    output = args.output or args.root / "opcode_counts.csv"
    write_csv(results, output)
    print(f"wrote {output}  ({len(results)} benchmarks)")

    warn_unresolved(results, threshold=0.10)

    if not args.quiet:
        print()
        print_summary(results, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())