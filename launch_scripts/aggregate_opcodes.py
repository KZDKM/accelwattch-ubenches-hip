#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

STATS_GLOB = "*stats*.csv"
DISASM_NAME = "disasm.txt"

ADDRESS_COLUMNS = ("Vaddr", "Addr", "Address", "Offset")
HITCOUNT_COLUMNS = ("Hitcount", "Hit_Count", "Count")
INSTRUCTION_COLUMNS = ("Instruction", "Inst", "Disassembly")

# llvm-objdump line: "\t3f30: 0a 20 04 68 \tv_add_u32_e32 v16, s2, v0"
DISASM_LINE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]{2}\s+)+\s*(\S+)"
)


def first_present(row: dict, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def parse_address(raw: str) -> int | None:
    text = raw.strip()
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text)
    except ValueError:
        try:
            return int(text, 16)
        except ValueError:
            return None


def load_disassembly(path: Path) -> dict[int, str]:
    table: dict[int, str] = {}
    if not path.is_file():
        return table
    with path.open(errors="replace") as handle:
        for line in handle:
            match = DISASM_LINE.match(line)
            if match:
                table[int(match.group(1), 16)] = match.group(2)
    return table


# finds the corresponding opcode from disasm by address
def opcode_for(row: dict, address: int | None, disasm: dict[int, str]) -> str:
    text = first_present(row, INSTRUCTION_COLUMNS)
    if text:
        return text.split()[0]
    if address is not None and address in disasm:
        return disasm[address]
    return f"<0x{address:x}>" if address is not None else "<unknown>"


# return opcode count
def read_ubench(directory: Path) -> tuple[dict[str, float], dict[int, float]]:
    disasm = load_disassembly(directory / DISASM_NAME)
    by_opcode: dict[str, float] = defaultdict(float)
    by_address: dict[int, float] = defaultdict(float)

    for path in sorted(directory.glob(f"**/{STATS_GLOB}")):
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
                by_opcode[opcode_for(row, address, disasm)] += hits
                if address is not None:
                    by_address[address] += hits

    return by_opcode, by_address


def load_results(
    results_dir: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[int, float]]]:
    opcodes: dict[str, dict[str, float]] = {}
    addresses: dict[str, dict[int, float]] = {}
    for directory in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        by_opcode, by_address = read_ubench(directory)
        if by_opcode:
            opcodes[directory.name] = by_opcode
            addresses[directory.name] = by_address
    return opcodes, addresses


def write_long_csv(results: dict[str, dict[str, float]], output: Path) -> None:
    """One row per (ubench, opcode) -- long format, easy to pivot."""
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ubench", "opcode", "hitcount", "fraction"])
        for ubench in sorted(results):
            counts = results[ubench]
            total = sum(counts.values()) or 1.0
            for opcode, hits in sorted(
                counts.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                writer.writerow([ubench, opcode, f"{hits:.0f}", f"{hits / total:.6f}"])


def write_address_csv(results: dict[str, dict[int, float]], output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ubench", "address", "hitcount"])
        for ubench in sorted(results):
            for address, hits in sorted(results[ubench].items()):
                writer.writerow([ubench, f"0x{address:x}", f"{hits:.0f}"])


def print_summary(results: dict[str, dict[str, float]], top: int) -> None:
    for ubench in sorted(results):
        counts = results[ubench]
        total = sum(counts.values())
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        width = max((len(op) for op, _ in ranked), default=8)

        print(f"{ubench}  (total traced instructions: {total:,.0f})")
        for opcode, hits in ranked:
            share = hits / total if total else 0.0
            print(f"    {opcode.ljust(width)}  {hits:14,.0f}  {share:6.1%}")
        if len(counts) > top:
            print(f"    ... {len(counts) - top} more opcodes")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV to write (default: RESULTS_DIR/opcode_counts.csv)",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="opcodes to show per ubench (default: 10)"
    )
    parser.add_argument(
        "--by-address",
        action="store_true",
        help="also write a per-instruction-address CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir
    if not results_dir.is_dir():
        print(f"error: no such directory: {results_dir}", file=sys.stderr)
        return 1

    opcodes, addresses = load_results(results_dir)
    if not opcodes:
        print(
            f"error: found no {STATS_GLOB} under {results_dir}; "
            "check the ATT output layout",
            file=sys.stderr,
        )
        return 1

    output = args.output or results_dir / "opcode_counts.csv"
    write_long_csv(opcodes, output)
    print(f"wrote {output}")

    if args.by_address:
        addr_output = output.with_name(output.stem + "_by_address.csv")
        write_address_csv(addresses, addr_output)
        print(f"wrote {addr_output}")

    print()
    print_summary(opcodes, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())