#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Parse BENCH_PHASE v1 logs without mistaking mixed bench throughput for decode.

  python3 -m software.lab.benchmark build/bench-*.log

Each row is one request, not a claim about a filled 16k/64k context. Startup/model
loading and per-token p95 are not measured by this firmware record.
"""

import json
import argparse
import statistics
from pathlib import Path

# BENCH_PHASE Records ------------------------------------------------------------------------------

FIELDS = set("v valid clock_hz requested_tokens generated_tokens prompt_tokens "
             "prefill_forwards reset_cycles prefill_cycles ttft_cycles decode_tokens "
             "decode_cycles decode_wall_cycles emit_cycles total_cycles "
             "context_capacity context_used".split())


def parse_record(line):
    fields = {}
    for item in line.split()[1:]:
        key, separator, value = item.partition("=")
        if not separator or key in fields or not value.isdecimal():
            raise ValueError(f"invalid/duplicate benchmark field: {item}")
        fields[key] = int(value)
    missing = FIELDS - fields.keys()
    if missing:
        raise ValueError(f"missing benchmark fields: {sorted(missing)}")
    if fields["v"] != 1 or fields["valid"] != 1 or not fields["clock_hz"]:
        raise ValueError("unsupported, failed or zero-clock benchmark")
    if not 0 < fields["generated_tokens"] <= fields["requested_tokens"]:
        raise ValueError("inconsistent generated token count")
    if fields["decode_tokens"] != fields["generated_tokens"] - 1:
        raise ValueError("first emitted token must not count as a decode step")
    if not 0 < fields["prompt_tokens"] <= fields["context_used"] <= fields["context_capacity"]:
        raise ValueError("invalid context occupancy/capacity")
    if fields["prefill_forwards"] != fields["prompt_tokens"]:
        raise ValueError("v1 requires one prefill forward per prompt token")
    if not fields["reset_cycles"] + fields["prefill_cycles"] <= fields["ttft_cycles"] <= fields["total_cycles"]:
        raise ValueError("inconsistent prefill/TTFT timing")
    if not fields["decode_cycles"] <= fields["decode_wall_cycles"] <= fields["total_cycles"] - fields["ttft_cycles"]:
        raise ValueError("inconsistent decode timing")
    if fields["decode_tokens"] and not fields["decode_cycles"]:
        raise ValueError("decode tokens with no measured interval")
    if not fields["decode_tokens"] and (fields["decode_cycles"] or fields["decode_wall_cycles"]):
        raise ValueError("decode interval with no decode tokens")
    hz = fields["clock_hz"]
    return dict(fields,
                request_complete=fields["generated_tokens"] == fields["requested_tokens"],
                prefill_s=fields["prefill_cycles"] / hz,
                ttft_s=fields["ttft_cycles"] / hz,
                decode_wall_tok_s=(fields["decode_tokens"] * hz / fields["decode_wall_cycles"]
                                   if fields["decode_wall_cycles"] else None),
                decode_excluding_emit_tok_s=(fields["decode_tokens"] * hz / fields["decode_cycles"]
                                             if fields["decode_cycles"] else None))


def read_records(path):
    records = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if line.startswith("BENCH_PHASE "):
            try:
                records.append(dict(parse_record(line), source=str(path), line=lineno))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no BENCH_PHASE records; legacy BENCH is not decode-only")
    return records


def summarize(records):
    # Do not average different contexts, clocks or partial requests into one speed.
    groups = {}
    for r in records:
        key = tuple(r[k] for k in ("clock_hz", "prompt_tokens", "generated_tokens",
                                  "requested_tokens", "context_capacity", "context_used"))
        groups.setdefault(key, []).append(r)
    return [dict(clock_hz=key[0], prompt_tokens=key[1], generated_tokens=key[2],
                 requested_tokens=key[3], context_capacity=key[4], context_used=key[5],
                 repetitions=len(rows), median_ttft_s=statistics.median(r["ttft_s"] for r in rows),
                 median_decode_wall_tok_s=(statistics.median(r["decode_wall_tok_s"] for r in rows)
                                           if rows[0]["decode_tokens"] else None))
            for key, rows in groups.items()]

# Run ----------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Benchmark console logs.")
    args = parser.parse_args()
    try:
        records = [r for path in args.logs for r in read_records(path)]
    except (OSError, ValueError) as exc:
        parser.exit(1, f"benchmark: {exc}\n")
    print(json.dumps(dict(schema_version=1, records=records, groups=summarize(records),
                          not_measured=["startup", "model_load", "per_token_p95"]), indent=2))


if __name__ == "__main__":
    main()
