# SPDX-License-Identifier: BSD-2-Clause
import pytest

from tools.benchmark_report import parse_record, read_records, summarize


RECORD = ("BENCH_PHASE v=1 valid=1 clock_hz=150000000 requested_tokens=3 generated_tokens=3 "
          "prompt_tokens=3 prefill_forwards=3 reset_cycles=100 prefill_cycles=900 "
          "ttft_cycles=1100 decode_tokens=2 decode_cycles=8589934652 "
          "decode_wall_cycles=8589934664 emit_cycles=21 total_cycles=8589936000 "
          "context_capacity=128 context_used=5")


def test_wide_cycles_and_decode_denominator():
    r = parse_record(RECORD)
    assert r["decode_wall_tok_s"] == 2 * 150000000 / 8589934664
    assert r["ttft_s"] == 1100 / 150000000
    assert r["decode_excluding_emit_tok_s"] > r["decode_wall_tok_s"]
    assert r["request_complete"]


@pytest.mark.parametrize("old,new", [("v=1", "v=2"), ("valid=1", "valid=0"),
                                    ("decode_tokens=2", "decode_tokens=3"),
                                    ("clock_hz=150000000", "clock_hz=0"),
                                    ("context_capacity=128", "context_capacity=4"),
                                    ("context_used=5", "context_used=2"),
                                    ("ttft_cycles=1100", "ttft_cycles=99"),
                                    ("prefill_forwards=3", "prefill_forwards=2"),
                                    ("requested_tokens=3", "requested_tokens=2"),
                                    ("decode_cycles=8589934652", "decode_cycles=0"),
                                    ("decode_wall_cycles=8589934664", "decode_wall_cycles=10"),
                                    ("total_cycles=8589936000", "total_cycles=500")])
def test_bad_measurements_fail(old, new):
    with pytest.raises(ValueError):
        parse_record(RECORD.replace(old, new))


def test_legacy_bench_not_accepted(tmp_path):
    path = tmp_path / "legacy.log"
    path.write_text("BENCH: 2 tokens, 2700000000 cycles total (0.110 tok/s)\n")
    with pytest.raises(ValueError, match="not decode-only"):
        read_records(path)


def test_log_and_grouping(tmp_path):
    path = tmp_path / "bench.log"
    path.write_text("console chatter\n" + RECORD + "\n" + RECORD + "\n")
    rows = read_records(path)
    assert summarize(rows)[0]["repetitions"] == 2
    rows.append(parse_record(RECORD.replace("context_capacity=128", "context_capacity=64")))
    assert len(summarize(rows)) == 2


def test_single_generated_token_has_no_decode_rate():
    record = (RECORD.replace("generated_tokens=3", "generated_tokens=1")
              .replace("decode_tokens=2", "decode_tokens=0")
              .replace("decode_cycles=8589934652", "decode_cycles=0")
              .replace("decode_wall_cycles=8589934664", "decode_wall_cycles=0"))
    r = parse_record(record)
    assert r["decode_wall_tok_s"] is None and not r["request_complete"]
    assert summarize([r])[0]["median_decode_wall_tok_s"] is None


def test_missing_and_duplicate_fields():
    with pytest.raises(ValueError, match="missing"):
        parse_record("BENCH_PHASE v=1")
    with pytest.raises(ValueError, match="duplicate"):
        parse_record(RECORD + " v=1")
