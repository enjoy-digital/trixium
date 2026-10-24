#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""firmware/seq.c seq_desc_engine_at: the engine descriptor encoder, compiled for the host.

The multi-row matvec fields (doc/plan-multirow.md: nr in w1[14:13], xstride in w14) are
optional: with nr = xstride = 0 every word must be the single-row encoding, bit for bit (old
images and list-cache entries stay valid). The single-row reference below is the encoder as it
was before the fields existed; the sequencer test helper (_desc_engine) must agree with it."""

import re
import random
import shutil
import subprocess
import ctypes as ct

import pytest

from common import ROOT

# Fields in the harness's order (struct engine_op members).
FIELDS = ("kind rows rem ilv supw cols wbase wlen sbeats cbeats pk bias rshift rshift_rpre qn xred "
          "qstride rshift2 xsrc dst chan rep dstride sstride nr xstride").split()

WIDTHS = dict(kind=3, rows=16, rem=5, ilv=1, supw=1, cols=14, wbase=32, wlen=24, sbeats=4,
              cbeats=8, pk=1, bias=7, rshift=6, rshift_rpre=1, qn=1, xred=1, qstride=16,
              rshift2=6, xsrc=24, dst=24, chan=5, rep=5, dstride=24, sstride=8, nr=2, xstride=12)

# Helpers ------------------------------------------------------------------------------------------

def single_row_words(f):
    """The engine descriptor before the multi-row fields (frozen copy of seq_desc_engine_at)."""
    d = [0]*16
    d[0]  = 1
    d[1]  = (f["kind"] | (f["ilv"] << 8) | (f["supw"] << 9) | (f["rshift_rpre"] << 10)
             | (f["qn"] << 11) | (f["xred"] << 12))
    d[2]  = f["rows"] | (f["rem"] << 16)
    d[3]  = f["cols"]
    d[4]  = f["wbase"]
    d[5]  = f["wlen"]
    d[6]  = f["sbeats"] | (f["pk"] << 4) | (f["cbeats"] << 8) | (f["qstride"] << 16)
    d[7]  = f["bias"] | (f["rshift"] << 16) | (f["rshift2"] << 22)
    d[8]  = f["xsrc"]
    d[9]  = f["dst"]
    d[10] = f["chan"]
    d[11] = f["rep"]
    d[12] = f["dstride"]
    d[13] = f["sstride"]
    return [w & 0xFFFFFFFF for w in d]

def _random_fields(rng, multirow):
    f = {k: rng.getrandbits(w) for k, w in WIDTHS.items()}
    if not multirow:
        f["nr"] = f["xstride"] = 0
    return f

@pytest.fixture(scope="module")
def encode(tmp_path_factory):
    cc = shutil.which("gcc")
    if not cc:
        pytest.skip("C compiler required")
    src = (ROOT / "firmware/seq.c").read_text()
    m = re.search(r"^void seq_desc_engine_at\(volatile uint32_t \*d, const struct engine_op \*op\)\n"
                  r"\{.*?^\}\n", src, re.S | re.M)
    assert m, "seq_desc_engine_at not found in firmware/seq.c"
    harness = "\n".join([
        '#include "engine.h"',
        m.group(0),
        "void encode(const uint32_t *f, uint32_t *out)",
        "{",
        "    struct engine_op op = {0};",
        *[f"    op.{k} = f[{i}];" for i, k in enumerate(FIELDS)],
        "    volatile uint32_t d[16];",
        "    for (int i = 0; i < 16; i++) d[i] = 0xdeadbeefu;",
        "    seq_desc_engine_at(d, &op);",
        "    for (int i = 0; i < 16; i++) out[i] = d[i];",
        "}",
    ])
    tmp = tmp_path_factory.mktemp("engine-desc")
    (tmp / "harness.c").write_text(harness)
    lib = tmp / "harness.so"
    r = subprocess.run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                        "-I", str(ROOT / "firmware"), str(tmp / "harness.c"), "-o", str(lib)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    dll = ct.CDLL(str(lib))
    dll.encode.argtypes = [ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint32)]
    def run(f):
        fin = (ct.c_uint32*len(FIELDS))(*[f[k] for k in FIELDS])
        out = (ct.c_uint32*16)()
        dll.encode(fin, out)
        return list(out)
    return run

# Tests --------------------------------------------------------------------------------------------

def test_single_row_encoding_unchanged(encode):
    """nr = xstride = 0: byte-identical to the single-row encoder, every word written."""
    rng = random.Random(7)
    cases = [{k: 0 for k in WIDTHS}, {k: (1 << w) - 1 for k, w in WIDTHS.items()}]
    cases[1]["nr"] = cases[1]["xstride"] = 0
    cases += [_random_fields(rng, multirow=False) for _ in range(256)]
    for f in cases:
        assert encode(f) == single_row_words(f), f

def test_multirow_fields(encode):
    """nr lands in w1[14:13], xstride in w14[11:0]; every other bit as the single-row op."""
    rng = random.Random(8)
    for _ in range(256):
        f = _random_fields(rng, multirow=True)
        want = single_row_words(f)
        want[1]  |= f["nr"] << 13
        want[14]  = f["xstride"]
        assert encode(f) == want, f

def test_sequencer_helper_agrees(encode):
    """The sequencer tests' descriptor helper encodes the same words (fields it knows)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_sequencer",
        ROOT / "test/gateware/core/test_sequencer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _desc_engine = mod._desc_engine
    rng = random.Random(9)
    known = ("kind rows rem ilv supw cols wbase wlen sbeats cbeats rshift_rpre qn xred qstride "
             "rshift2 bias rshift xsrc dst chan nr xstride").split()
    for _ in range(64):
        f = {k: (rng.getrandbits(WIDTHS[k]) if k in known else 0) for k in WIDTHS}
        raw = _desc_engine(**{k: f[k] for k in known})
        words = [int.from_bytes(raw[4*i:4*i + 4], "little") for i in range(16)]
        assert words == encode(f), f
