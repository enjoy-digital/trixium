#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Compiled (Verilator) driver for any resident scratchpad client.

A client exposes `cmd` (a stream.Endpoint), cancel/busy/done/error and the one-cycle
scratchpad ports (sp_rd_*, sp_wr_*). `build_client()` emits its Verilog and a C++ driver
generated from its command layout; `run_client()` runs a whole scratchpad image through a
list of commands - reads answered one clock later, a random write grant whose payload must
hold while ungranted - and returns the final image and one stats line per command.
Rejected commands (expected error bits != 0) must never touch memory; they are cancelled.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np

from migen.fhdl import verilog

_CPP = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "V{top}.h"
static void require(bool ok, const char *what) {{ if (!ok) {{ std::fprintf(stderr, "%s\n", what); std::exit(1); }} }}
static std::vector<uint32_t> load(const char *path) {{
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {{}});
    require(b.size() % 4 == 0, "file length");
    std::vector<uint32_t> w(b.size()/4); std::memcpy(w.data(), b.data(), b.size()); return w;
}}
int main(int argc, char **argv) {{
    require(argc == 5, "usage: image commands out grant_percent");
    std::vector<uint32_t> mem = load(argv[1]), cmds = load(argv[2]);
    const uint32_t NF = {nfields} + 1, grant_pct = (uint32_t)std::atoi(argv[4]);
    require(mem.size() % 8 == 0 && cmds.size() % NF == 0, "shapes");
    const uint32_t beats = mem.size()/8;
    V{top} d;
    uint32_t state = 777; auto rnd = [&]() {{ state = state*1664525u + 1013904223u; return state >> 8; }};
    uint32_t pending[8] = {{0}}; bool held = false; uint32_t ha = 0, hw = 0, hd[8] = {{0}};
    uint64_t cycle = 0;
    auto tick = [&](bool grant) {{
        d.sp_wr_gnt = grant; d.sys_clk = 0; d.eval();
        if (held) require(d.sp_wr_req && d.sp_wr_adr == ha && d.sp_wr_we == hw &&
                          !std::memcmp(hd, &d.sp_wr_dat[0], 32), "write payload changed while ungranted");
        if (d.sp_wr_req && grant) {{
            require(d.sp_wr_adr < beats, "write out of range");
            for (int l = 0; l < 8; l++) if (d.sp_wr_we >> l & 1) mem[8*d.sp_wr_adr + l] = d.sp_wr_dat[l];
        }}
        held = d.sp_wr_req && !grant; ha = d.sp_wr_adr; hw = d.sp_wr_we; std::memcpy(hd, &d.sp_wr_dat[0], 32);
        bool rd = d.sp_rd_req;
        if (rd) {{ require(d.sp_rd_adr < beats, "read out of range"); for (int l = 0; l < 8; l++) pending[l] = mem[8*d.sp_rd_adr + l]; }}
        d.sys_clk = 1; d.eval();
        for (int l = 0; l < 8; l++) d.sp_rd_dat[l] = rd ? pending[l] : 0xA5A5A5A5u;
        cycle++;
    }};
    d.sys_rst = 1; tick(true); tick(true); d.sys_rst = 0;
    for (size_t c = 0; c < cmds.size(); c += NF) {{
        const uint32_t *r = &cmds[c];
{assign}
        d.cmd_valid = 1; d.sys_clk = 0; d.eval(); require(d.cmd_ready, "command not accepted");
        tick(true); d.cmd_valid = 0;
{scramble}
        const uint64_t start = cycle; bool done = false; const uint32_t want = r[NF - 1];
        for (uint64_t n = 0; n < 4000000 && !done; n++) {{
            tick((rnd() % 100) < grant_pct);
            if (want) {{ require(!d.done && !d.sp_wr_req, "rejected command touched memory"); if (n == 40) break; }}
            else if (d.error) {{ std::fprintf(stderr, "cmd %zu: unexpected error %u\n", c/NF, d.error); return 1; }}
            done = d.done;
        }}
        if (want) {{
            if (d.error != want) {{ std::fprintf(stderr, "cmd %zu: error %u, want %u\n", c/NF, d.error, want); return 1; }}
            require(d.busy, "failed command must stay busy");
            d.cancel = 1; tick(true); d.cancel = 0; tick(true); require(!d.busy && !d.error, "cancel");
            std::printf("cmd %zu rejected %u\n", c/NF, want);
        }} else {{
            require(done, "timeout");
            std::printf("cmd %zu cycles %llu reads %u writes %u\n", c/NF, (unsigned long long)(cycle - start),
                        d.read_beats, d.output_beats);
        }}
    }}
    std::ofstream o(argv[3], std::ios::binary); o.write(reinterpret_cast<const char *>(mem.data()), 4*mem.size());
    return 0;
}}
'''


def build_client(dut, layout, tmp_path, top):
    verilator = shutil.which("verilator")
    if not verilator:
        import pytest
        pytest.skip("Verilator required")
    signals = {name: getattr(dut, name) for name in (
        "cancel", "busy", "done", "error", "read_beats", "output_beats",
        "sp_rd_req", "sp_rd_adr", "sp_rd_dat", "sp_wr_req", "sp_wr_adr", "sp_wr_dat", "sp_wr_we", "sp_wr_gnt")}
    for name, _ in layout:
        signals["cmd_" + name] = getattr(dut.cmd, name)
    signals.update(cmd_valid=dut.cmd.valid, cmd_ready=dut.cmd.ready)
    for name, signal in signals.items():
        signal.name_override = name
    rtl = tmp_path / f"{top}.v"
    verilog.convert(dut, ios=set(signals.values()), name=top).write(str(rtl))
    assign = "\n".join(f"        d.cmd_{name} = r[{i}];" for i, (name, _) in enumerate(layout))
    scramble = "\n".join(f"        d.cmd_{name} = (~r[{i}]) & 0x{(1 << w) - 1:x}u;"
                         for i, (name, w) in enumerate(layout))
    cpp = tmp_path / f"{top}.cpp"
    cpp.write_text(_CPP.format(top=top, nfields=len(layout), assign=assign, scramble=scramble))
    built = subprocess.run([verilator, "--cc", "--exe", "--build", "-j", "4", "-Wno-fatal", "-O3",
                            "--Mdir", str(tmp_path / "obj"), "--top-module", top, str(rtl), str(cpp)],
                           capture_output=True, text=True, timeout=900)
    assert built.returncode == 0, built.stdout[-4000:] + built.stderr[-4000:]
    return tmp_path / f"obj/V{top}"


def run_client(binary, layout, mem, commands, tmp_path, grant_pct=75):
    """commands: [(dict of fields, want_error)]. Returns (final image, stdout lines)."""
    image, cmdfile, out = tmp_path / "image.bin", tmp_path / "commands.bin", tmp_path / "out.bin"
    np.asarray(mem, dtype="<u4").tofile(image)
    rec = []
    for fields, err in commands:
        rec += [int(fields.get(name, 0)) & ((1 << 32) - 1) for name, _ in layout] + [err]
    np.array(rec, dtype="<u4").tofile(cmdfile)
    run = subprocess.run([str(binary), str(image), str(cmdfile), str(out), str(grant_pct)],
                         capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stdout + run.stderr[-4000:]
    return np.fromfile(out, dtype="<u4"), run.stdout.splitlines()
