#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Compiled (Verilator) driver for any resident scratchpad client.

A client exposes `cmd` (a stream.Endpoint), cancel/busy/done/error and the one-cycle
scratchpad ports (sp_rd_*, sp_wr_*). `build_client()` compiles it (common.verilator: cached)
with a C++ driver generated from its command layout; `run_client()` runs a whole scratchpad
image through a list of commands - the SpClient model (test/common/rtl/sp_client.h): reads
answered one clock later, a random write grant whose payload must hold while ungranted - and
returns the final image and one stats line per command. Rejected commands (expected error
bits != 0) must never touch memory; they are cancelled.
"""

import tempfile
from pathlib import Path

import numpy as np

from common.verilator import Simulator

_CPP = r'''
#include "sim_top.h"
#include "sp_client.h"
int main(int argc, char **argv) {{
    sim_require(argc == 5, "usage: image commands out grant_percent");
    std::vector<uint8_t> image = sim_load(argv[1]), cmdb = sim_load(argv[2]);
    sim_require(image.size() % 32 == 0 && cmdb.size() % 4 == 0, "file length");
    std::vector<uint32_t> cmds(cmdb.size()/4);
    std::memcpy(cmds.data(), cmdb.data(), cmdb.size());
    const uint32_t NF = {nfields} + 1, grant_pct = (uint32_t)std::atoi(argv[4]);
    sim_require(cmds.size() % NF == 0, "shapes");
    VerilatedContext ctx;
    SIM_TOP d(&ctx);
    Sim sim;
    sim_bind(sim, d);
    SpClient sp(sim, "sp", image.size(), grant_pct);
    sp.mem = image;
    sim.models.push_back(&sp);
    sim.max_cycles = ~0ull;
    sp.force_grant = true;
    sim.reset(2);
    for (size_t c = 0; c < cmds.size(); c += NF) {{
        const uint32_t *r = &cmds[c];
{assign}
        d.cmd_valid = 1; sim.settle(); sim_require(d.cmd_ready, "command not accepted");
        sim.tick(); d.cmd_valid = 0;
{scramble}
        sp.force_grant = false;
        const uint64_t start = sim.cycle; bool done = false; const uint32_t want = r[NF - 1];
        for (uint64_t n = 0; n < 4000000 && !done; n++) {{
            sim.tick();
            if (want) {{ sim_require(!d.done && !d.sp_wr_req, "rejected command touched memory"); if (n == 40) break; }}
            else if (d.error) {{ std::fprintf(stderr, "cmd %zu: unexpected error %u\n", c/NF, d.error); return 1; }}
            done = d.done;
        }}
        sp.force_grant = true;
        if (want) {{
            if (d.error != want) {{ std::fprintf(stderr, "cmd %zu: error %u, want %u\n", c/NF, d.error, want); return 1; }}
            sim_require(d.busy, "failed command must stay busy");
            d.cancel = 1; sim.tick(); d.cancel = 0; sim.tick(); sim_require(!d.busy && !d.error, "cancel");
            std::printf("cmd %zu rejected %u\n", c/NF, want);
        }} else {{
            sim_require(done, "timeout");
            std::printf("cmd %zu cycles %llu reads %u writes %u\n", c/NF, (unsigned long long)(sim.cycle - start),
                        d.read_beats, d.output_beats);
        }}
    }}
    sim_save(argv[3], sp.mem.data(), sp.mem.size());
    sim.finish();
    return 0;
}}
'''


def build_client(factory, layout, top, key):
    """The compiled client (a common.verilator.Simulator), cached under build/verilator/ by
    `key`: `factory` builds the client on a cache miss only (a hit neither constructs nor
    converts it)."""
    return Simulator(lambda: _construct(factory(), layout), name=top, driver=_driver(layout),
                     key=f"rtl_client:{key}")


def _construct(dut, layout):
    signals = {name: getattr(dut, name) for name in (
        "cancel", "busy", "done", "error", "read_beats", "output_beats",
        "sp_rd_req", "sp_rd_adr", "sp_rd_dat", "sp_wr_req", "sp_wr_adr", "sp_wr_dat", "sp_wr_we", "sp_wr_gnt")}
    for name, _ in layout:
        signals["cmd_" + name] = getattr(dut.cmd, name)
    signals.update(cmd_valid=dut.cmd.valid, cmd_ready=dut.cmd.ready)
    for name, signal in signals.items():
        signal.name_override = name
    return dict(dut=dut, ios=signals)


def _driver(layout):
    assign = "\n".join(f"        d.cmd_{name} = r[{i}];" for i, (name, _) in enumerate(layout))
    scramble = "\n".join(f"        d.cmd_{name} = (~r[{i}]) & 0x{(1 << w) - 1:x}u;"
                         for i, (name, w) in enumerate(layout))
    return _CPP.format(nfields=len(layout), assign=assign, scramble=scramble)


def run_client(sim, layout, mem, commands, tmp_path, grant_pct=75):
    """commands: [(dict of fields, want_error)]. Returns (final image, stdout lines)."""
    with tempfile.TemporaryDirectory(dir=tmp_path) as tmp:
        tmp = Path(tmp)
        image, cmdfile, out = tmp / "image.bin", tmp / "commands.bin", tmp / "out.bin"
        np.asarray(mem, dtype="<u4").tofile(image)
        rec = []
        for fields, err in commands:
            rec += [int(fields.get(name, 0)) & ((1 << 32) - 1) for name, _ in layout] + [err]
        np.array(rec, dtype="<u4").tofile(cmdfile)
        stdout = sim.run(args=[image, cmdfile, out, grant_pct], timeout=1800).stdout
        return np.fromfile(out, dtype="<u4"), stdout.splitlines()
