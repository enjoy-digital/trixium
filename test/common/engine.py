#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The engine testbench shared by the engine, sequencer and scratchpad tests.

EngineBench is the scratchpad + MatVecEngine on N memory channels, optionally with the
loader, the vector unit, the sequencer and the CSR front-end, as the tests used to build it
by hand. It runs a common.verilator.Script on either backend:

- "verilator": the emitted Verilog, compiled (cached), the channels answered by the C++
  AXI slave (test/common/rtl/axi_slave.h). The default.
- "migen": Migen's simulator with HBMChannelModel channels - the slow cross-check.

Both see the same signal names (cmd_<field>, done, vu_prog, seq_busy, ...), the scratchpad as
memory "sp" and channel i's memory as "axi<i>".
"""

import hashlib

from migen import *

from litex.soc.interconnect.axi import AXIInterface

from gateware.core.scratchpad import Scratchpad
from gateware.core.engine import MatVecEngine
from gateware.memory.hbm_model import HBMChannelModel, words_from_bytes

from common.verilator import Script, Simulator, run_migen

# Constants ----------------------------------------------------------------------------------------

CMD_FIELDS = ("kind", "rows", "cols", "wbase", "wlen", "sbeats", "cbeats", "bias", "rshift",
              "xsrc", "dst", "chan")

# Engine DUT ---------------------------------------------------------------------------------------

class EngineDUT(Module):
    """Scratchpad + MatVecEngine (+ loader / vector unit / sequencer / CSR front).

    hbm: None leaves the AXI ports open (compiled backend: C++ slaves); otherwise one dict of
    HBMChannelModel arguments per channel (Migen backend). sp_init: scratchpad bytes (Migen
    backend; the compiled one loads an image).
    """
    def __init__(self, n_channels, xbuf_groups=32, chan_stride=0, chan_mod=0, chan_skew=0,
                 loader=False, loader_width=128, vecunit=False, seq=False, seed7=True,
                 front=False, overlap=False, sp_size=64*1024, hbm=None, sp_init=None):
        self.axis = axis = [AXIInterface(data_width=256, address_width=33, id_width=6)
                            for _ in range(n_channels)]
        init = words_from_bytes(sp_init) if sp_init is not None else None
        self.submodules.scratchpad = Scratchpad(size=sp_size, init=init)
        if vecunit:
            from gateware.core.vector import VectorUnit
            self.submodules.vu = VectorUnit()
        if seq:
            from gateware.core.sequencer import Sequencer
            self.submodules.seq = Sequencer()
        self.submodules.engine = MatVecEngine(self.scratchpad, axis, xbuf_groups=xbuf_groups,
            chan_stride=chan_stride, chan_mod=chan_mod, chan_skew=chan_skew,
            vecunit=self.vu if vecunit else None, seq=self.seq if seq else None)
        if front:
            from gateware.core.csr import EngineCSR
            self.submodules.front = EngineCSR(self.engine, seq=self.seq if seq else None)
        if seq:
            self.comb += [
                self.vu.seq_en.eq(self.seq.busy),
                self.vu.seq_start.eq(self.seq.vec_start),
                self.vu.seq_pc.eq(self.seq.vec_pc),
                self.seq.vec_done.eq(self.vu.done),
                self.seq.vec_res6.eq(self.vu.res6.status),
            ]
            self.comb += [self.vu.seq_args[i].eq(self.seq.vec_args[i]) for i in range(8)]
            if seed7:
                self.comb += self.vu.seq_seed7.eq(self.seq.vec_seed7)
        if overlap:
            # The SoC's sideband idle wiring when there is no sequencer: read grants are
            # units_idle-gated.
            self.comb += self.scratchpad.units_idle.eq(~self.engine.sp_busy & ~self.vu.busy)
        if loader:
            from gateware.memory.loader import Loader
            self.submodules.loader = Loader(axis, data_width=loader_width, chan_stride=chan_stride,
                                            chan_mod=chan_mod, chan_skew=chan_skew)
            self.comb += [
                self.engine.store_cmd.connect(self.loader.store_cmd),
                self.engine.store_data.connect(self.loader.store_data),
                self.engine.store_busy.eq(self.loader.store_busy),
            ]
        self.models = []
        for i, kw in enumerate(hbm or []):
            model = HBMChannelModel(axis[i], **kw)
            setattr(self.submodules, f"model{i}", model)
            self.models.append(model)
        self.signals = self._signals()

    def _signals(self):
        e   = self.engine
        sig = {f"cmd_{name}": getattr(e.cmd, name) for name, *_ in e.cmd.description.payload_layout}
        sig.update(cmd_valid=e.cmd.valid, cmd_ready=e.cmd.ready)
        for name in ("done", "error", "busy", "absmax", "argmax_val", "argmax_idx", "cyc_both",
                     "gw_ovf", "dbg_gw_we", "dbg_result_cnt", "dbg_mv_total", "dbg_ax_total"):
            if hasattr(e, name):         # gw_ovf: only with the vector unit's arbitration.
                sig[name] = getattr(e, name)
        for i, r in enumerate(e.readers):
            sig[f"reader{i}_idle"] = r.idle
        for i, p in enumerate(e.pipes):
            for f in ("valid", "row", "wide", "head"):
                sig[f"pipe{i}_{f}"] = getattr(p.source, f)
        sp = self.scratchpad
        for slot in range(2):
            for f in ("req", "adr", "sel", "bsel", "wdat", "ok"):
                sig[f"sbw{slot}_{f}"] = getattr(sp, f"_sbw_{f}")[slot]
            for f in ("req", "adr", "sel", "gnt"):
                sig[f"sb{slot}_{f}"] = getattr(sp, f"_sb_{f}")[slot]
        sig.update(sb_rdat=sp._sb_rdat, sbw_pending=sp.sbw_pending)
        if hasattr(self, "vu"):
            vu = self.vu
            for name in ("prog", "prog_adr", "pc", "submit") + tuple(f"arg{i}" for i in range(8)):
                sig[f"vu_{name}"]    = getattr(vu, name).storage
                sig[f"vu_{name}_re"] = getattr(vu, name).re
            sig.update(vu_busy=vu.busy, vu_done=vu.done)
        if hasattr(self, "seq"):
            s = self.seq
            for name in ("base", "count", "start"):
                sig[f"seq_{name}"]    = getattr(s, name).storage
                sig[f"seq_{name}_re"] = getattr(s, name).re
            sig.update(seq_busy=s.busy, seq_dcount=s.status.fields.dcount)
            for i in range(4):
                sig[f"seq_kreg{i}"] = s.kregs[i]
        if hasattr(self, "front"):
            f = self.front
            if not hasattr(self, "seq"):
                for name in CMD_FIELDS + ("submit",):
                    sig[f"front_{name}"]    = getattr(f, name).storage
                sig["front_submit_re"] = f.submit.re
                sig["front_pending"]   = f.status.fields.pending
            sig["front_status"] = f.status.status
        if hasattr(self, "loader"):
            l = self.loader
            for name in ("chan", "base", "length", "start"):
                sig[f"loader_{name}"] = getattr(l, name).storage
            sig.update(loader_start_re=l.start.re, loader_done=l.status.fields.done,
                       loader_sink_valid=l.sink.valid, loader_sink_ready=l.sink.ready,
                       loader_sink_data=l.sink.data)
        return sig

# Engine Bench -------------------------------------------------------------------------------------

_simulators = {}

class EngineBench:
    """An EngineDUT configuration that runs Scripts on either backend (see the module doc)."""
    def __init__(self, n_channels, **config):
        self.n_channels = n_channels
        self.config     = config

    def simulator(self):
        key = (self.n_channels, tuple(sorted(self.config.items())))
        if key not in _simulators:
            def build():
                dut = EngineDUT(self.n_channels, **self.config)
                return dict(dut=dut, ios=dut.signals, axi={f"axi{i}": a for i, a in enumerate(dut.axis)},
                            memories={"sp": dut.scratchpad.mem})
            name = "engine_n{}_{}".format(self.n_channels, hashlib.sha1(repr(key).encode()).hexdigest()[:8])
            _simulators[key] = Simulator(build, name=name, key=f"EngineDUT{key!r}")
        return _simulators[key]

    def run(self, backend, script, sp=b"", payloads=None, latency=8, stall_rate=0, reorder_ids=0,
            model_size=64*1024, axi_log=()):
        """Run `script` after a reset. payloads: per-channel memory images; latency/stall_rate:
        a value or a per-channel list (stall_rate in sixteenths, as HBMChannelModel)."""
        n   = self.n_channels
        lat = latency    if isinstance(latency,    (list, tuple)) else [latency]*n
        stl = stall_rate if isinstance(stall_rate, (list, tuple)) else [stall_rate]*n
        payloads = list(payloads or [b""]*n)
        sp = bytes(sp)
        full = Script()
        full.reset(2)
        full.ops += script.ops
        if backend == "migen":
            hbm = [dict(size=model_size, init=words_from_bytes(payloads[i]) if payloads[i] else None,
                        latency=lat[i], stall_rate=stl[i], reorder_ids=reorder_ids) for i in range(n)]
            dut = EngineDUT(n, hbm=hbm, sp_init=sp.ljust(self.config.get("sp_size", 64*1024), b"\0"),
                            **self.config)
            mems = {"sp": dut.scratchpad.mem}
            mems.update({f"axi{i}": m.mem for i, m in enumerate(dut.models)})
            return run_migen(dut, dut.signals, full, memories=mems,
                             axi_log={f"axi{i}": dut.axis[i] for i in axi_log})
        assert backend == "verilator", backend
        axi = {}
        for i in range(n):
            axi[f"axi{i}"] = dict(size=model_size, latency=lat[i], r_stall=(stl[i]*100 + 8)//16,
                                  reorder=reorder_ids, max_len=16, seed=i + 1, log=int(i in axi_log),
                                  init=payloads[i])
        return self.simulator().run(full, images={"sp": sp} if sp else None, axi=axi)

# Script Helpers -----------------------------------------------------------------------------------

def submit(s, cmd, fields=None, prefix="cmd_"):
    """Engine command: set the fields (and `fields` not in cmd to 0), then the valid/ready
    handshake."""
    for name in fields or ():
        if name not in cmd:
            s.set(prefix + name, 0)
    for name, value in cmd.items():
        s.set(prefix + name, value)
    s.handshake(prefix + "valid", prefix + "ready", tag="cmd")


def csr_write(s, name, value):
    """A CSRStorage write as the bank does it: storage, with re for one cycle."""
    s.set(name, value)
    s.set(name + "_re", 1)
    s.step()
    s.set(name + "_re", 0)


def vec_upload(s, words):
    """Vector-unit program upload through its CSRs."""
    s.set("vu_prog_adr", 0)
    s.set("vu_prog_adr_re", 1)
    s.step()
    s.set("vu_prog_adr_re", 0)
    for wd in words:
        s.set("vu_prog", wd)
        s.set("vu_prog_re", 1)
        s.step()
    s.set("vu_prog_re", 0)
    s.step()


def vec_submit(s, args):
    """Vector-unit program start through its CSRs."""
    for i, a in enumerate(args):
        s.set(f"vu_arg{i}", a & 0xFFFFFFFF)
    s.set("vu_pc", 0)
    s.set("vu_submit", 1)
    s.set("vu_submit_re", 1)
    s.step()
    s.set("vu_submit_re", 0)
    s.step()


def seq_start(s, base, count):
    """Sequencer list start (base, count, then start with re), as the tests drove it."""
    s.set("seq_base", base)
    s.set("seq_count", count)
    s.step()
    s.set("seq_start", 1)
    s.set("seq_start_re", 1)
    s.step()
    s.set("seq_start_re", 0)
    s.step()
