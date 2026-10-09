#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Compiled (Verilator) simulation of Migen/LiteX modules, with a Migen fallback.

A Simulator converts a module to Verilog (Migen's converter, the one the SoC build uses) and
builds it with Verilator into build/verilator/<top>-<hash>/: the hash covers the Verilog, the
generated bindings, every C++ source and the flags, so an unchanged DUT is never rebuilt, and
a file lock makes concurrent builds (pytest-xdist, run_guarded -j) wait for one another.

The C++ side lives in test/common/rtl/: sim.h (signals and memories by name, cycle cap),
axi_slave.h (AXI4 slave memory: latency, random stalls on every channel, bursts, IDs,
same-ID serialization), sp_client.h (resident scratchpad client port), script.h and
script_main.cpp (the default driver: runs a Script). A testbench with its own main()
includes the same headers (`driver=`; rtl_client.py does).

    sim = Simulator(dut, ios={"done": dut.done, ...}, axi={"axi0": axi}, memories={"sp": mem})
    s   = Script()
    s.set("cmd_kind", 0); s.handshake("cmd_valid", "cmd_ready")
    s.wait({"done": 1}, cap=50000, tag="op"); s.read("sp", 0x4000, 256, "res")
    r = sim.run(s, images={"sp": sp_bytes}, axi={"axi0": dict(init=payload, latency=8)})
    r.reads["res"], r.waits["op"], r.peeks, r.axi_log

The same Script runs on the Migen model of the DUT with run_migen() - the slow cross-check
of the emitted Verilog against Migen's own semantics.

Data goes through files (images in, memory reads out); results come back as "@" lines.
"""

import os
import re
import fcntl
import shutil
import hashlib
import tempfile
import functools
import subprocess
from pathlib import Path

from migen import *
from migen.fhdl import verilog
from migen.sim import passive

from common import ROOT, RTL

# Constants ----------------------------------------------------------------------------------------

CACHE   = Path(os.environ.get("TRIXIUM_VERILATOR_CACHE", ROOT / "build" / "verilator"))
HEADERS = ("sim.h", "axi_slave.h", "sp_client.h", "script.h")
FLAGS   = ["-Wno-fatal", "-O3", "--x-assign", "fast", "--x-initial", "fast"]

# Helpers ------------------------------------------------------------------------------------------

def verilator_path():
    """The verilator binary, or skip the calling test when there is none."""
    path = shutil.which("verilator")
    if not path:
        import pytest
        pytest.skip("Verilator required")
    return path


def endpoint_ios(prefix, ep):
    """{prefix_field: Signal} for a stream.Endpoint (valid/ready/first/last, payload, params)."""
    ios = {f"{prefix}_valid": ep.valid, f"{prefix}_ready": ep.ready}
    for name in ("first", "last"):
        if hasattr(ep, name):
            ios[f"{prefix}_{name}"] = getattr(ep, name)
    for name, *_ in list(ep.description.payload_layout) + list(ep.description.param_layout):
        ios[f"{prefix}_{name}"] = getattr(ep, name)
    return ios


def axi_ios(prefix, axi):
    """{prefix_<ch>_<field>: Signal} for an AXIInterface (the names axi_slave.h looks up)."""
    ios = {}
    for ch in ("aw", "w", "b", "ar", "r"):
        ios.update(endpoint_ios(f"{prefix}_{ch}", getattr(axi, ch)))
    return ios


def _hex(value):
    return f"{value & ((1 << 1024) - 1):x}"

# Script -------------------------------------------------------------------------------------------

class Script:
    """A testbench as a list of operations, run by either backend (see test/common/rtl/script.h)."""
    def __init__(self):
        self.ops = []

    def set(self, name, value):
        self.ops.append(("set", name, int(value)))

    def sets(self, prefix="", **fields):
        for name, value in fields.items():
            self.set(prefix + name, value)

    def step(self, n=1):
        if n:
            self.ops.append(("step", int(n)))

    def reset(self, n=2):
        self.ops.append(("reset", int(n)))

    def wait(self, cond, cap=100000, tag="wait", fatal=True):
        """Cycle until every cond holds: {name: value} (==) or {"!name": value} (!=)."""
        if isinstance(cond, str):
            cond = {cond: 1}
        conds = [(n[1:], "!=", v) if n.startswith("!") else (n, "==", v) for n, v in cond.items()]
        self.ops.append(("wait", int(cap), bool(fatal), tag, conds))

    def handshake(self, valid, ready, cap=100000, tag="handshake"):
        self.ops.append(("handshake", valid, ready, int(cap), tag))

    def peek(self, name, tag=None):
        self.ops.append(("peek", name, tag or name))

    def count(self, name, tag):
        self.ops.append(("count", name, tag))

    def count_end(self, tag):
        self.ops.append(("count_end", tag))

    def read(self, mem, addr, length, tag):
        self.ops.append(("read", mem, int(addr), int(length), tag))

    def dump(self, *names):
        self.ops.append(("dump",) + tuple(names))

    def text(self):
        lines = []
        for op in self.ops:
            kind = op[0]
            if kind == "set":
                lines.append(f"set {op[1]} {_hex(op[2])}")
            elif kind == "wait":
                _, cap, fatal, tag, conds = op
                cs = " ".join(f"{n}{o}{_hex(v)}" for n, o, v in conds)
                lines.append(f"wait {cap} {int(fatal)} {tag} {cs}")
            else:
                lines.append(" ".join(str(x) for x in op))
        return "\n".join(lines) + "\n"

# Result -------------------------------------------------------------------------------------------

class Result:
    """What a run returns. peeks: {tag: value} (a list when a tag repeats); waits: {tag: (cycles,
    met)}; counts / handshakes: {tag: cycles}; reads: {tag: bytes}; axi_log: [(prefix, "ar"|"aw",
    cycle, addr, len, id)]; axi_stats: {prefix: (ar, r, aw, w)}; cycles: total."""
    def __init__(self):
        self.peeks      = {}
        self.waits      = {}
        self.counts     = {}
        self.handshakes = {}
        self.reads      = {}
        self.axi_log    = []
        self.axi_stats  = {}
        self.cycles     = 0
        self.stdout     = ""
        self._multi     = set()

    def _add(self, d, tag, value):
        key = (id(d), tag)
        if tag not in d:
            d[tag] = value
        elif key in self._multi:
            d[tag].append(value)
        else:
            d[tag] = [d[tag], value]
            self._multi.add(key)

    def peek(self, tag, signed_width=None):
        v = self.peeks[tag]
        if signed_width and v >> (signed_width - 1) & 1:
            v -= 1 << signed_width
        return v

    def axi_addresses(self, prefix, kind):
        return [a for p, k, _, a, _, _ in self.axi_log if p == prefix and k == kind]

# Simulator ----------------------------------------------------------------------------------------

class Simulator:
    """A Migen/LiteX module compiled with Verilator (cached; see the module doc).

    dut      : the module, or with `key` a factory returning dict(dut=, ios=, axi=, memories=).
    ios      : {name: Signal} - the ports the testbench drives/reads, by these names.
    axi      : {prefix: AXIInterface} - ports for AxiSlave models (axi_ios()).
    memories : {name: Memory} - DUT memories exposed to images/reads (made public).
    driver   : C++ main (source text); default script_main.cpp.
    key      : a string naming the DUT's construction. With it, a build whose key, gateware /
               model / Migen / LiteX sources and C++ sources are unchanged is reused without
               constructing or converting the DUT at all (seconds per configuration).
    """
    def __init__(self, dut, ios=None, axi=None, memories=None, name="dut", driver=None, flags=(),
                 key=None):
        self.top    = name
        self.driver = driver
        self.flags  = list(flags)
        self.binary = None
        if key is not None:
            h = hashlib.sha256()
            for part in [key, name, self._driver_source(), " ".join(FLAGS + self.flags),
                         _verilator_version(), _source_fingerprint()] + [(RTL / f).read_text() for f in HEADERS]:
                h.update(part.encode())
                h.update(b"\0")
            index = CACHE / "keys" / f"{name}-{h.hexdigest()[:24]}"
            try:
                self.dir    = CACHE / index.read_text().strip()
                self.binary = self.dir / "obj" / f"V{self.top}"
                if (self.dir / "ok").exists() and self.binary.exists():
                    return
            except OSError:
                pass
            built = dut()
            dut, ios, axi, memories = (built["dut"], built.get("ios"), built.get("axi"),
                                       built.get("memories"))
        # ios: {name: Signal}, or signals named by the converter (a harness using d.<name>).
        self.ios = dict(ios or {}) if isinstance(ios, dict) or ios is None else {None: list(ios)}
        for prefix, a in (axi or {}).items():
            self.ios.update(axi_ios(prefix, a))
        self.memories = dict(memories or {})
        self._build(dut)
        if key is not None:
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(self.dir.name + "\n")

    def _driver_source(self):
        return self.driver if self.driver is not None else (RTL / "script_main.cpp").read_text()

    # Build ----------------------------------------------------------------------------------------

    def _build(self, dut):
        unnamed = self.ios.pop(None, [])
        conv  = verilog.convert(dut, ios=set(self.ios.values()) | set(unnamed), name=self.top)
        ns    = conv.ns
        vsrc  = conv.main_source
        self.ios.update({ns.get_name(s): s for s in unnamed})
        names = {n: ns.get_name(s) for n, s in self.ios.items()}
        mems  = {}
        for n, m in self.memories.items():
            vname = ns.get_name(m)
            if not re.search(rf"reg \[{m.width - 1}:0\] {re.escape(vname)}\[0:{m.depth - 1}\];", vsrc):
                raise ValueError(f"memory {n}: '{vname}' not found in the Verilog")
            assert m.width % 8 == 0, f"memory {n}: width {m.width} is not whole bytes"
            mems[n] = (vname, m.width, m.depth)

        header = self._header(names, {n: len(s) for n, s in self.ios.items()}, mems)
        vlt    = "`verilator_config\n" + "".join(
            f'public_flat_rw -module "{self.top}" -var "{v}"\n' for v, _, _ in mems.values())
        driver = self._driver_source()
        h = hashlib.sha256()
        for part in [vsrc, header, vlt, driver, " ".join(FLAGS + self.flags), _verilator_version()] + \
                    [(RTL / f).read_text() for f in HEADERS] + \
                    [f"{k}\n{v}" for k, v in sorted(conv.data_files.items())]:
            h.update(part.encode())
            h.update(b"\0")
        self.dir = CACHE / f"{self.top}-{h.hexdigest()[:16]}"
        self.binary = self.dir / "obj" / f"V{self.top}"
        stamp = self.dir / "ok"
        CACHE.mkdir(parents=True, exist_ok=True)
        with open(CACHE / f"{self.dir.name}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if stamp.exists() and self.binary.exists():
                return
            verilator = verilator_path()
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir.mkdir(parents=True)
            (self.dir / f"{self.top}.v").write_text(vsrc)
            for fname, content in conv.data_files.items():
                (self.dir / fname).write_text(content)
            (self.dir / "sim_top.h").write_text(header)
            (self.dir / "public.vlt").write_text(vlt)
            (self.dir / "main.cpp").write_text(driver)
            cmd = [verilator, "--cc", "--exe", "--build", "-j", str(min(8, os.cpu_count() or 4))] + \
                  FLAGS + self.flags + \
                  ["--Mdir", "obj", "--top-module", self.top,
                   "-CFLAGS", f"-I{RTL.resolve()} -I{self.dir.resolve()}",
                   "public.vlt", f"{self.top}.v", "main.cpp"]
            built = subprocess.run(cmd, cwd=self.dir, capture_output=True, text=True, timeout=3600)
            (self.dir / "build.log").write_text(built.stdout + built.stderr)
            assert built.returncode == 0, built.stdout[-4000:] + built.stderr[-4000:]
            stamp.write_text("ok\n")

    def _header(self, names, widths, mems):
        top = f"V{self.top}"
        lines = [
            "// Generated by test/common/verilator.py.",
            "#pragma once",
            f'#include "{top}.h"',
            f'#include "{top}___024root.h"',
            '#include "sim.h"',
            f"#define SIM_TOP {top}",
            f"static void sim_bind(Sim &sim, {top} &d) {{",
            "    sim.clock = [&d](int v) { d.sys_clk = v; };",
            "    sim.eval  = [&d]() { d.eval(); };",
        ]
        for n, v in sorted(names.items()):
            lines.append(f'    sim.add("{n}", {widths[n]}, &d.{v});')
        for n, (v, w, depth) in sorted(mems.items()):
            lines.append(f'    sim.add_memory(sim_dut_memory("{n}", &d.rootp->{self.top}__DOT__{v}, {w}, {depth}));')
        lines.append("}")
        return "\n".join(lines) + "\n"

    # Run ------------------------------------------------------------------------------------------

    def run(self, script=None, images=None, axi=None, spclient=None, cycles=50_000_000, args=None,
            timeout=3600):
        """Run a Script (default driver) or the custom driver with `args` (paths are str()).

        images  : {memory: bytes} loaded before the script (DUT memories start zeroed).
        axi     : {prefix: dict(init=bytes, size=, latency=, r_stall=, ar_stall=, aw_stall=,
                  w_stall=, b_stall=, reorder=, max_len=, seed=, log=, write_order=)}.
        spclient: (name, size, grant_pct) - a SpClient on the sp_* ports.
        """
        with tempfile.TemporaryDirectory(prefix=f"{self.top}-") as tmp:
            tmp = Path(tmp)
            if self.driver is not None:
                cmd = [str(self.binary)] + [str(a) for a in (args or [])]
            else:
                cfg = [f"cycles {cycles}"]
                for prefix, opts in (axi or {}).items():
                    opts = dict(opts)
                    init = opts.pop("init", None)
                    cfg.append(f"axi {prefix} " + " ".join(f"{k}={int(v) if isinstance(v, bool) else v}"
                                                          for k, v in opts.items()))
                    if init:
                        images = dict(images or {}, **{prefix: init})
                if spclient:
                    cfg.append("spclient {} {} {}".format(*spclient))
                for i, (mem, data) in enumerate((images or {}).items()):
                    path = tmp / f"image{i}.bin"
                    path.write_bytes(bytes(data))
                    cfg.append(f"image {mem} {path}")
                (tmp / "config.txt").write_text("\n".join(cfg) + "\n")
                (tmp / "script.txt").write_text(script.text())
                cmd = [str(self.binary), str(tmp / "config.txt"), str(tmp / "script.txt"), str(tmp)]
            run = subprocess.run(cmd, cwd=self.dir, capture_output=True, text=True, timeout=timeout)
            if run.returncode != 0:
                raise AssertionError(f"{self.top}: " + run.stdout[-2000:] + run.stderr[-4000:])
            return _parse(run.stdout, tmp)


@functools.lru_cache(maxsize=None)
def _source_fingerprint():
    """Names, sizes and mtimes of every Python source a DUT construction can depend on."""
    import migen
    import litex
    roots = [ROOT / "gateware", ROOT / "model", Path(__file__).parent,
             Path(migen.__file__).parent, Path(litex.__file__).parent]
    h = hashlib.sha256()
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            st = path.stat()
            h.update(f"{path}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def _verilator_version():
    try:
        return subprocess.run([shutil.which("verilator") or "verilator", "--version"],
                              capture_output=True, text=True).stdout
    except OSError:
        return ""


def _parse(stdout, tmp):
    r = Result()
    r.stdout = stdout
    for line in stdout.splitlines():
        if not line.startswith("@"):
            continue
        f = line.split()
        kind = f[0][1:]
        if kind == "peek":
            r._add(r.peeks, f[1], int(f[2], 16))
        elif kind == "wait":
            r._add(r.waits, f[1], (int(f[2]), f[3] == "1"))
        elif kind == "count":
            r._add(r.counts, f[1], int(f[2]))
        elif kind == "handshake":
            r._add(r.handshakes, f[1], int(f[2]))
        elif kind == "read":
            r._add(r.reads, f[1], (tmp / f"{f[1]}.bin").read_bytes())
        elif kind == "axi":
            r.axi_log.append((f[1], f[2], int(f[3]), int(f[4], 16), int(f[5]), int(f[6])))
        elif kind == "axistat":
            r.axi_stats[f[1]] = tuple(int(x) for x in f[2:])
        elif kind == "cycles":
            r.cycles = int(f[2])
    return r

# Migen Backend ------------------------------------------------------------------------------------

def run_migen(dut, signals, script, memories=None, axi_log=None, vcd_name=None):
    """Run a Script on the Migen model of the DUT (same results as Simulator.run).

    signals : {name: Signal} (the same names as the Verilator ios).
    memories: {name: Memory} for reads (the AXI models' memories by their prefix, too).
    axi_log : {prefix: AXIInterface} whose AR/AW handshakes are logged.
    """
    r = Result()
    memories = memories or {}
    state = dict(dirty=False, cycle=0, done=False)
    counters = {}
    dumps = []

    def sig(name):
        try:
            return signals[name]
        except KeyError:
            raise KeyError(f"no signal '{name}' in the Migen signal map") from None

    def mask(name, v):
        return v & ((1 << len(sig(name))) - 1)

    def tick():
        for tag, (name, n) in counters.items():
            if (yield sig(name)):
                counters[tag] = (name, n + 1)
        yield
        state["cycle"] += 1
        state["dirty"] = False

    def flush():
        # Writes are only visible after the next edge in Migen's simulator: one cycle makes
        # them visible to the reads that follow, as a settle does in the compiled backend.
        if state["dirty"]:
            yield from tick()

    def fail(msg):
        values = []
        for name in dumps:
            values.append(f"  {name} = 0x{mask(name, (yield sig(name))):x}")
        raise AssertionError(f"{msg} (cycle {state['cycle']})\n" + "\n".join(values))

    def gen():
        for op in script.ops:
            kind = op[0]
            if kind == "set":
                yield sig(op[1]).eq(op[2])
                state["dirty"] = True
            elif kind == "step":
                for _ in range(op[1]):
                    yield from tick()
            elif kind == "reset":
                for _ in range(op[1]):
                    yield from tick()
            elif kind == "wait":
                _, cap, fatal, tag, conds = op
                yield from flush()
                n, ok = 0, False
                while True:
                    ok = True
                    for name, o, v in conds:
                        cur = mask(name, (yield sig(name)))
                        ok = ok and ((cur == mask(name, v)) == (o == "=="))
                    if ok or n >= cap:
                        break
                    yield from tick()
                    n += 1
                if not ok and fatal:
                    yield from fail(f"{tag}: timeout after {n} cycles")
                r._add(r.waits, tag, (n, ok))
            elif kind == "handshake":
                _, valid, ready, cap, tag = op
                yield sig(valid).eq(1)
                yield from tick()
                n = 0
                while not (yield sig(ready)):
                    yield from tick()
                    n += 1
                    if n >= cap:
                        yield from fail(f"{tag}: {valid} never accepted after {n} cycles")
                yield sig(valid).eq(0)
                state["dirty"] = True
                r._add(r.handshakes, tag, n)
            elif kind == "peek":
                yield from flush()
                r._add(r.peeks, op[2], mask(op[1], (yield sig(op[1]))))
            elif kind == "count":
                counters[op[2]] = (op[1], 0)
            elif kind == "count_end":
                r._add(r.counts, op[1], counters.pop(op[1])[1])
            elif kind == "read":
                _, mem, addr, length, tag = op
                m  = memories[mem]
                wb = m.width//8
                data = bytearray()
                for w in range(addr//wb, (addr + length + wb - 1)//wb):
                    data += int((yield m[w])).to_bytes(wb, "little")
                off = addr % wb
                r._add(r.reads, tag, bytes(data[off:off + length]))
            elif kind == "dump":
                dumps.extend(op[1:])
            else:
                raise ValueError(f"unknown script operation {kind}")
        r.cycles = state["cycle"]
        state["done"] = True

    @passive
    def probe(prefix, axi):
        cycle = 0
        while True:
            for kind, ch in (("ar", axi.ar), ("aw", axi.aw)):
                if (yield ch.valid) and (yield ch.ready):
                    r.axi_log.append((prefix, kind, cycle, (yield ch.addr), (yield ch.len),
                                      (yield ch.id) if hasattr(ch, "id") else 0))
            cycle += 1
            yield

    gens = [gen()] + [probe(p, a) for p, a in (axi_log or {}).items()]
    run_simulation(dut, gens, vcd_name=vcd_name)
    return r
