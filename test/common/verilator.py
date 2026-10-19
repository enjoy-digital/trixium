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

Migen testbench generators run unchanged on the compiled DUT through Compiled (the model
as a shared library, rtl/cosim.cpp, driven through ctypes with run_simulation's semantics):

    run(dut, [gen()], backend)                    # "verilator" (default) or "migen"
    run(dut, gen(), channels=[ChannelModel(axi, size=..., init=..., latency=...)])

Script data goes through files (images in, memory reads out); results come back as "@"
lines.
"""

import os
import re
import random
import fcntl
import ctypes
import shutil
import hashlib
import tempfile
import functools
import subprocess
from pathlib import Path

from migen import *
from migen.fhdl import verilog
from migen.fhdl.structure import _Assign, _Slice
from migen.fhdl.specials import _MemoryLocation
from migen.sim import passive

from common import ROOT, RTL

# Constants ----------------------------------------------------------------------------------------

CACHE   = Path(os.environ.get("TRIXIUM_VERILATOR_CACHE", ROOT / "build" / "verilator"))
HEADERS = ("sim.h", "axi_slave.h", "sp_client.h", "script.h")
FLAGS   = ["-Wno-fatal", "-O3", "--x-assign", "fast", "--x-initial", "fast"]

# Helpers ------------------------------------------------------------------------------------------

def tracing():
    """Waveforms on request only (pytest --waves, or TRIXIUM_TRACE=1): a tracing build is a
    different (slower) model, cached separately."""
    return os.environ.get("TRIXIUM_TRACE") == "1"


_trace_runs = {}

def trace_path(name, ext):
    """build/traces/<current test id>/<name>-<n>.<ext>: one file per run of the test."""
    test = os.environ.get("PYTEST_CURRENT_TEST", "no-test").rsplit(" ", 1)[0]
    d = ROOT / "build" / "traces" / re.sub(r"[^A-Za-z0-9_.-]+", "_", test)
    d.mkdir(parents=True, exist_ok=True)
    n = _trace_runs[d] = _trace_runs.get(d, -1) + 1
    return d / f"{name}-{n}.{ext}"


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

    def random(self, name, pct, seed=1):
        """Drive `name` 1 on `pct` percent of the cycles from now on (pct < 0: stop)."""
        self.ops.append(("random", name, int(pct), int(seed)))

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
    driver   : C++ main (source text); default script_main.cpp. With `key`, also a function of
               the factory's dict returning it (a driver generated from the DUT's shape): the key
               then names the driver as well (see file_key()).
    key      : a string naming the DUT's construction. With it, a build whose key, gateware /
               model / Migen / LiteX sources and C++ sources are unchanged is reused without
               constructing or converting the DUT at all (seconds per configuration).
    """
    def __init__(self, dut, ios=None, axi=None, memories=None, name="dut", driver=None, flags=(),
                 key=None):
        self.top    = name
        self.driver = driver
        self.flags  = list(flags) + (["--trace-fst"] if tracing() else [])
        self.binary = None
        if key is not None:
            h = hashlib.sha256()
            driver_text = "" if callable(driver) else self._driver_source()
            for part in [key, name, driver_text, " ".join(FLAGS + self.flags),
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
            if callable(driver):
                self.driver = driver(built)
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

        self.inputs = set(re.findall(r"^\s*input\s+(?:wire\s+)?(?:signed\s+)?(?:\[[^\]]*\]\s*)?(\w+)", vsrc, re.M))
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
            # Only the binary is used: drop the precompiled headers and objects (~100 MB a build).
            for f in (self.dir / "obj").iterdir():
                if f.suffix in (".gch", ".o", ".a", ".d"):
                    f.unlink()
            stamp.write_text("ok\n")

    def _header(self, names, widths, mems):
        top = f"V{self.top}"
        lines = [
            "// Generated by test/common/verilator.py.",
            "#pragma once",
            f'#include "{top}.h"',
            f'#include "{top}___024root.h"',
            '#include "sim.h"',
            "#if VM_TRACE",
            '#include "verilated_fst_c.h"',
            "#endif",
            f"#define SIM_TOP {top}",
            f"static void sim_bind(Sim &sim, {top} &d) {{",
            "    sim.clock = [&d](int v) { d.sys_clk = v; };",
            "    sim.eval  = [&d]() { d.eval(); };",
            "#if VM_TRACE",
            "    // Tracing builds only (TRIXIUM_TRACE=1 / pytest --waves): dump to $TRIXIUM_TRACE_FILE.",
            '    if (const char *path = std::getenv("TRIXIUM_TRACE_FILE")) {',
            "        d.contextp()->traceEverOn(true);",
            "        auto *t = new VerilatedFstC;",
            "        d.trace(t, 99);",
            "        t->open(path);",
            "        sim.dump = [t](uint64_t time) { t->dump(time); };",
            "        sim_on_exit = [t]() { t->close(); };",
            "    }",
            "#endif",
        ]
        for n, v in sorted(names.items()):
            flag = ", true" if v in self.inputs else ""
            lines.append(f'    sim.add("{n}", {widths[n]}, &d.{v}{flag});')
        for n, (v, w, depth) in sorted(mems.items()):
            lines.append(f'    sim.add_memory(sim_dut_memory("{n}", &d.rootp->{self.top}__DOT__{v}, {w}, {depth}));')
        lines.append("}")
        return "\n".join(lines) + "\n"

    # Run ------------------------------------------------------------------------------------------

    def run(self, script=None, images=None, axi=None, cycles=50_000_000, args=None,
            timeout=3600):
        """Run a Script (default driver) or the custom driver with `args` (paths are str()).

        images  : {memory: bytes} loaded before the script (DUT memories start zeroed).
        axi     : {prefix: dict(init=bytes, size=, latency=, r_stall=, ar_stall=, aw_stall=,
                  w_stall=, b_stall=, reorder=, max_len=, seed=, log=, write_order=)}.
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
                for i, (mem, data) in enumerate((images or {}).items()):
                    path = tmp / f"image{i}.bin"
                    path.write_bytes(bytes(data))
                    cfg.append(f"image {mem} {path}")
                (tmp / "config.txt").write_text("\n".join(cfg) + "\n")
                (tmp / "script.txt").write_text(script.text())
                cmd = [str(self.binary), str(tmp / "config.txt"), str(tmp / "script.txt"), str(tmp)]
            env = dict(os.environ)
            env.pop("TRIXIUM_TRACE_FILE", None)
            if tracing():
                env["TRIXIUM_TRACE_FILE"] = str(trace_path(self.top, "fst"))
            run = subprocess.run(cmd, cwd=self.dir, env=env, capture_output=True, text=True,
                                 timeout=timeout)
            if run.returncode != 0:
                raise AssertionError(f"{self.top}: " + run.stdout[-2000:] + run.stderr[-4000:])
            return _parse(run.stdout, tmp)


def file_key(path, *parts):
    """A Simulator key naming a construction made in the file `path` (a test file: its text,
    e.g. a driver template, is hashed in) with arguments `parts`."""
    text = Path(path).read_bytes()
    return f"{Path(path).name}:{hashlib.sha256(text).hexdigest()[:16]}:{parts!r}"


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

def run_migen(dut, signals, script, memories=None, axi_log=None, vcd_name=None, clocks=None):
    """Run a Script on the Migen model of the DUT (same results as Simulator.run).

    signals : {name: Signal} (the same names as the Verilator ios).
    memories: {name: Memory} for reads (the AXI models' memories by their prefix, too).
    axi_log : {prefix: AXIInterface} whose AR/AW handshakes are logged.
    clocks  : run_simulation's {domain: period} (default: sys at 10); the script's cycles are sys's.
    """
    r = Result()
    memories = memories or {}
    state = dict(dirty=False, cycle=0, done=False)
    counters = {}
    randoms  = {}
    dumps    = []

    def sig(name):
        try:
            return signals[name]
        except KeyError:
            raise KeyError(f"no signal '{name}' in the Migen signal map") from None

    def mask(name, v):
        return v & ((1 << len(sig(name))) - 1)

    def tick():
        for name, (pct, rng) in randoms.items():
            yield sig(name).eq(int(rng.random()*100 < pct))
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
            elif kind == "random":
                _, name, pct, seed = op
                if pct < 0:
                    randoms.pop(name, None)
                else:
                    randoms[name] = (pct, random.Random(seed))
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
    if vcd_name is None and tracing():
        vcd_name = str(trace_path("migen", "vcd"))
    run_simulation(dut, gens, vcd_name=vcd_name, **({} if clocks is None else dict(clocks=clocks)))
    return r

# Compiled Co-Simulation ---------------------------------------------------------------------------

LIB_FLAGS = ["-CFLAGS", "-fPIC", "-LDFLAGS", "-shared"]

_libraries = {}

def _library(path):
    """The co-simulation library at `path`, loaded once per process (ctypes)."""
    path = str(path)
    if path not in _libraries:
        lib = ctypes.CDLL(path)
        P, I, U, ULL, B = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32),
                           ctypes.c_ulonglong, ctypes.c_char_p)
        for fn, res, args in [
                ("cosim_new",           P,    [B]),
                ("cosim_free",          None, [P]),
                ("cosim_error",         B,    [P]),
                ("cosim_signal",        I,    [P, B]),
                ("cosim_width",         I,    [P, I]),
                ("cosim_is_input",      I,    [P, I]),
                ("cosim_get",           I,    [P, I, U]),
                ("cosim_set",           None, [P, I, U]),
                ("cosim_memory",        I,    [P, B]),
                ("cosim_memory_size",   ULL,  [P, I]),
                ("cosim_memory_access", I,    [P, I, ULL, B, ULL, I]),
                ("cosim_tick",          I,    [P, ULL]),
                ("cosim_cycle",         ULL,  [P])]:
            f = getattr(lib, fn)
            f.restype, f.argtypes = res, args
        _libraries[path] = lib
    return _libraries[path]


class Compiled:
    """A DUT compiled as a co-simulation library that runs Migen testbench generators unchanged.

    run() keeps run_simulation's contract on the Verilated model: every generator runs once
    per sys clock; `(yield sig)` / `(yield mem[i])` read the settled state before the edge;
    `yield sig.eq(v)` / `yield mem[i].eq(v)` take effect after it (Migen commits them with
    the edge); "passive" generators do not keep the simulation alive.

    dut      : the module, constructed by the test (its Signals/Memories are the handles).
    signals  : {name: Signal} the generators read or drive: ports of the compiled DUT (the
               undriven ones are inputs, at their Migen reset value until driven).
    memories : {name: Memory} the generators read or write (made public).
    axi      : {prefix: AXIInterface} answered by C++ AXI slaves (axi_slave.h).
    key      : the DUT's construction (see Simulator): a hit skips the Verilog conversion.
    """
    def __init__(self, dut, signals, memories=None, axi=None, name="cosim", key=None):
        self.signals  = dict(signals)
        self.memories = dict(memories or {})
        self.axi      = dict(axi or {})
        build = dict(dut=dut, ios=self.signals, axi=self.axi, memories=self.memories)
        if key is not None:
            key = f"{key}|{sorted(self.signals)}|{sorted(self.memories)}|{sorted(self.axi)}"
            self.sim = Simulator(lambda: build, name=name, driver=_COSIM, flags=LIB_FLAGS, key=key)
        else:
            self.sim = Simulator(dut, ios=self.signals, axi=self.axi, memories=self.memories,
                                 name=name, driver=_COSIM, flags=LIB_FLAGS)
        self.handles = dict(self.signals)
        for prefix, a in self.axi.items():
            self.handles.update(axi_ios(prefix, a))

    def run(self, generators, slaves=None, aliases=None, cycles=None):
        """slaves: {prefix: AxiConfig options (size, latency, r_stall, ...; init: bytes)} for the
        AXI ports; aliases: {Signal or Memory: name} for testbench objects that are C++ models
        here (an HBMChannelModel's memory -> "axi0", its freeze -> "axi0_freeze"). Returns the
        cycle count."""
        if not isinstance(generators, (list, tuple)):
            generators = [generators]
        lib = _library(self.sim.binary)
        cfg, images = [f"cycles {cycles}"] if cycles else [], {}
        for prefix, opts in (slaves or {}).items():
            opts = dict(opts)
            if opts.get("init"):
                images[prefix] = bytes(opts["init"])
            opts.pop("init", None)
            cfg.append(f"axi {prefix} " + " ".join(f"{k}={int(v) if isinstance(v, bool) else v}"
                                                  for k, v in opts.items()))
        # Memory .init files are read at construction, relative to the build directory; a
        # tracing build dumps to $TRIXIUM_TRACE_FILE (read at construction too).
        cwd, trace = os.getcwd(), os.environ.pop("TRIXIUM_TRACE_FILE", None)
        if tracing():
            os.environ["TRIXIUM_TRACE_FILE"] = str(trace_path(self.sim.top, "fst"))
        os.chdir(self.sim.dir)
        try:
            h = lib.cosim_new("\n".join(cfg).encode())
        finally:
            os.chdir(cwd)
            os.environ.pop("TRIXIUM_TRACE_FILE", None)
            if trace is not None:
                os.environ["TRIXIUM_TRACE_FILE"] = trace
        try:
            run = _CosimRun(lib, h, self, aliases or {})
            run.check(-1 if lib.cosim_error(h) else 0)
            for prefix, data in images.items():
                handle = lib.cosim_memory(h, prefix.encode())
                run.check(lib.cosim_memory_access(h, handle, 0, data, len(data), 1))
            return run.run(generators)
        finally:
            lib.cosim_free(h)


_COSIM = (RTL / "cosim.cpp").read_text()


def module_handles(module, depth=3):
    """({name: Signal}, {name: Memory}) of a module tree, for Compiled: the public Signal,
    Record (stream endpoint) and Memory attributes, of submodules down to `depth` levels."""
    signals, memories, seen = {}, {}, set()

    def visit(obj, name, level):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, Signal):
            signals[name] = obj
        elif isinstance(obj, Memory):
            if obj.width % 8 == 0:
                memories[name] = obj
        elif isinstance(obj, Record):
            for attr, value in vars(obj).items():
                if not attr.startswith("_"):
                    visit(value, f"{name}_{attr}", level)
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj[:64]):
                visit(value, f"{name}{i}", level)
        elif (isinstance(obj, Module) and level <= depth) or (level == 0 and hasattr(obj, "__dict__")):
            # A module tree; at the top, any object holding them (e.g. a LiteDRAM native port).
            for attr, value in vars(obj).items():
                if not attr.startswith("_"):
                    visit(value, f"{name}_{attr}" if name else attr, level + 1)

    visit(module, "", 0)
    return signals, memories


class ChannelModel:
    """An HBMChannelModel's stand-in for compiled runs (same arguments): the channel is a C++ AXI
    slave on the DUT's port, `mem` and `freeze` are the testbench's handles to it (run() maps
    them). A DUT builds HBMChannelModel(axi, ...) for Migen, ChannelModel(axi, ...) otherwise."""
    def __init__(self, axi, size=64*1024, init=None, latency=16, stall_rate=0, reorder_ids=0,
                 write_order="aw_first"):
        self.axi    = axi
        self.mem    = Memory(256, size//32)
        self.freeze = Signal()
        self.options = dict(size=size, latency=latency, r_stall=(stall_rate*100 + 8)//16,
                            reorder=reorder_ids, max_len=16, write_order=write_order)
        if init:
            self.options["init"] = b"".join(int(w).to_bytes(32, "little") for w in init)


def run(dut, generators, backend="verilator", signals=None, memories=None, depth=3, name="tb",
        key=None, clocks=None, axi=None, channels=(), **kwargs):
    """run_simulation(dut, generators) on either backend: "migen" (Migen's simulator) or
    "verilator" (Compiled). signals / memories: what the generators touch (default: every
    handle module_handles() finds); clocks: Migen's (the compiled DUT derives its domains from
    sys_clk); axi: {prefix: AXIInterface} for C++ slaves; channels: ChannelModels (each one
    "axi<i>"); kwargs go to Compiled.run (slaves, aliases, cycles)."""
    if backend == "migen":
        run_simulation(dut, generators, **({} if clocks is None else dict(clocks=clocks)),
                       vcd_name=str(trace_path("migen", "vcd")) if tracing() else None)
        return
    assert backend == "verilator", backend
    if signals is None or memories is None:
        found_signals, found_memories = module_handles(dut, depth)
        signals  = found_signals  if signals  is None else signals
        memories = found_memories if memories is None else memories
    axi = dict(axi or {})
    if channels:
        slaves  = kwargs.setdefault("slaves", {})
        aliases = kwargs.setdefault("aliases", {})
        for i, channel in enumerate(channels):
            axi[f"axi{i}"]    = channel.axi
            slaves[f"axi{i}"] = dict(channel.options, seed=i + 1)
            aliases[channel.mem]    = f"axi{i}"
            aliases[channel.freeze] = f"axi{i}_freeze"
    return Compiled(dut, signals, memories, axi=axi, name=name, key=key).run(generators, **kwargs)


class _CosimRun:
    """One run of a Compiled DUT: the handles, Migen's value semantics, the scheduler."""
    def __init__(self, lib, h, compiled, aliases):
        self.lib    = lib
        self.h      = h
        self.sigs   = {}  # id(Signal) -> (handle, width, signed, buffer)
        self.mems   = {}  # id(Memory) -> (handle, bytes per word)
        self.names  = {}  # id(object) -> name in the compiled model
        self.writes = []
        for objects in (compiled.handles, compiled.memories):
            for name, obj in objects.items():
                self.names[id(obj)] = name
        for obj, name in aliases.items():
            self.names[id(obj)] = name
        self.keep = list(aliases)  # The aliased objects outlive the run (ids stay unique).
        # Undriven ports start at their Migen reset value (the compiled model zeroes them).
        for sig in compiled.signals.values():
            if sig.reset.value and lib.cosim_is_input(h, self._signal(sig)[0]):
                self._set(sig, sig.reset.value)
        # Memories without init start zeroed, as in Migen.
        for mem in compiled.memories.values():
            if not mem.init:
                handle, _ = self._memory(mem)
                size = lib.cosim_memory_size(h, handle)
                self.check(lib.cosim_memory_access(h, handle, 0, bytes(size), size, 1))

    def check(self, rc):
        if rc:
            raise AssertionError("compiled simulation: " + self.lib.cosim_error(self.h).decode())

    def _signal(self, sig):
        entry = self.sigs.get(id(sig))
        if entry is None:
            name   = self.names.get(id(sig))
            handle = -1 if name is None else self.lib.cosim_signal(self.h, name.encode())
            if handle < 0:
                raise KeyError(f"{sig!r} is not a port of the compiled DUT: add it to the signals")
            width = self.lib.cosim_width(self.h, handle)
            entry = self.sigs[id(sig)] = (handle, width, sig.signed,
                                          (ctypes.c_uint32*((width + 31)//32))())
        return entry

    def _memory(self, mem):
        entry = self.mems.get(id(mem))
        if entry is None:
            name   = self.names.get(id(mem))
            handle = -1 if name is None else self.lib.cosim_memory(self.h, name.encode())
            if handle < 0:
                raise KeyError(f"memory {mem.name_override} is not exported by the compiled DUT: "
                               "add it to the memories")
            entry = self.mems[id(mem)] = (handle, (mem.width + 7)//8)
        return entry

    # Values ---------------------------------------------------------------------------------------

    def _get(self, sig):
        handle, width, signed, buf = self._signal(sig)
        self.check(self.lib.cosim_get(self.h, handle, buf))
        v = int.from_bytes(bytes(buf), "little") & ((1 << width) - 1)
        if signed and v >> (width - 1):
            v -= 1 << width
        return v

    def _set(self, sig, value):
        handle, width, _, buf = self._signal(sig)
        if not self.lib.cosim_is_input(self.h, handle):
            raise ValueError(f"{sig!r} is driven by the DUT: a testbench write would corrupt it")
        value &= (1 << width) - 1
        words = (ctypes.c_uint32*len(buf)).from_buffer_copy(value.to_bytes(4*len(buf), "little"))
        self.lib.cosim_set(self.h, handle, words)

    def _access(self, mem, index, value=None):
        handle, wb = self._memory(mem)
        if value is None:
            out = ctypes.create_string_buffer(wb)
            self.check(self.lib.cosim_memory_access(self.h, handle, index*wb, out, wb, 0))
            return int.from_bytes(out.raw, "little")
        data = (value & ((1 << (8*wb)) - 1)).to_bytes(wb, "little")
        self.check(self.lib.cosim_memory_access(self.h, handle, index*wb, data, wb, 1))

    def _eval(self, v):
        if isinstance(v, (int, bool)):
            return int(v)
        if isinstance(v, Constant):
            return v.value
        if isinstance(v, Signal):
            return self._get(v)
        if isinstance(v, _Slice):
            return (self._eval(v.value) >> v.start) & ((1 << (v.stop - v.start)) - 1)
        if isinstance(v, _MemoryLocation):
            return self._access(v.memory, self._eval(v.index))
        if isinstance(v, Cat):
            r, offset = 0, 0
            for part in v.l:
                r |= (self._eval(part) & ((1 << len(part)) - 1)) << offset
                offset += len(part)
            return r
        raise TypeError(f"compiled simulation: unsupported value {v!r}")

    def _request(self, request):
        if isinstance(request, (list, tuple)):
            return [self._request(r) for r in request]
        if isinstance(request, _Assign):
            self.writes.append((request.l, self._eval(request.r)))
            return None
        return self._eval(request)

    def _commit(self):
        for target, value in self.writes:
            if isinstance(target, _MemoryLocation):
                self._access(target.memory, self._eval(target.index), value)
            elif isinstance(target, _Slice):
                mask = ((1 << (target.stop - target.start)) - 1) << target.start
                self._set(target.value, (self._get(target.value) & ~mask) | ((value << target.start) & mask))
            else:
                self._set(target, value)
        self.writes = []

    # Scheduler ------------------------------------------------------------------------------------

    def run(self, generators):
        active  = list(generators)
        passive = set()
        while any(id(g) not in passive for g in active):
            for g in list(active):
                reply = None
                while True:
                    try:
                        request = g.send(reply)
                    except StopIteration:
                        active.remove(g)
                        break
                    if request is None:
                        break
                    if isinstance(request, str):
                        if request not in ("passive", "active"):
                            raise ValueError(f"unknown simulator command '{request}'")
                        (passive.add if request == "passive" else passive.discard)(id(g))
                        reply = None
                    else:
                        reply = self._request(request)
            self.check(self.lib.cosim_tick(self.h, 1))
            self._commit()
        return self.lib.cosim_cycle(self.h)
