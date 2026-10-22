#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""trixium smi over Etherbone: the card status of software/user/trixium_smi.c for a board without
PCIe (the Efinix Ti375 kit), through a litex_server and the build's csr.csv.

  python3 -m software.host.ebsmi --csr-csv build/ti375_v7/csr.csv [-w 1] [--json]

The same view as trixium_smi: the bitstream's identifier, CPU / engine state and sticky errors,
the engine and scratchpad-unit utilization and the per-channel read rate over a short interval,
then what the host tools recorded in /tmp/trixium (the weights the deploy uploaded, the chat
process). It only reads registers, plus perf_chan (the per-channel counter mux, which the
firmware only uses in its stall diagnostics), so it can run next to a chat.
"""

import os
import sys
import json
import time
import argparse

STATE_DIR  = "/tmp/trixium"
WIDTH      = 88
BEAT_BYTES = 32   # The engine's 256-bit channel ports.

# Host records (the same files as software/user/trixium_loader.c loader_state_write) -------------

def state_write(kind, devnum=0, **fields):
    """{"pid": <this process>, **fields} to STATE_DIR/trixium<devnum>.<kind>.json, atomically.
    Best effort: the monitor is optional."""
    try:
        if not os.path.isdir(STATE_DIR):
            os.mkdir(STATE_DIR)
            os.chmod(STATE_DIR, 0o777)
        path = os.path.join(STATE_DIR, f"trixium{devnum}.{kind}.json")
        tmp  = f"{path}.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"pid": os.getpid(), **fields}, f)
            f.write("\n")
        os.chmod(tmp, 0o666)
        os.rename(tmp, path)
    except OSError:
        pass

def state_read(kind, devnum=0):
    try:
        with open(os.path.join(STATE_DIR, f"trixium{devnum}.{kind}.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None

def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False

# Registers -----------------------------------------------------------------------------------------

class Card:
    def __init__(self, csr_csv, host="localhost", port=1234):
        from litex.tools.litex_client import RemoteClient
        self.bus = RemoteClient(host=host, port=port, csr_csv=csr_csv)
        self.bus.open()
        self.regs = self.bus.regs
        self.n_ch = int(self.bus.constants.trixium_n_channels)
        self.platform = str(getattr(self.bus.constants, "config_platform_name", ""))
        # The memory port's bytes per clock where the channels outrun it (the Ti375's shared
        # 512-bit port(s)); builds before the constant: one port.
        port = getattr(self.bus.constants, "trixium_port_bytes", None)
        if port is None and self.platform.startswith("efinix"):
            port = 64
        self.peak_bytes = min(self.n_ch*BEAT_BYTES, int(port)) if port else self.n_ch*BEAT_BYTES

    def close(self):
        self.bus.close()

    def has(self, name):
        return hasattr(self.regs, name)

    def read(self, name):
        return getattr(self.regs, name).read()

    def identifier(self):
        base  = self.bus.bases.identifier_mem
        chars = self.bus.read(base, 256)
        return bytes(c & 0xff for c in chars).split(b"\0")[0].decode(errors="replace")

    def sample(self):
        r = self.read
        s = dict(time=time.monotonic(),
                 cyc_total=r("trixium_engine_csr_cyc_total"), cyc_eng=r("trixium_engine_csr_cyc_eng"),
                 cyc_vec=r("trixium_engine_csr_cyc_vec"), cyc_both=r("trixium_engine_csr_cyc_both"),
                 done=r("trixium_engine_csr_done_count"),
                 pf_hits=r("trixium_engine_csr_pf_hits") if self.has("trixium_engine_csr_pf_hits") else 0,
                 beats=[], ar=[])
        for c in range(self.n_ch):
            self.regs.trixium_engine_csr_perf_chan.write(c)
            s["beats"].append(r("trixium_engine_csr_perf_beats"))
            s["ar"].append(r("trixium_engine_csr_perf_ar"))
        return s

# Status --------------------------------------------------------------------------------------------

def d32(a, b):
    return (b - a) & 0xffffffff

def status(card, interval=0.25):
    """One status dict: identity, state, the counters over `interval`, the host records."""
    st      = card.read("trixium_engine_csr_status")
    ident   = card.identifier()
    cpu     = ("in reset" if card.read("cpumon_cpu_reset") & 1 else "running") if card.has("cpumon_cpu_reset") else "n/a"
    errors  = card.read("ctrl_bus_errors") if card.has("ctrl_bus_errors") else 0
    s0 = card.sample()
    time.sleep(interval)
    s1 = card.sample()
    dt      = s1["time"] - s0["time"]
    cycles  = d32(s0["cyc_total"], s1["cyc_total"])
    util    = lambda k: d32(s0[k], s1[k])/cycles if cycles else 0.0
    clk     = cycles/dt if dt > 0 else 0.0
    share   = card.peak_bytes/(card.n_ch*BEAT_BYTES)   # A channel's beats per clock at the port's peak.
    ch_util = [d32(s0["beats"][c], s1["beats"][c])/cycles/share if cycles else 0.0 for c in range(card.n_ch)]
    gbs     = sum(d32(s0["beats"][c], s1["beats"][c]) for c in range(card.n_ch))*BEAT_BYTES/dt/1e9 if dt > 0 else 0.0
    ar_wait = sum(d32(s0["ar"][c], s1["ar"][c])/cycles if cycles else 0.0 for c in range(card.n_ch))/card.n_ch
    load, chat = state_read("load"), state_read("chat")
    if chat and not pid_alive(chat.get("pid")):
        chat = None
    return dict(identifier=ident, platform=card.platform, clock_mhz=clk/1e6, cpu=cpu,
                bus_errors=errors, engine_error=(st >> 1) & 1, engine_run=(st >> 8) & 1,
                engine_busy=st & 1, seconds=dt, engine=util("cyc_eng"), units=util("cyc_vec"),
                overlap=util("cyc_both"), ops_s=d32(s0["done"], s1["done"])/dt if dt > 0 else 0.0,
                prefetch_s=d32(s0["pf_hits"], s1["pf_hits"])/dt if dt > 0 else 0.0,
                channels=card.n_ch, gbs=gbs, peak_gbs=card.peak_bytes*clk/1e9,
                request_wait=ar_wait, channel_util=ch_util, load=load, chat=chat)

# Table ---------------------------------------------------------------------------------------------

class Table:
    def __init__(self, color):
        self.teal  = "\033[38;2;53;194;174m" if color else ""
        self.dim   = "\033[38;2;95;120;113m" if color else ""
        self.bold  = "\033[1m"               if color else ""
        self.reset = "\033[0m"               if color else ""

    def visible(self, s):
        out, esc = 0, False
        for ch in s:
            if ch == "\033": esc = True
            elif esc: esc = ch != "m"
            else: out += 1
        return out

    def head(self, title, first=False):
        n = WIDTH - 2
        s = self.dim + ("╭" if first else "├")
        if title:
            s += f"─ {self.reset}{self.bold}{self.teal}{title}{self.dim} "
            n -= self.visible(title) + 3
        print(s + "─"*n + ("╮" if first else "┤") + self.reset)

    def line(self, text):
        pad = max(WIDTH - 4 - self.visible(text), 0)
        print(f"{self.dim}│{self.reset} {text}{' '*pad} {self.dim}│{self.reset}")

    def bar(self, x, n):
        k = min(max(int(x*n + 0.5), 0), n)
        return self.teal + "█"*k + self.dim + "░"*(n - k) + self.reset

    def end(self):
        print(self.dim + "╰" + "─"*(WIDTH - 2) + "╯" + self.reset)

def ago(t):
    age = time.time() - t
    return f"{age:.0f} s ago" if age < 120 else f"{age/60:.0f} min ago" if age < 7200 else f"{age/3600:.1f} h ago"

def show(s, where):
    t = Table(sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
    date = time.strftime("%a %b %d %H:%M:%S %Y")
    print(f"{'trixium_smi':<{WIDTH - len(date)}}{date}")
    t.head(f"Card · {where}", first=True)
    t.line(f"{'Bitstream':<12}{s['identifier']}")
    t.line(f"{'Link':<12}{'Etherbone':<26}sys {s['clock_mhz']:.1f} MHz (measured)")
    errors = " ".join(e for e, on in (("engine-err", s["engine_error"]),) if on)
    if s["bus_errors"]:
        errors += f" bus-err {s['bus_errors']}"
    engine = "streaming" if s["engine_run"] else "busy" if s["engine_busy"] else "idle"
    t.line(f"{'State':<12}CPU {s['cpu']:<22}engine {engine:<19}{errors.strip() or 'no errors'}")
    t.head(f"Utilization over {s['seconds']*1e3:.0f} ms")
    t.line(f"Engine  {t.bar(s['engine'], 10)} {100*s['engine']:3.0f}%     "
           f"Units   {t.bar(s['units'], 10)} {100*s['units']:3.0f}%     Overlap {100*s['overlap']:3.0f}%")
    t.line(f"{s['ops_s']:.0f} engine ops/s · {s['prefetch_s']:.0f} served from prefetch/s")
    t.head(f"Memory · {s['channels']} channels")
    t.line(f"Read    {t.bar(s['gbs']/s['peak_gbs'] if s['peak_gbs'] else 0, 10)} {s['gbs']:6.2f} GB/s of "
           f"{s['peak_gbs']:.1f} GB/s port peak   request wait {100*s['request_wait']:3.0f}%")
    for c0 in range(0, s["channels"], 6):
        t.line("  ".join(f"{c:2d} {t.bar(u, 4)}{100*u:3.0f}%"
                         for c, u in enumerate(s["channel_util"]) if c0 <= c < c0 + 6))
    t.head("Model")
    load = s["load"]
    if load:
        t.line(f"{'Weights':<12}{os.path.basename(load.get('image', '?'))}  "
               f"{load.get('bytes', 0)/(1 << 30):.2f} GiB, uploaded {ago(load.get('time', 0))}")
    else:
        t.line(f"{'Weights':<12}unknown (no upload recorded)")
    chat = s["chat"]
    if chat:
        t.line(f"{'Process':<12}pid {chat['pid']} {chat.get('tool', '')}"
               f"{' · ' + chat['model'] if chat.get('model') else ''}  {chat.get('state', '')}, "
               f"{chat.get('tokens', 0):.0f} tokens")
        rate = chat.get("tok_s", 0)
        if rate == 0 and chat.get("state") == "generating":
            t.line(f"{'Decode':<12}{t.bar(0, 10)}   prefill, waiting for the first token")
        else:
            t.line(f"{'Decode':<12}{t.bar(min(rate/40, 1.0), 10)} {rate:6.2f} tok/s   "
                   f"first token {chat.get('ttft', 0):.2f} s")
    else:
        t.line(f"{'Process':<12}none")
    t.end()

# Main ----------------------------------------------------------------------------------------------

def run(csr_csv, host="localhost", port=1234, interval=0.25, watch=0.0, as_json=False, where=None):
    card = Card(csr_csv, host, port)
    try:
        while True:
            s = status(card, interval)
            if as_json:
                print(json.dumps(s))
            else:
                if watch:
                    print("\033[H\033[J", end="")
                show(s, where or f"{host}:{port}")
            if not watch:
                break
            time.sleep(watch)
    except KeyboardInterrupt:
        pass
    finally:
        card.close()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csr-csv",  required=True,             help="The loaded build's csr.csv.")
    parser.add_argument("--host",     default="localhost",       help="litex_server host.")
    parser.add_argument("--port",     default=1234, type=int,    help="litex_server port.")
    parser.add_argument("-i",         default=0.25, type=float,  help="Counter sampling interval (s).", dest="interval")
    parser.add_argument("-w",         default=0.0,  type=float,  help="Refresh every N seconds (0 = once).", dest="watch")
    parser.add_argument("--json",     action="store_true",       help="One JSON object per sample.")
    args = parser.parse_args(argv)
    run(args.csr_csv, args.host, args.port, args.interval, args.watch, args.json)

if __name__ == "__main__":
    main()
