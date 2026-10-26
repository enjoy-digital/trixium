#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""One-command deployment + measurement on a lab board, driven from the workstation over ssh.

  trixium deploy --board xcu1525 --model 27b                  # build/xcu1525 on the XCU1525 host
  trixium deploy --board fk33    --model 1p7b                 # build/fk33 on the FK33 host
  trixium deploy --board ypcb    --model 1p7b                 # build/ypcb on the YPCB-00338 host
  trixium deploy --board xcu1525 --model 8b --no-reconfigure  # next model, board already deployed
  trixium deploy --board xcu1525 --model 27b --dry-run        # print every command, run nothing
  trixium deploy --board ti375   --model 2b                   # build/ti375 on the kit plugged in here

The board hosts come from ~/.config/trixium/boards.ini (doc/boards.ini.example), one section per
board, overridable with TRIXIUM_<BOARD>_<KEY> environment variables (TRIXIUM_XCU1525_HOST=...).
ssh runs in BatchMode: an ssh key must be installed on the host.

The sequence and its order:
  1. timing gate      - refuse a bitstream that missed timing badly (it can take the host down).
                        XCU1525: the strict verdict for information, then the experiment gate under
                        the DDR4-1500 approval; FK33: the empirical gate it has always shipped on;
                        YPCB-00338: the FK33's envelope gate plus the board's clock coverage
  2. firmware         - built for this build tree under firmware/.build.lock (firmware/ is shared
                        with every other build, each against its own CSR map), shipped from a
                        private snapshot (build/<name>/trixium.bin)
  3. BOTH header flavours - the generated soc.h AND the driver-flavour kernel/*.h that user/
                        pulls in: missing the second lets an include guard win and trixium_load
                        silently uploads to the old base
  4. the bitstream archived in build/archive/ (the board-side csr.bit is overwritten by every
     deploy, and build/ gets cleaned for disk space)
  5. driver + host tools built ON THE HOST against its kernel with this build's headers
  6. reconfigure + warm reboot (software/host/board.py reconfigure): XCU1525 openFPGALoader + the IR
     capture 0x35d75 (DONE in the three SLRs); FK33 OpenOCD pld load; YPCB-00338 openFPGALoader
     on a Digilent HS2 cable. Never go on against a removed device
  7. boot + weights: XCU1525 = the DDR4 boot order of software/host/board.py (the BIOS calibrates
     controller 0, the host-side training fixes what it could not, the firmware boots without a
     CPU reset and calibrates 1..3, the host trains them too, engreset, the round trip), THEN the
     weights (the loader writes to controllers the firmware initialises). YPCB-00338 = the same
     order without host training: the BIOS levels both DDR3 controllers (liblitedram's CK phase
     leveling). FK33 = weights, then the serial boot (retried)
  8. tokens BEFORE numbers - wrong tokens stop here with a non-zero exit: a fast wrong answer is
     not a result
  9. bench

Images: models/bons/. The XCU1525's windows are 704 MB (1 GB stride): its 8B and 27B images are
its own (bonsai-<model>-n16-xcu1525.img, trixium convert --board xcu1525); its 1.7B
image is the FK33's, byte-identical. The FK33's image must match the bitstream's channel count
(bonsai-<model>-n<channels>.img, from the build's soc.h). The YPCB-00338's are its own
(bonsai-<model>-n<channels>-ypcb.img, trixium convert --board ypcb: eight channels with 5-trit
codes by default, four windows per DDR3 controller).
"""

import os
import re
import sys
import glob
import time
import fcntl
import shlex
import shutil
import hashlib
import argparse
import subprocess
import configparser

# Paths --------------------------------------------------------------------------------------------

REPO       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOARDS_INI = os.environ.get("TRIXIUM_BOARDS_INI", os.path.expanduser("~/.config/trixium/boards.ini"))

# Models -------------------------------------------------------------------------------------------

# The prompt "The capital of France is" in each model's own vocabulary (the 27B's tokenizer is
# Qwen3.5's, not Qwen3's) and the reference continuation: the fp32 golden of each model (software/
# lab/fp32_check.py / model.golden --raw), checked on the cards.
MODELS = {
    #        tokenizer  prompt                     want                                       bench  timeout
    "1p7b": ("1.7b", "785 6722 315 9625 374", "12095 11 323 279 6722 315 17689 374 24081",    64,  1200), # " Paris, and the capital of Spain is Madrid"
    "8b"  : ("8b",   "785 6722 315 9625 374", "12095 13 576 6722 315 9625 374 12095 13 576",  32,  1200), # " Paris. The capital of France is Paris. The"
    "27b" : ("27b",  "760 6511 314 9338 369", "11751 13 198 760 6511 314 9564 369",            4,  7000), # " Paris.\nThe capital of Germany is" (PrismML's llama.cpp fork, CUDA, greedy)
    "2b"  : ("2b",   "760 6511 314 9338 369", "11751 13 198 760 6511 314 9338 369",            64,  1200), # " Paris.\nThe capital of France is" (Ternary-Bonsai-2-2B, model/golden.py, greedy)
    "4b"  : ("4b",   "760 6511 314 9338 369", "11751 13 198 760 6511 314 279 3516 4042",       32,  1800), # " Paris.\nThe capital of the United States is" (Ternary-Bonsai-2-4B, model/golden.py, greedy)
}

# Boards: the LiteX platform name of each board's bitstream (build/<name>/gateware/<platform>.bit).
PLATFORMS = {
    "xcu1525" : "sqrl_xcu1525",
    "xcu1525-standalone" : "sqrl_xcu1525",   # trixium_xcu1525.py --standalone: no host (software/host/xcu1525.py)
    "fk33"    : "sqrl_fk33",
    "ypcb"    : "ypcb_00338_1p1",
    "ti375"   : "efinix_ti375_c529_dev_kit",
}

# A board whose image computes a model differently has its own reference: the Ti375's 2B image
# has no int16 embedding table (1 GB of memory), so the firmware restores each embedding row from
# the rotated ternary row exactly (model/bons.py restored_row_exact), 1 LSB away from the
# converter's float32 table at most - which flips the 2B's third token. The mirror of that image
# (model/mirror.py) gives this sequence; the card ran it bit-exact.
BOARD_GOLDEN = {
    ("ti375", "2b"): "11751 13 561 6511 314 17163 369 23327",
}

# Board Hosts --------------------------------------------------------------------------------------

BOARD_KEYS = {
    "host"          : None,                     # Board host (name or address).
    "user"          : None,                     # ssh user (default: ssh's).
    "dir"           : "trixium",                # Deploy directory on the host (relative = $HOME).
    "sudo_password" : None,                     # Remote sudo (through SUDO_ASKPASS); none = sudo -n.
    "askpass"       : "/tmp/trixium-askpass",   # The askpass helper the deploy writes on the host.
}

def board_config(board, path=BOARDS_INI):
    """The [board] section of boards.ini, with TRIXIUM_<BOARD>_<KEY> environment overrides."""
    ini = configparser.ConfigParser()
    ini.read(path, encoding="utf-8")
    section = ini[board] if ini.has_section(board) else {}
    conf    = {}
    for key, default in BOARD_KEYS.items():
        conf[key] = os.environ.get(f"TRIXIUM_{board.upper()}_{key.upper()}", section.get(key, default))
    if not conf["host"]:
        raise SystemExit(f"no host for board {board}: add a [{board}] section to {path} "
            "(see doc/boards.ini.example) or set TRIXIUM_" + board.upper() + "_HOST")
    conf["target"] = f"{conf['user']}@{conf['host']}" if conf["user"] else conf["host"]
    return conf

# Remote -------------------------------------------------------------------------------------------

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
SCP = ["scp", "-q", "-o", "BatchMode=yes"]

class DeployError(Exception):
    pass


class Remote:
    """ssh/scp to the board host; with dry_run, every command is printed and none runs."""
    def __init__(self, conf, dry_run=False):
        self.conf    = conf
        self.target  = conf["target"]
        self.dir     = conf["dir"]
        self.dry_run = dry_run

    def show(self, cmd):
        print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)

    def local(self, cmd, cwd=REPO, pattern=None, check=True):
        """A local command; `pattern` filters its output lines. Returns its exit code."""
        if self.dry_run:
            self.show(cmd)
            return 0
        r   = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = r.stdout.splitlines()
        if pattern is not None:
            out = [l for l in out if re.search(pattern, l)]
        if out:
            print("\n".join(out), flush=True)
        if check and r.returncode:
            raise DeployError(f"{cmd[0]} failed ({r.returncode})")
        return r.returncode

    def sudo(self):
        if self.conf["sudo_password"]:
            return f"SUDO_ASKPASS={shlex.quote(self.conf['askpass'])} sudo -A"
        return "sudo -n"

    def ssh(self, cmd, capture=False, check=False, cd=True, input=None, timeout=None):
        """Run `cmd` (a shell command line) on the host, in the deploy directory. Returns
        (exit code, output if captured)."""
        if cd:
            cmd = f"cd {shlex.quote(self.dir)} && {cmd}"
        full = SSH + [self.target, cmd]
        if self.dry_run:
            self.show(full)
            return 0, ""
        try:
            r = subprocess.run(full, input=input, text=True, timeout=timeout,
                stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
        except subprocess.TimeoutExpired:
            return 124, ""
        if check and r.returncode:
            raise DeployError(f"remote command failed ({r.returncode}): {cmd}")
        return r.returncode, (r.stdout or "")

    def scp(self, srcs, dst):
        """Copy local files to <dir>/<dst> on the host (missing sources are reported)."""
        srcs = [s if os.path.isabs(s) else os.path.join(REPO, s) for s in srcs]
        missing = [s for s in srcs if not os.path.exists(s)]
        if missing and not self.dry_run:
            raise DeployError("missing: " + " ".join(os.path.relpath(m, REPO) for m in missing))
        dst = dst if os.path.isabs(dst) else f"{self.dir}/{dst}"
        cmd = SCP + srcs + [f"{self.target}:{dst}"]
        if self.dry_run:
            self.show([os.path.relpath(c, REPO) if os.path.isabs(c) and c.startswith(REPO) else c for c in cmd])
            return
        if subprocess.run(cmd).returncode:
            raise DeployError(f"scp to {dst} failed")

    def write_askpass(self):
        """The SUDO_ASKPASS helper (mode 700; rewritten after a reboot clears /tmp)."""
        if not self.conf["sudo_password"]:
            return
        askpass = shlex.quote(self.conf["askpass"])
        cmd     = f"umask 077 && cat > {askpass} && chmod 700 {askpass}"
        if self.dry_run:
            self.show(SSH + [self.target, cmd + "  # < '#!/bin/sh\\necho <sudo_password>'"])
            return
        script = f"#!/bin/sh\necho {shlex.quote(self.conf['sudo_password'])}\n"
        self.ssh(cmd, cd=False, input=script, capture=True, check=True)

    def wait(self, interval=30, tries=None):
        """Until the host answers ssh (BatchMode, ConnectTimeout: no ping first, so ssh's own host
        config applies, e.g. a ProxyCommand pinning the route to a lab interface)."""
        host = self.conf["host"]
        if self.dry_run:
            self.show(SSH + [self.target, "true"])
            return
        n = 0
        while True:
            if subprocess.run(SSH + [self.target, "true"], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL).returncode == 0:
                return
            n += 1
            if tries is not None and n >= tries:
                raise DeployError(f"{host} does not come back")
            time.sleep(interval)

    def sync(self, path):
        """Copy a (large) file unless the host already has the same size and md5 prefix: a
        regenerated image keeps its name."""
        name = os.path.basename(path)
        cmd  = f"f={shlex.quote(name)}; [ -f \"$f\" ] && echo $(stat -c %s \"$f\")-$(md5sum \"$f\" | cut -c1-12)"
        if self.dry_run:
            self.ssh(cmd)
            self.scp([path], ".")
            return
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                md5.update(chunk)
        lsum = f"{os.path.getsize(path)}-{md5.hexdigest()[:12]}"
        _, rsum = self.ssh(cmd, capture=True)
        if rsum.strip() != lsum:
            print(f"copying {os.path.relpath(path, REPO)} ({lsum}, board has '{rsum.strip()}')", flush=True)
            self.scp([path], ".")

# Helpers ------------------------------------------------------------------------------------------

def say(text):
    print(f"\n########## {text} ({time.strftime('%H:%M:%S')})", flush=True)


def soc_constant(soc_h, name, default=None):
    try:
        with open(soc_h, encoding="utf-8") as f:
            m = re.search(rf"#define {name} (\S+)", f.read())
    except OSError:
        return default
    return m.group(1) if m else default


def build_firmware(remote, out):
    """The firmware for build tree `out`, under firmware/.build.lock, copied to out/trixium.bin
    (a failed build must not ship the last one)."""
    fw_dir = os.path.join(REPO, "firmware")
    binary = os.path.join(out, "trixium.bin")
    if remote.dry_run:
        print(f"# under flock {os.path.relpath(fw_dir, REPO)}/.build.lock: rm firmware/*.o firmware/trixium.bin")
        remote.local(["make", "-C", "firmware", f"BUILD_DIR={out}"])
        remote.show(["cp", "firmware/trixium.bin", os.path.relpath(binary, REPO)])
        return binary
    with open(os.path.join(fw_dir, ".build.lock"), "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for f in glob.glob(os.path.join(fw_dir, "*.o")) + [os.path.join(fw_dir, "trixium.bin"), binary]:
            if os.path.exists(f):
                os.remove(f)
        remote.local(["make", "-C", "firmware", f"BUILD_DIR={out}"], pattern=r"(?i) error|OBJCOPY", check=False)
        if os.path.exists(os.path.join(fw_dir, "trixium.bin")):
            shutil.copy(os.path.join(fw_dir, "trixium.bin"), binary)
    if not os.path.exists(binary):
        raise DeployError("firmware build failed")
    with open(binary, "rb") as f:
        print(f"{hashlib.sha256(f.read()).hexdigest()}  {os.path.relpath(binary, REPO)}")
    return binary

# Deploy -------------------------------------------------------------------------------------------

def deploy(build, board, model, reconfigure=True, boot=False, dry_run=False, approval=None,
    serializer_baseline="v8", bench=None):
    if board == "fk33" and model == "27b":
        raise DeployError("the 27B does not fit the FK33's 8 GB HBM")
    if board == "ypcb" and model == "27b":
        raise DeployError("the 27B does not fit the YPCB-00338's 4 GB DDR3 (nor its units)")
    if board == "fk33" and model in ("2b", "4b"):
        raise DeployError("the FK33 bitstream has no hybrid-model units")
    tok, prompt, want, bench_n, timeout = MODELS[model]
    bench_n  = bench_n if bench is None else bench
    conf     = board_config(board)
    remote   = Remote(conf, dry_run)
    out      = os.path.join(REPO, "build", build)
    name     = os.path.basename(out)
    gw       = os.path.join(out, "gateware")
    platform = PLATFORMS.get(board, f"sqrl_{board}")
    bit      = os.path.join(gw, f"{platform}.bit")
    report   = os.path.join(gw, f"{platform}_timing.rpt")
    soc_h    = os.path.join(out, "software", "include", "generated", "soc.h")
    drv      = os.path.join(out, "driver", "kernel")
    sudo     = remote.sudo()
    board_py = f"{sudo} python3 -m software.host.board"
    stem = f"bonsai2-{model}" if model in ("2b", "4b") else f"bonsai-{model}" # cli.image_stem's names.
    if board == "xcu1525":
        image = f"{stem}-n16.img" if model == "1p7b" else f"{stem}-n16-xcu1525.img"
    elif board == "ypcb":
        # A build on fewer controllers than the board's two (--engine-memctrl 1) has its own
        # windows: `trixium convert --memctrl 1` names its image -m1 (software/cli.py).
        memctrl = soc_constant(soc_h, "TRIXIUM_N_MEMCTRL", "2")
        msuffix = "" if str(memctrl) == "2" else f"-m{memctrl}"
        image = f"{stem}-n{soc_constant(soc_h, 'TRIXIUM_N_CHANNELS', '8')}{msuffix}-ypcb.img"
        if model in ("2b", "4b") and soc_constant(soc_h, "TRIXIUM_DNSTREAM_LANES") is None:
            raise DeployError(f"{name} has no hybrid-model units (build trixium_ypcb.py --hybrid)")
    else:
        image = f"bonsai-{model}-n{soc_constant(soc_h, 'TRIXIUM_N_CHANNELS', '16')}.img"
    image = os.path.join(REPO, "models", "bons", image)
    print(f"deploy {os.path.relpath(out, REPO)} -> {board} at {conf['target']}:{conf['dir']}, "
          f"model {model} ({os.path.relpath(image, REPO)})")

    # Timing gate.
    if reconfigure:
        say("timing gate")
        timing = [sys.executable, "-m", "software.lab.timing"]
        if board == "xcu1525":
            # Deploy on the experiment gate (concepts are validated on hardware even when timing
            # does not close), which still refuses anything outside the measured host-safety
            # envelope; the strict release verdict is printed alongside.
            print("release gate:")
            remote.local(timing + ["--strict", "--serializer-baseline", serializer_baseline, report], check=False)
            gate = timing + ["--experiment", "--serializer-baseline", serializer_baseline] + \
                (["--approval", approval] if approval else []) + [report]
        elif board == "ypcb":
            gate = timing + ["--board", "ypcb", report]
        else:
            gate = timing + [report]
        if remote.local(gate, check=False):
            raise DeployError(f"refusing {os.path.relpath(out, REPO)}")
    if not os.path.exists(image):
        message = f"no image {os.path.relpath(image, REPO)} (convert it first: trixium convert)"
        if not dry_run:
            raise DeployError(message)
        print("warning: " + message)

    # Board host.
    say("wait for the board host")
    remote.wait()
    remote.write_askpass()

    # Firmware + headers.
    say(f"firmware + headers from {os.path.relpath(out, REPO)}")
    if not os.path.exists(os.path.join(drv, "soc.h")) and not dry_run:
        raise DeployError(f"no {os.path.relpath(out, REPO)}/driver/: rebuild with ./trixium_{board}.py --build")
    firmware = build_firmware(remote, out)
    for constant in ["TRIXIUM_N_CHANNELS", "TRIXIUM_CHAN_WEIGHT_BASE", "TRIXIUM_CHAN_STRIDE"]:
        print(f"{constant} {soc_constant(soc_h, constant, '?')}")

    # Copy.
    remote.ssh(f"mkdir -p {shlex.quote(conf['dir'])} && cd {shlex.quote(conf['dir'])} && mkdir -p prog "
        "gen/software/include/generated gen/driver/kernel gen/driver/user/liblitepcie user lib "
        "software/host software/lab models/1.7b models/8b models/27b", cd=False, check=True)
    # The remote tree keeps the repository's layout, so `python3 -m software.host.X` (and the
    # card-side clients software.lab.mirror / software.lab.fp32_check) import the same way there.
    tools = {
        "software"      : ["__init__.py", "client.py"],
        "software/host" : ["__init__.py", "board.py", "console.py", "cpurst.py", "cpumon.py", "rdlevel.py", "driver.py"],
        "software/lab"  : ["__init__.py", "mirror.py", "fp32_check.py"],
    }
    for directory, files in tools.items():
        files   = [os.path.join(directory, f) for f in files]
        present = [f for f in files if os.path.exists(os.path.join(REPO, f))]
        for f in sorted(set(files) - set(present)):
            print(f"warning: {f} not found, not copied")
        remote.scp(present, f"{directory}/")
    if reconfigure:
        say("archive + copy")
        archive = os.path.join(REPO, "build", "archive")
        if dry_run:
            remote.show(["cp", os.path.relpath(bit, REPO), f"build/archive/{name}.bit"])
            remote.show(["cp", os.path.relpath(report, REPO), f"build/archive/{name}_timing.rpt"])
        else:
            os.makedirs(archive, exist_ok=True)
            shutil.copy(bit,    os.path.join(archive, f"{name}.bit"))
            shutil.copy(report, os.path.join(archive, f"{name}_timing.rpt"))
        if board == "fk33":
            # The FK33's OpenOCD configuration (litex-boards) for software/host/board.py reconfigure.
            import litex_boards
            cfg = os.path.join(os.path.dirname(litex_boards.__file__), "prog", "openocd_xcvu33p_ft2232.cfg")
            remote.scp([cfg], "openocd_fk33.cfg")
        remote.scp([bit], "prog/csr.bit")
        generated = [os.path.join(out, "software", "include", "generated", h) for h in ["csr.h", "soc.h", "mem.h"]]
        remote.scp(generated, "gen/software/include/generated/")
        # The driver: the build's generated LitePCIe driver (name "trixium": trixium.ko,
        # /dev/trixium0) and liblitepcie, built ON THE HOST against its kernel.
        remote.scp(sorted(glob.glob(os.path.join(drv, "*"))), "gen/driver/kernel/")
        remote.scp(sorted(glob.glob(os.path.join(out, "driver", "user", "liblitepcie", "*.[ch]"))),
            "gen/driver/user/liblitepcie/")
        # Every host-tool source: the loader module, the tools and their Makefile change together.
        remote.scp([os.path.join("software", "user", f) for f in ["trixium_loader.c", "trixium_loader.h", "trixium_state.c",
            "trixium_load.c", "trixium_chat.c", "trixium_smi.c", "Makefile"]], "user/")
        # trixium_chat's tokenizer is the firmware's (software/lib, ../lib from user/).
        remote.scp([os.path.join("software", "lib", f) for f in ["trixium_tok.c", "trixium_tok.h"]], "lib/")
        # The tokenizers (the C chat tool wants the .bin, the Python client the .json).
        for m in ["1.7b", "8b", "27b", "2b", "4b"]:
            if os.path.exists(os.path.join(REPO, "models", m, "tokenizer.bin")):
                remote.ssh(f"mkdir -p models/{m}")
                remote.scp([os.path.join("models", m, f) for f in ["tokenizer.bin", "tokenizer.json"]], f"models/{m}/")
        rc, _ = remote.ssh("cd gen/driver/kernel && rm -f *.ko *.o *.mod* .*.cmd && (make 2>&1 | grep -iE ' error' | head -3); "
            "ls trixium.ko liteuart.ko")
        if rc:
            raise DeployError("driver build failed on the host")

        # Reconfigure + reboot.
        say("reconfigure + reboot")
        rc, _ = remote.ssh(f"{board_py} reconfigure --board {board} prog/csr.bit")
        # 1 = the board-side stop (JTAG busy, not configured); a reboot closing the connection is
        # 255. Never go on against a removed device.
        if rc not in [0, 255]:
            raise DeployError("reconfigure failed - stop")
        if not dry_run:
            time.sleep(30)
        remote.wait(interval=15, tries=40)
        if not dry_run:
            time.sleep(15)
        remote.write_askpass()

        # Drivers + host tools.
        say("drivers + host tools")
        remote.ssh(f"{sudo} python3 -m software.host.driver load --dir gen/driver/kernel", check=True)
        remote.ssh(f"{sudo} dmesg | grep -m1 'Version TriXium'; lspci -nn | grep -E '10ee:'")
        remote.ssh("cd user && rm -f *.o *.d trixium_load trixium_chat trixium_smi && "
            "(make BUILD_DIR=../gen 2>&1 | grep -i error | head -3); ls trixium_load trixium_chat trixium_smi", check=True)

    # Boot, weights, tokens first.
    if reconfigure or boot or board == "fk33":
        remote.scp([firmware], "trixium.bin")
    say(f"weights + boot + tokens first - want: {want} ...")
    remote.sync(image)
    img = os.path.basename(image)
    gen = f"16 {prompt} --want {shlex.quote(want)} --timeout {timeout}"
    if board in ["xcu1525", "ypcb"]:
        if reconfigure or boot:
            if remote.ssh(f"{board_py} boot {tok} --board {board}")[0]:
                raise DeployError("board boot or DDR training failed - refusing to load weights")
        if remote.ssh(f"{board_py} model {img} {gen}")[0]:
            raise DeployError("model load, generation or TOKENS failed - refusing to benchmark")
    else:
        remote.ssh(f"{board_py} load {img}")
        if remote.ssh(f"{board_py} boot {tok} --board fk33")[0]:
            raise DeployError("firmware boot failed (no TRIXIUM-READY) - refusing to go on")
        if remote.ssh(f"{board_py} gen {gen}")[0]:
            raise DeployError("generation or TOKENS failed - refusing to benchmark")

    # Bench.
    say(f"measure (bench {bench_n})")
    remote.ssh(f"{board_py} bench {bench_n}")
    print("\n  Same-boot numbers on one calibration are the only ones that compare to 1%; boots drift\n"
          "  ~4-5%. References: doc/boards.md.")
    return 0

# Ti375 --------------------------------------------------------------------------------------------

class Local(Remote):
    """The Ti375 kit hangs off this workstation (USB JTAG + UART, the wired NIC): local commands
    only (the firmware build is shared with the board-host flow)."""
    def __init__(self, dry_run=False):
        Remote.__init__(self, {"target": "local", "dir": REPO}, dry_run)


def deploy_ti375(build, model, reconfigure=True, dry_run=False, bench=None, firmware=None, check=False):
    """The kit's sequence: firmware, the bitstream archived and loaded volatile over JTAG, the
    serial boot, the weights over UDP (software/host/udpload.py), hdrload + imgcheck, tokens
    before numbers, bench. ~/.config/trixium/boards.ini [ti375] (or TRIXIUM_TI375_<KEY>): tty
    (/dev/ttyUSB2), ip (<board-host>), iface (the NIC that reaches the board, when another one
    routes its subnet)."""
    from software.host import ti375
    from software.host.console import run as console_run
    if model not in ("2b",):
        raise DeployError("the Ti375's 1 GB holds the 2B (bonsai2-2b-n4-ti375.img); the 4B does not fit")
    tty, ip, iface = ti375.config()
    tok, prompt, want, bench_n, timeout = MODELS[model]
    want    = BOARD_GOLDEN.get(("ti375", model), want)
    bench_n = bench_n if bench is None else bench
    local   = Local(dry_run)
    out     = os.path.join(REPO, "build", build)
    name    = os.path.basename(out)
    bit     = os.path.join(out, "gateware", f"{PLATFORMS['ti375']}.bit")
    soc_h   = os.path.join(out, "software", "include", "generated", "soc.h")
    image   = os.path.join(REPO, "models", "bons",
                           f"bonsai2-{model}-n{soc_constant(soc_h, 'TRIXIUM_N_CHANNELS', '4')}-ti375.img")
    print(f"deploy {os.path.relpath(out, REPO)} -> ti375 (tty {tty}, {ip}{' via ' + iface if iface else ''}), "
          f"model {model} ({os.path.relpath(image, REPO)})")
    if dry_run:
        for step in ["firmware", "archive + JTAG load (volatile)", "serial boot", "litex_server",
                     "udpload", "hdrload + imgcheck", f"gen {prompt} (want {want})", f"bench {bench_n}"]:
            print(f"# {step}")
        return 0

    say("firmware")
    # --firmware: a binary built elsewhere (a private tree while a simulation compiles in the
    # shared firmware/), for this build's CSR map.
    firmware = firmware or build_firmware(local, out)
    if reconfigure:
        say("bitstream (JTAG, volatile) + serial boot")
        archive = os.path.join(REPO, "build", "archive")
        os.makedirs(archive, exist_ok=True)
        shutil.copy(bit, os.path.join(archive, f"{name}.bit"))
        # litex_term attached before the load answers the BIOS's serial boot at once (and ahead
        # of a firmware in the flash).
        boot = ti375.boot(firmware, tty, bit=bit)
        print("\n".join(l for l in boot.splitlines() if l.startswith(("TriXium", "BONS", "SPTEST", "hybrid: emb"))))
    else:
        # --no-reconfigure: the firmware already running (the BIOS's serial boot needs its prompt:
        # a new firmware means a new load).
        out_txt = console_run([""], tty, wait=5, baud=ti375.BAUD)
        if "trixium>" not in out_txt:
            raise DeployError("no TriXium firmware answers on the console - deploy without --no-reconfigure")
    if not ti375.ping(ip, iface):
        raise DeployError(f"the board does not answer at {ip}")
    say("weights over UDP")
    # The upload (each transfer CRC-checked on builds with the port's CRC), the host's load record
    # (smi, `trixium chat --board ti375`'s skip), hdrload.
    csr_csv = os.path.join(out, "csr.csv")
    _, dt = ti375.ensure_model(image, csr_csv, force=True)
    print(f"{os.path.getsize(image)/1e6:.1f} MB uploaded, ready in {dt:.1f} s")
    if check or "udpload_crc" not in open(csr_csv).read():
        # The deep check: every tensor fetched back and its CRC compared (~7 min on the CPU).
        out_txt = console_run(["imgcheck"], tty, wait=900, until=" bad", baud=ti375.BAUD)   # "imgcheck: ... N bad"
        print("\n".join(l for l in out_txt.splitlines() if l.startswith("imgcheck")))
        if " 0 bad" not in out_txt:
            raise DeployError("weights wrong on the board - refusing to go on")
    say(f"tokens first - want: {want} ...")
    out_txt = console_run([f"gen 8 {prompt}"], tty, wait=timeout, until="GEN DONE", baud=ti375.BAUD)
    got = next((l[4:].strip() for l in out_txt.splitlines() if l.startswith("GEN:")), "")
    print(f"got:  {got}")
    if got != want:
        raise DeployError("TOKENS differ - refusing to benchmark")
    say(f"measure (bench {bench_n})")
    out_txt = console_run([f"bench {bench_n}"], tty, wait=3600, until="BENCH_PHASE", baud=ti375.BAUD)
    print("\n".join(l for l in out_txt.splitlines() if l.startswith("BENCH")))
    return 0

# XCU1525 standalone ---------------------------------------------------------------------------------

def deploy_xcu1525_standalone(build, model, reconfigure=True, dry_run=False, bench=None, firmware=None,
                              persist=False):
    """The standalone card (trixium_xcu1525.py --standalone, doc/plan-xcu1525-standalone.md) from the
    workstation its JTAG, console and network reach (software/host/xcu1525.py; ~/.config/trixium/
    boards.ini [xcu1525]: tty, ip, iface): firmware, the bitstream archived and loaded volatile
    over JTAG, the firmware booted from the console's serial boot, the weights over UDP into the
    DRAM (or, with `persist`, into an SSD slot the firmware then boots), tokens before numbers,
    bench. The card trains its own DDR4 (liblitedram's write training): the boot waits for it.
    Not yet run on hardware: --dry-run lists the steps."""
    from software.host import xcu1525
    from software.host.console import run as console_run
    tty, ip, iface = xcu1525.config()
    tok, prompt, want, bench_n, timeout = MODELS[model]
    bench_n = bench_n if bench is None else bench
    local   = Local(dry_run)
    out     = os.path.join(REPO, "build", build)
    name    = os.path.basename(out)
    bit     = os.path.join(out, "gateware", f"{PLATFORMS['xcu1525-standalone']}.bit")
    report  = os.path.join(out, "gateware", f"{PLATFORMS['xcu1525-standalone']}_timing.rpt")
    stem    = f"bonsai2-{model}" if model in ("2b", "4b") else f"bonsai-{model}"
    image   = os.path.join(REPO, "models", "bons", f"{stem}-n16.img" if model == "1p7b" else f"{stem}-n16-xcu1525.img")
    tokenizer = os.path.join(REPO, "models", tok, "tokenizer.bin")
    print(f"deploy {os.path.relpath(out, REPO)} -> xcu1525 standalone (tty {tty}, {ip}{' via ' + iface if iface else ''}), "
          f"model {model} ({os.path.relpath(image, REPO)}{', to the SSD' if persist else ''})")
    if dry_run:
        for step in ["timing gate", "firmware", "archive + JTAG load (volatile)", "boot: BIOS + the card's DDR4 training, "
                     "serial boot of the firmware, TRIXIUM-READY", "litex_server", "model --persist (SSD slot 0) + stboot" if persist
                     else "udpload + hdrload", f"gen {prompt} (want {want})", f"bench {bench_n}"]:
            print(f"# {step}")
        return 0
    if reconfigure:
        say("timing gate")
        timing = [sys.executable, "-m", "software.lab.timing"]
        print("release gate:")
        local.local(timing + ["--strict", "--serializer-baseline", "v8", report], check=False)
        if local.local(timing + ["--experiment", "--serializer-baseline", "v8", "--approval", "ddr4-overclock", report], check=False):
            raise DeployError(f"refusing {os.path.relpath(out, REPO)}")
    say("firmware")
    firmware = firmware or build_firmware(local, out)
    if reconfigure:
        say("bitstream (JTAG, volatile) + serial boot")
        archive = os.path.join(REPO, "build", "archive")
        os.makedirs(archive, exist_ok=True)
        shutil.copy(bit, os.path.join(archive, f"{name}.bit"))
        # The bitstream over JTAG; the BIOS trains controller 0 (its retries up to three times), then
        # its serial boot: the firmware through litex_term (ti375.boot's pty dance), which then
        # trains controllers 1-3 and probes the SSD. Up to 15 minutes before TRIXIUM-READY.
        from software.host import ti375
        xcu1525.load(bit)
        ti375.boot(firmware, tty, timeout=900)
    say("weights" + (" -> SSD slot 0" if persist else " over UDP"))
    csr_csv = os.path.join(out, "csr.csv")
    xcu1525.model(image, csr_csv, tokenizer if os.path.exists(tokenizer) else None,
                  persist_slot=0 if persist else None, default=persist, verbose=True)
    say(f"tokens first - want: {want} ...")
    out_txt = console_run([f"gen 8 {prompt}"], tty, wait=timeout, until="GEN DONE", baud=xcu1525.BAUD)
    got = next((l[4:].strip() for l in out_txt.splitlines() if l.startswith("GEN:")), "")
    print(f"got:  {got}")
    if got != want:
        raise DeployError("TOKENS differ - refusing to benchmark")
    say(f"measure (bench {bench_n})")
    out_txt = console_run([f"bench {bench_n}"], tty, wait=3600, until="BENCH_PHASE", baud=xcu1525.BAUD)
    print("\n".join(l for l in out_txt.splitlines() if l.startswith("BENCH")))
    return 0

# Run ----------------------------------------------------------------------------------------------

def add_arguments(parser):
    parser.add_argument("build",                 nargs="?",                                      help="Build name in build/ (default: the board's).")
    parser.add_argument("--board",               default="xcu1525", choices=list(PLATFORMS), help="Board.")
    parser.add_argument("--model",               default="1p7b", choices=list(MODELS),           help="Model.")
    parser.add_argument("--no-reconfigure",      action="store_true",                            help="Board already deployed: firmware, weights, tokens and bench only.")
    parser.add_argument("--boot",                action="store_true",                            help="With --no-reconfigure: boot the (rebuilt) firmware first (XCU1525).")
    parser.add_argument("--approval",            default="ddr4-overclock",                    help="XCU1525 timing gate approval (software/lab/timing.py; '' for none).")
    parser.add_argument("--serializer-baseline", default="v8",                                   help="XCU1525 timing gate serializer baseline.")
    parser.add_argument("--bench",               type=int,                                       help="bench n (default: per model).")
    parser.add_argument("--check",               action="store_true",                            help="--board ti375: imgcheck after the upload (the deep check; builds with the upload CRC skip it).")
    parser.add_argument("--dry-run",             action="store_true",                            help="Print every command, run nothing.")
    parser.add_argument("--firmware",            default=None,                                   help="Ti375: ship this firmware binary (built for the build's CSR map) instead of building one.")
    parser.add_argument("--persist",             action="store_true",                            help="--board xcu1525-standalone: the model into SSD slot 0 (the power-up boot) instead of the DRAM.")


def run(args):
    try:
        if args.board == "ti375":
            return deploy_ti375(args.build or args.board, args.model, not args.no_reconfigure,
                                args.dry_run, args.bench, args.firmware, args.check)
        if args.board == "xcu1525-standalone":
            return deploy_xcu1525_standalone(args.build or "xcu1525_standalone", args.model, not args.no_reconfigure,
                                             args.dry_run, args.bench, args.firmware, args.persist)
        return deploy(args.build or args.board, args.board, args.model,
            reconfigure         = not args.no_reconfigure,
            boot                = args.boot,
            dry_run             = args.dry_run,
            approval            = args.approval or None,
            serializer_baseline = args.serializer_baseline,
            bench               = args.bench)
    except DeployError as e:
        print(f"STOP: {e}", flush=True)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deploy a build on a lab board and measure it.")
    add_arguments(parser)
    sys.exit(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
