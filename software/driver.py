#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Load/unload trixium.ko + liteuart.ko and open up the device nodes.

  trixium driver load                         # build/xcu1525/driver/kernel (the build's driver)
  trixium driver load --dir build/fk33/driver/kernel
  trixium driver unload
  sudo python3 -m software.driver load --dir gen/driver/kernel   # on a board host

The driver is the build's generated LitePCIe driver (name "trixium"). Runs itself through sudo
when not root.
"""

import os
import sys
import glob
import argparse
import subprocess

# Paths --------------------------------------------------------------------------------------------

KERNEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build", "xcu1525", "driver", "kernel")

# Helpers ------------------------------------------------------------------------------------------

def loaded(module):
    with open("/proc/modules", encoding="utf-8") as f:
        return any(line.split(" ", 1)[0] == module for line in f)


def insmod(path):
    """insmod; returns its error text ("" when loaded)."""
    r = subprocess.run(["insmod", path], capture_output=True, text=True)
    return "" if r.returncode == 0 else (r.stderr.strip() or f"insmod {path} failed")


def make(kdir, *targets):
    subprocess.run(["make", "-s", "-C", kdir, *targets], check=True)

# Load / Unload ------------------------------------------------------------------------------------

def load(kdir=KERNEL_DIR):
    """Load trixium.ko then liteuart.ko from `kdir`, rebuilding trixium.ko when the running
    kernel refuses it. Returns 0 on success."""
    if loaded("trixium"):
        print("trixium module already installed.")
        return 0

    # liteuart binds on top of trixium: reload it after the PCIe driver.
    if loaded("liteuart"):
        subprocess.run(["rmmod", "liteuart"])

    # trixium.ko.
    ko  = os.path.join(kdir, "trixium.ko")
    err = insmod(ko)
    if "Invalid module format" in err:
        print("Kernel may have changed, try to rebuild module")
        make(kdir, "clean")
        make(kdir)
        err = insmod(ko)
    elif "No such file or directory" in err:
        print("Module not compiled")
        make(kdir)
        err = insmod(ko)
    if "Required key not available" in err or "unsigned module" in err:
        print("Can't insert kernel module, secure boot is probably enabled")
        print("Please disable it from BIOS")
        return 1
    if err:
        print(err, file=sys.stderr)
        return 1

    # liteuart.ko.
    err = insmod(os.path.join(kdir, "liteuart.ko"))
    if err:
        print(err, file=sys.stderr)

    # Open up the device nodes.
    for node in glob.glob("/dev/trixium[0-9]*") + glob.glob("/dev/ttyLXU[0-9]*"):
        os.chmod(node, 0o666)
    return 0


def unload():
    """Remove liteuart then trixium (liteuart binds on top of trixium)."""
    for module in ["liteuart", "trixium"]:
        if loaded(module):
            subprocess.run(["rmmod", module])
    return 0

# Run ----------------------------------------------------------------------------------------------

def main(argv=None):
    argv   = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Load/unload the TriXium kernel modules.")
    parser.add_argument("action", choices=["load", "unload"], help="Load or unload trixium.ko + liteuart.ko.")
    parser.add_argument("--dir",  default=KERNEL_DIR,         help="Directory of trixium.ko/liteuart.ko (and their Makefile).")
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd  = ["sudo", sys.executable, "-m", "software.driver", args.action, "--dir", os.path.abspath(args.dir)]
        sys.exit(subprocess.call(cmd, cwd=repo))
    sys.exit(load(args.dir) if args.action == "load" else unload())


if __name__ == "__main__":
    main()
