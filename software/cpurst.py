#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Pulse the SoC's CPU reset over BAR0, to get the BIOS serialboot prompt back.

Firmware loads over serial, and the BIOS only listens for it right after a CPU reset. Once
the firmware is running, `trixium_chat -k` just reports "no magic from the BIOS" - reloading
firmware needs this first.

CPU reset (bit 1), never SoC reset (bit 0): bit 0 takes the PCIe endpoint down with it and
the host loses the device until a warm reboot. HBM keeps its contents across a CPU reset,
so the uploaded weights survive.

  sudo python3 -m software.cpurst
"""

import os
import sys
import glob
import mmap
import time
import argparse

# Constants ----------------------------------------------------------------------------------------

CTRL_RESET_OFFSET = 0x0   # CSR_CTRL_RESET_ADDR - CSR_BASE
CPU_RESET_BIT     = 1 << 1

# Helpers ------------------------------------------------------------------------------------------

def find_xilinx_device():
    """PCI address of the first Xilinx (0x10ee) device, or None."""
    for path in sorted(glob.glob("/sys/bus/pci/devices/*/vendor")):
        with open(path, encoding="utf-8") as f:
            if f.read().strip() == "0x10ee":
                return os.path.basename(os.path.dirname(path))
    return None

# CPU Reset ----------------------------------------------------------------------------------------

def cpu_reset(dev=None):
    """Pulse ctrl_reset's CPU reset bit on `dev` (default: the first Xilinx device)."""
    dev = dev or find_xilinx_device()
    if dev is None:
        raise OSError("no Xilinx device found")
    fd = os.open(f"/sys/bus/pci/devices/{dev}/resource0", os.O_RDWR | os.O_SYNC)
    m  = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    m.seek(CTRL_RESET_OFFSET)
    m.write(CPU_RESET_BIT.to_bytes(4, "little"))
    time.sleep(0.05)
    m.seek(CTRL_RESET_OFFSET)
    m.write((0).to_bytes(4, "little"))
    m.close()
    os.close(fd)
    return dev

# Run ----------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Pulse the SoC's CPU reset over BAR0.")
    parser.add_argument("-d", "--device", help="PCI address, e.g. 0000:01:00.0 (default: first Xilinx).")
    args = parser.parse_args(argv)
    try:
        dev = cpu_reset(args.device)
    except OSError as e:
        sys.exit(str(e))
    print(f"{dev}: CPU reset pulsed - the BIOS should be at the serialboot prompt")


if __name__ == "__main__":
    main()
