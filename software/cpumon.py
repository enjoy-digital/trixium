#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Print the CPU bus monitor (gateware/common.py) twice, 0.5 s apart, over BAR0.

  sudo python3 -m software.cpumon [--csr-h gen/software/include/generated/csr.h]
"""

import os
import re
import mmap
import time
import struct
import argparse

from software.cpurst import find_xilinx_device

# CPU Monitor --------------------------------------------------------------------------------------

def cpumon(csr_h="gen/software/include/generated/csr.h"):
    """The monitor's registers, two snapshots 0.5 s apart, as printable lines."""
    with open(csr_h, encoding="utf-8") as f:
        text = f.read()
    regs = [(m.group(1).lower(), int(m.group(2), 16)) for m in
            re.finditer(r"#define CSR_CPUMON_(\w+)_ADDR \(CSR_BASE \+ (0x[0-9a-fA-F]+)L?\)", text)]
    if not regs:
        raise OSError("no CSR_CPUMON_* in " + csr_h)
    dev = find_xilinx_device()
    if dev is None:
        raise OSError("no Xilinx device found")
    fd = os.open(f"/sys/bus/pci/devices/{dev}/resource0", os.O_RDWR | os.O_SYNC)
    m  = mmap.mmap(fd, os.fstat(fd).st_size or 0x20000)
    rd = lambda o: struct.unpack("<I", m[o:o + 4])[0]

    # Two snapshots: a register that differs between them is live.
    snaps = []
    for _ in range(2):
        snaps.append({n: rd(o) for n, o in regs})
        time.sleep(0.5)
    m.close()
    os.close(fd)
    lines = []
    for n, _ in regs:
        a0, a1 = snaps[0][n], snaps[1][n]
        lines.append(f"{n:12s} 0x{a0:08x} -> 0x{a1:08x}{'' if a0 == a1 else '  (moving)'}")
    return lines

# Run ----------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Print the CPU bus monitor CSRs over BAR0.")
    parser.add_argument("--csr-h", default="gen/software/include/generated/csr.h", help="Generated csr.h.")
    args = parser.parse_args(argv)
    try:
        print("\n".join(cpumon(args.csr_h)))
    except OSError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
