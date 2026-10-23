#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Run one of Efinity's programming scripts with only one FTDI device visible.

Efinity's ftdi_program.py names FTDI devices by VID:PID and serial number; two FT4232H without a
serial number (the Ti375 kit and any other quad adapter on the machine) get the same URLs and it
refuses to program. This wrapper, run with Efinity's own Python, filters pyusb's device lookup -
and pyftdi's own enumeration (UsbTools._find_devices, straight on the libusb backend) down to
the device at one USB bus/address, then runs the script unchanged:

  <efinity>/bin/python3 efx_select.py <bus> <address> <efinity>/pgm/bin/efx_pgm/ftdi_program.py ...

It also makes the JTAG bridge's page programs wait the flash's maximum program time: the driver
does not poll the busy bit after a page program, it sleeps the datasheet's typical time (0.2 ms
on the kit's IS25WP512M, 0.8 ms max), and only after a failed verify erases and writes everything
again with the maximum - most first passes failed. (Erases poll the busy bit.)
"""

import os
import sys
import runpy

import usb.core


def main():
    bus, address, script = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    sys.argv = [script] + sys.argv[4:]
    lookup = usb.core.find

    def visible(dev):
        return dev.idVendor != 0x0403 or (dev.bus == bus and dev.address == address)

    def find(*args, **kwargs):
        if kwargs.get("find_all"):
            return (d for d in lookup(*args, **kwargs) if visible(d))
        kwargs["find_all"] = True
        return next((d for d in lookup(*args, **kwargs) if visible(d)), None)

    usb.core.find = find
    from pyftdi.usbtools import UsbTools
    pyftdi_find = UsbTools._find_devices.__func__

    def find_devices(cls, *args, **kwargs):
        return {d for d in pyftdi_find(cls, *args, **kwargs) if visible(d)}

    UsbTools._find_devices = classmethod(find_devices)
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    pgm_bin = os.path.dirname(os.path.dirname(os.path.abspath(script)))
    home    = os.environ.get("EFINITY_HOME", os.path.dirname(os.path.dirname(pgm_bin)))
    sys.path[:0] = [pgm_bin]
    sys.path.append(os.path.join(home, "debugger", "bin"))   # As ftdi_program.py does (efx_dbg).
    from efx_pgm.jtag2SpiFlash import _SpiFlashDevice
    _SpiFlashDevice._wait_for_completion_fast = _SpiFlashDevice._wait_for_completion_slow
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
