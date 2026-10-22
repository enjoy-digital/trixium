#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Scripted console over the PCIe liteuart tty, for automated hardware checks.

litex_term wants an interactive terminal; this just sends commands and prints what comes back.

  python3 -m software.host.console                      # drain whatever the SoC has printed
  python3 -m software.host.console -c selftest -w 240   # run a command, print the reply
  trixium console "stats" "bench 64"               # the same from the CLI
"""

import os
import sys
import time
import fcntl
import termios
import argparse

# Console ------------------------------------------------------------------------------------------

class ConsoleBusy(OSError):
    pass


class Console:
    """The firmware console on a liteuart tty, raw and non-blocking, locked for one session."""
    def __init__(self, tty="/dev/ttyLXU0", baud=None):
        # baud: a serial UART's rate (the Ti375 kit's FTDI port, 115200); None for the PCIe
        # liteuart tty, which has none.
        self.tty = tty
        self.fd  = os.open(tty, os.O_RDWR | os.O_NOCTTY)
        # One console session at a time: the firmware reads one byte stream, and two readers
        # split its replies between them (an idle trixium_chat left at its prompt is enough).
        # Advisory lock, shared by every console tool.
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            raise ConsoleBusy(f"{tty} is in use by another console session (trixium_chat, "
                "software.host.console, the Python client, litex_term?) - close it first")
        a = termios.tcgetattr(self.fd)
        a[0] = a[1] = a[3] = 0                                   # raw: no in/out/local processing
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        if baud is not None:
            a[4] = a[5] = getattr(termios, f"B{baud}")
        a[6][termios.VMIN]  = 0
        a[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        os.set_blocking(self.fd, False)

    def close(self):
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def drain(self, seconds, quiet_after, until=None, echo=True):
        """Read for up to `seconds`: until `until` (bytes) arrives, or, without it, until the
        SoC has been silent for `quiet_after` seconds."""
        out = b""
        t0  = last = time.time()
        while time.time() - t0 < seconds:
            try:
                d = os.read(self.fd, 4096)
                if d:
                    out += d
                    last = time.time()
                    if echo:
                        sys.stdout.write(d.decode(errors="replace"))
                        sys.stdout.flush()
            except BlockingIOError:
                pass
            if until is not None and until in out:
                break
            if until is None and out and time.time() - last > quiet_after:
                break
            time.sleep(0.02)
        return out

    def run(self, cmds, wait=20.0, quiet_after=2.0, until=None, echo=True):
        """Send each command and collect its reply; no command = a bare newline. Returns the
        whole transcript as text."""
        out   = self.drain(0.5, 0.3, echo=echo)                  # flush anything pending
        until = until.encode() if until is not None else None
        for c in (cmds or [""]):
            os.write(self.fd, (c + "\n").encode())
            out += self.drain(wait, quiet_after, until, echo)
        return out.decode(errors="replace")


def run(cmds, tty="/dev/ttyLXU0", wait=20.0, quiet_after=2.0, until=None, echo=False, baud=None):
    """One console session: open, send `cmds`, close. Returns the transcript (\\r removed)."""
    with Console(tty, baud) as console:
        return console.run(cmds, wait, quiet_after, until, echo).replace("\r", "")

# Run ----------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Scripted console over the PCIe liteuart tty.")
    parser.add_argument("-t", "--tty",     default="/dev/ttyLXU0",           help="LiteUART tty.")
    parser.add_argument("-c", "--cmd",     action="append", default=[],      help="Command to send (repeatable).")
    parser.add_argument("-w", "--wait",    type=float,      default=20.0,    help="Seconds to wait for each command's output.")
    parser.add_argument("--quiet-after",   type=float,      default=2.0,     help="Stop early once the SoC has been silent this long (s).")
    parser.add_argument("--until",                                           help="Wait until this literal marker is received.")
    parser.add_argument("--baud",          type=int,        default=None,    help="A serial UART's rate (the Ti375 kit: 115200).")
    args = parser.parse_args(argv)

    try:
        with Console(args.tty, args.baud) as console:
            console.run(args.cmd, args.wait, args.quiet_after, args.until)
    except ConsoleBusy as e:
        sys.exit(str(e))
    print()


if __name__ == "__main__":
    main()
