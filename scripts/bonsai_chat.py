#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""bonsai_chat: host-side chat REPL for the TriXium firmware.

Tokenizes prompts on the host (Qwen3 tokenizer), drives the firmware's `gen` command over the
crossover UART (/dev/ttyLXU0 via the liteuart driver, or any pty/tty the firmware console is
on), decodes returned token ids.

Usage: .venv/bin/python scripts/bonsai_chat.py [--port /dev/ttyLXU0] \
           [--tokenizer models/1.7b/tokenizer.json] [-n 64]
"""

import argparse
import sys
import time

import serial
from tokenizers import Tokenizer

def send_cmd(ser, cmd, timeout=600):
    ser.write((cmd + "\n").encode())
    ser.flush()
    deadline = time.time() + timeout
    out = []
    line = b""
    while time.time() < deadline:
        b1 = ser.read(1)
        if not b1:
            continue
        if b1 == b"\n":
            s = line.decode(errors="replace").rstrip()
            out.append(s)
            if s.startswith("GEN DONE") or s.startswith("hdrload:") or "unknown" in s:
                return out
            line = b""
        else:
            line += b1
    raise TimeoutError(f"no completion for: {cmd}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      default="/dev/ttyLXU0")
    parser.add_argument("--tokenizer", default="models/1.7b/tokenizer.json")
    parser.add_argument("-n",          default=64, type=int)
    parser.add_argument("--raw",       action="store_true", help="No chat template.")
    args = parser.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    ser = serial.Serial(args.port, 115200, timeout=0.1)

    print(f"connected to {args.port}; type a prompt (Ctrl-C to quit)")
    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip():
            continue
        text = prompt if args.raw else \
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok.encode(text).ids
        if len(ids) > 60:
            print("(prompt truncated to 60 tokens)")
            ids = ids[:60]
        cmd = f"gen {args.n} " + " ".join(str(i) for i in ids)
        t0  = time.time()
        out = send_cmd(ser, cmd)
        dt  = time.time() - t0
        gen = []
        for line in out:
            if line.startswith("GEN:"):
                gen = [int(t) for t in line[4:].split()]
        if gen:
            print(f"bonsai> {tok.decode(gen)}")
            print(f"        ({len(gen)} tokens in {dt:.1f}s, {len(gen)/dt:.2f} tok/s)")
        else:
            print("\n".join(out))

if __name__ == "__main__":
    main()
