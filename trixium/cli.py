#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""`trixium`: the one entry point for the model path.

    trixium convert   models/1.7b/Ternary-Bonsai-1.7B-PQ2_0.gguf  # -> models/bons/bonsai-1p7b-n16.img
    trixium convert   models/8b/Ternary-Bonsai-8B-PQ2_0.gguf      # -> models/bons/bonsai-8b-n16.img
    trixium verify    models/bons/bonsai-1p7b-n16.img
    trixium tokenizer models/1.7b/tokenizer.json                 # -> models/1.7b/tokenizer.bin
    trixium deploy    xcu1525 [--board xcu1525|fk33]              # bitstream + boot
    trixium chat      [--tty /dev/ttyLXU0] [-n 256]              # on the machine with the card
    trixium console   "stats" "bench 64"                         # raw firmware commands

`deploy` drives the lab board from the workstation (tools/<board>_bringup.sh with
tools/board.env); `chat` and `console` run where the card's /dev/ttyLXU0 is.
"""

import os
import sys
import shlex
import argparse
import subprocess

# Paths --------------------------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Helpers ------------------------------------------------------------------------------------------

def _run(cmd, cwd=REPO):
    return subprocess.call(cmd, cwd=cwd)


def _board_env():
    """Read tools/board.env (KEY=value lines) without sourcing it."""
    env = {}
    with open(os.path.join(REPO, "tools", "board.env"), encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" in line and not line.startswith(("H=", "REPO=")):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"')
    return env

# Commands -----------------------------------------------------------------------------------------

def cmd_convert(args):
    from trixium.model.convert import convert
    out = args.out
    if out is None:
        stem = "bonsai-8b" if "8B" in os.path.basename(args.gguf) else "bonsai-1p7b"
        out  = os.path.join(REPO, "models", "bons", f"{stem}-n{args.channels}.img")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tokenizer = args.tokenizer
    if tokenizer is None:
        candidate = os.path.join(os.path.dirname(args.gguf), "tokenizer.json")
        tokenizer = candidate if os.path.exists(candidate) else None
    manifest = out.rsplit(".", 1)[0] + ".json"
    convert(args.gguf, args.channels, out, manifest, tokenizer, args.layers, args.check,
            emb16=not args.no_emb16)
    print(f"{out}\n{manifest}")


def cmd_verify(args):
    from trixium.model.convert import verify
    sys.exit(0 if verify(args.image) else 1)


def cmd_tokenizer(args):
    out = args.out or os.path.join(os.path.dirname(args.tokenizer_json), "tokenizer.bin")
    sys.exit(_run([sys.executable, os.path.join(REPO, "tools", "export_tokenizer.py"),
                   args.tokenizer_json, "-o", out]))


def cmd_deploy(args):
    sys.exit(_run([os.path.join(REPO, "tools", f"{args.board}_bringup.sh"), args.build]))


def cmd_chat(args):
    flags = ["-n", str(args.n)]
    if args.temp or args.top_p != 1.0 or args.repeat != 1.0:
        flags += ["-S", f"{args.temp},{args.top_p},{args.repeat}"]
    if args.seed:
        flags += ["-s", str(args.seed)]
    if args.system:
        flags += ["-y", args.system]
    if args.firmware:
        flags += ["-k", args.firmware]
    if args.image:
        flags += ["-m", args.image]
    if args.remote:
        # From the workstation: the chat runs on the board host over ssh (tools/board.env), in
        # the tree the deploy script maintains there.
        env    = _board_env()
        remote = (f"cd {shlex.quote(env['BOARD_DIR'])} && ./user/trixium_chat -t {shlex.quote(args.tty)} "
                  + " ".join(shlex.quote(f) for f in flags))
        os.execvp("ssh", ["ssh", "-t", f"{env['BOARD_USER']}@{env['BOARD_HOST']}", remote])
    exe = os.path.join(REPO, "trixium", "software", "user", "trixium_chat")
    if not os.path.exists(exe):
        sys.exit(f"{exe} not built: make -C trixium/software/user BUILD_DIR=../../../build/<build>")
    os.execv(exe, [exe, "-t", args.tty, "-z", args.tokenizer] + flags)


def cmd_console(args):
    cmd = [sys.executable, os.path.join(REPO, "tools", "pcie_console.py"), "-t", args.tty,
           "-w", str(args.wait), "--quiet-after", str(args.quiet)]
    for c in args.commands:
        cmd += ["-c", c]
    sys.exit(_run(cmd))

# Main ---------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="trixium", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Convert.
    p = subparsers.add_parser("convert", help="GGUF -> BONS image (streams; ~13 s, ~2 GB for the 1.7B).")
    p.add_argument("gguf",                                    help="Source GGUF.")
    p.add_argument("--channels",  default=16, type=int,       help="Engine channels of the bitstream (default: 16).")
    p.add_argument("--out",                                   help="Image (default: models/bons/bonsai-<model>-n<channels>.img).")
    p.add_argument("--tokenizer",                             help="tokenizer.json hashed into the image (default: next to the GGUF).")
    p.add_argument("--layers",    type=int,                   help="Truncate to the first K layers.")
    p.add_argument("--check",     action="store_true",        help="Re-read and CRC every tensor.")
    p.add_argument("--no-emb16",  action="store_true",        help="Leave out the int16 embedding table (-622 MB, ~2.6%% slower per token).")
    p.set_defaults(fn=cmd_convert)

    # Verify.
    p = subparsers.add_parser("verify", help="Verify every tensor of a v2/v3 image.")
    p.add_argument("image", help="BONS image.")
    p.set_defaults(fn=cmd_verify)

    # Tokenizer.
    p = subparsers.add_parser("tokenizer", help="tokenizer.json -> tokenizer.bin for the C chat tool.")
    p.add_argument("tokenizer_json", help="Source tokenizer.json.")
    p.add_argument("-o", "--out",    help="Output (default: tokenizer.bin next to the JSON).")
    p.set_defaults(fn=cmd_tokenizer)

    # Deploy.
    p = subparsers.add_parser("deploy", help="Configure the lab board with a build and boot it (tools/board.env).")
    p.add_argument("build",   default="xcu1525", nargs="?",                  help="Build name in build/ (default: xcu1525).")
    p.add_argument("--board", default="xcu1525", choices=["xcu1525", "fk33"], help="Board.")
    p.set_defaults(fn=cmd_deploy)

    # Chat.
    tokenizer = os.path.join(REPO, "models", "1.7b", "tokenizer.bin")
    p = subparsers.add_parser("chat", help="Chat REPL on the machine with the card.")
    p.add_argument("--tty",            default="/dev/ttyLXU0",  help="Console tty.")
    p.add_argument("--tokenizer",      default=tokenizer,       help="tokenizer.bin.")
    p.add_argument("-n",               default=256, type=int,   help="Reply cap in tokens (replies end at <|im_end|>).")
    p.add_argument("--temp",           default=0.0, type=float, help="Sampling temperature (0 = greedy).")
    p.add_argument("--top-p",          default=1.0, type=float, help="Sampling top-p.")
    p.add_argument("--repeat",         default=1.0, type=float, help="Repetition penalty (1.1 breaks greedy loops).")
    p.add_argument("--seed",           default=0,   type=int,   help="Sampling PRNG seed.")
    p.add_argument("--system",                                  help="System prompt (a system turn ahead of the conversation).")
    p.add_argument("--remote",         action="store_true",     help="Run the chat on the board host over ssh (tools/board.env).")
    p.add_argument("-k", "--firmware",                          help="Boot this firmware first (trixium.bin).")
    p.add_argument("-m", "--image",                             help="Upload this weight image first.")
    p.set_defaults(fn=cmd_chat)

    # Console.
    p = subparsers.add_parser("console", help="Send firmware console commands and print the replies.")
    p.add_argument("commands", nargs="+",                help="Firmware console commands.")
    p.add_argument("--tty",    default="/dev/ttyLXU0",   help="Console tty.")
    p.add_argument("--wait",   default=60.0, type=float, help="Seconds to wait for each command's output.")
    p.add_argument("--quiet",  default=3.0,  type=float, help="Stop early once the console has been silent this long (s).")
    p.set_defaults(fn=cmd_console)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
