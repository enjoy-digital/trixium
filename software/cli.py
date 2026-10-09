#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""`trixium`: the one entry point for the model path.

    trixium models    [--tiny]                                   # PrismML models (or the tiny ones)
    trixium convert   models/1.7b/Ternary-Bonsai-1.7B-PQ2_0.gguf  # -> models/bons/bonsai-1p7b-n16.img
    trixium convert   models/8b/Ternary-Bonsai-8B-PQ2_0.gguf      # -> models/bons/bonsai-8b-n16.img
    trixium verify    models/bons/bonsai-1p7b-n16.img
    trixium tokenizer models/1.7b/tokenizer.json                 # -> models/1.7b/tokenizer.bin
    trixium deploy    xcu1525 --board xcu1525 --model 27b       # bitstream + boot + tokens + bench
    trixium validate  [--steps sim,tests]                        # release validation (hours)
    trixium driver    load|unload                                # trixium.ko + liteuart.ko
    trixium chat      [--tty /dev/ttyLXU0] [-n 256]              # on the machine with the card
    trixium chat      --remote --board fk33                      # on a lab board host, over ssh
    trixium console   "stats" "bench 64"                         # raw firmware commands

`deploy` drives a lab board from the workstation (software/deploy.py, board hosts in
~/.config/trixium/boards.ini); `chat` and `console` run where the card's /dev/ttyLXU0 is.
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


# Commands -----------------------------------------------------------------------------------------

def cmd_convert(args):
    from model.convert import convert
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
    from model.convert import verify
    sys.exit(0 if verify(args.image) else 1)


def cmd_tokenizer(args):
    from model.tokenizer import export_tokenizer
    out = args.out or os.path.join(os.path.dirname(args.tokenizer_json), "tokenizer.bin")
    export_tokenizer(args.tokenizer_json, out)


# PrismML Ternary-Bonsai models: GGUF PQ2_0 (ggml type 142: 2-bit codes + one fp16 scale per 128
# weights, the block earlier releases shipped as "Q2_0"; the *_Q2_0_g64 files do not convert) +
# the tokenizer/config of the matching "unpacked" repo.
MODELS = {
    "1.7b": ("Ternary-Bonsai-1.7B", ["Ternary-Bonsai-1.7B-PQ2_0.gguf"]),
    "8b":   ("Ternary-Bonsai-8B",   ["Ternary-Bonsai-8B-PQ2_0.gguf"]),
}
MODELS_FREE_GB = 8 # Disk guard.

# The tiny random models the tests and the simulation use (seconds): (channels, config).
TINY_MODELS = [(2, "tiny"), (4, "tiny"), (8, "tiny"), (16, "tiny"),
               (2, "hybrid"), (8, "hybrid"), (8, "hybrid-1k")]


def cmd_models(args):
    out = args.out or os.path.join(REPO, "models", "tiny" if args.tiny else "")
    if args.tiny:
        from model import tinygen
        configs = {"tiny": tinygen.BONSAI_TINY, "hybrid": tinygen.BONSAI_TINY_HYBRID,
                   "hybrid-1k": tinygen.BONSAI_TINY_HYBRID_1K}
        for channels, name in TINY_MODELS:
            img, _, _ = tinygen.generate(out, channels, cfg=configs[name])
            print(img)
        return
    import shutil
    from huggingface_hub import hf_hub_download
    os.makedirs(out, exist_ok=True)
    free = shutil.disk_usage(out).free/1e9
    if free < MODELS_FREE_GB:
        sys.exit(f"only {free:.0f} GB free in {out}, need >= {MODELS_FREE_GB} GB.")
    for name, (repo, ggufs) in MODELS.items():
        local_dir = os.path.join(out, name)
        for filename in ggufs:
            print(hf_hub_download(f"prism-ml/{repo}-gguf", filename, local_dir=local_dir))
        for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
            print(hf_hub_download(f"prism-ml/{repo}-unpacked", filename, local_dir=local_dir))


def cmd_deploy(args):
    from software.deploy import run
    sys.exit(run(args))


def cmd_validate(args):
    from software.validate import run
    sys.exit(run(args))


def cmd_driver(args):
    from software.driver import main as driver_main
    driver_main([args.action, "--dir", args.dir])


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
        # From the workstation: the chat runs on the board host over ssh (boards.ini), in the
        # tree the deploy maintains there.
        from software.deploy import board_config
        conf   = board_config(args.board)
        remote = (f"cd {shlex.quote(conf['dir'])} && ./user/trixium_chat -t {shlex.quote(args.tty)} "
                  + " ".join(shlex.quote(f) for f in flags))
        os.execvp("ssh", ["ssh", "-t", conf["target"], remote])
    exe = os.path.join(REPO, "software", "user", "trixium_chat")
    if not os.path.exists(exe):
        sys.exit(f"{exe} not built: make -C software/user BUILD_DIR=../../build/<build>")
    os.execv(exe, [exe, "-t", args.tty, "-z", args.tokenizer] + flags)


def cmd_console(args):
    from software.console import main as console_main
    argv = ["-t", args.tty, "-w", str(args.wait), "--quiet-after", str(args.quiet)]
    for c in args.commands:
        argv += ["-c", c]
    console_main(argv)


def cmd_smi(args):
    exe = os.path.join(REPO, "software", "user", "trixium_smi")
    if not os.path.exists(exe):
        sys.exit(f"{exe} not built: make -C software/user BUILD_DIR=../../build/<build>")
    os.execv(exe, [exe] + args)

# Main ---------------------------------------------------------------------------------------------

def main(argv=None):
    # smi passes everything through to trixium_smi (its own options and commands).
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["smi"]:
        cmd_smi(argv[1:])

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

    # Models.
    p = subparsers.add_parser("models", help="Download the PrismML models (~2.6 GB) or generate the tiny test models.")
    p.add_argument("--tiny", action="store_true", help="Generate the tiny random models (tests, simulation) instead.")
    p.add_argument("--out",                       help="Output directory (default: models/, models/tiny/ with --tiny).")
    p.set_defaults(fn=cmd_models)

    # Deploy.
    from software import deploy, validate
    p = subparsers.add_parser("deploy", help="Deploy a build on a lab board: timing gate, configure, boot, tokens, bench.")
    deploy.add_arguments(p)
    p.set_defaults(fn=cmd_deploy)

    # Validate.
    p = subparsers.add_parser("validate", help="Release validation: bitstreams, tests, simulation, boards (hours).")
    validate.add_arguments(p)
    p.set_defaults(fn=cmd_validate)

    # Driver.
    p = subparsers.add_parser("driver", help="Load/unload trixium.ko + liteuart.ko (through sudo).")
    p.add_argument("action", choices=["load", "unload"],                                    help="Load or unload.")
    p.add_argument("--dir",  default=os.path.join(REPO, "build", "xcu1525", "driver", "kernel"), help="Directory of the built modules.")
    p.set_defaults(fn=cmd_driver)

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
    p.add_argument("--remote",         action="store_true",     help="Run the chat on a lab board host over ssh (boards.ini).")
    p.add_argument("--board",          default="xcu1525",       help="Lab board of --remote (a boards.ini section).")
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

    # SMI.
    subparsers.add_parser("smi", help="Card status and diagnostics (runs trixium_smi; -h for its options).")

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
