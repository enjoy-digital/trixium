#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""`trixium`: the one entry point for the model path.

    trixium models    [--tiny]                                   # PrismML models (or the tiny ones)
    trixium convert   models/1.7b/Ternary-Bonsai-1.7B-PQ2_0.gguf  # -> models/bons/bonsai-1p7b-n16.img
    trixium convert   models/8b/Ternary-Bonsai-8B-PQ2_0.gguf      # -> models/bons/bonsai-8b-n16.img
    trixium convert   --board xcu1525 models/27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf  # -> ...-n16-xcu1525.img
    trixium convert   --board ypcb models/1.7b/Ternary-Bonsai-1.7B-PQ2_0.gguf   # -> ...-n8-ypcb.img (5-trit codes)
    trixium verify    models/bons/bonsai-1p7b-n16.img
    trixium tokenizer models/1.7b/tokenizer.json                 # -> models/1.7b/tokenizer.bin
    trixium deploy    xcu1525 --board xcu1525 --model 27b       # bitstream + boot + tokens + bench
    trixium validate  [--steps sim,tests]                        # release validation (hours)
    trixium driver    load|unload                                # trixium.ko + liteuart.ko
    trixium chat      [--tty /dev/ttyLXU0] [-n 256]              # on the machine with the card
    trixium chat      --remote --board fk33                      # on a lab board host, over ssh
    trixium console   "stats" "bench 64"                         # raw firmware commands
    trixium demo      --board ypcb --build ypcb                 # the README's chat + smi GIF

`deploy` drives a lab board from the workstation (software/lab/deploy.py, board hosts in
~/.config/trixium/boards.ini); `chat` and `console` run where the card's /dev/ttyLXU0 is.
"""

import os
import re
import sys
import shlex
import argparse

# Paths --------------------------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Commands -----------------------------------------------------------------------------------------

# Each board's weight windows (convert's chan_stride/window/window_reserve) and image name
# suffix. The FK33's are convert()'s defaults (256 MB HBM stride); the XCU1525 has 704 MB windows
# at a 1 GB stride, the last one 96 KB short (the bank-group skew); the YPCB-00338 splits each
# 2 GB DDR3 controller above the weight base evenly between its windows (trixium_ypcb.py: 864 MB
# at 4 channels, 432 MB at 8), the last one short of its bank skew (64 KB / windows each).
BOARD_SUFFIX = {"fk33": "", "xcu1525": "-xcu1525", "ypcb": "-ypcb", "ti375": "-ti375"}

# Engine channels of each board's bitstream (convert's default --channels), and the boards whose
# bitstream unpacks 5-trit codes (convert's default --trit5).
# The boards' memory systems: (controllers, bytes per controller, default engine channels, trit5
# weights). A board with one controller (the small boards: one DRAM) puts every channel on it.
BOARDS = {
    "fk33":    dict(memctrl=0, ctrl_size=0,       channels=16, trit5=False),   # HBM: no windows.
    "xcu1525": dict(memctrl=4, ctrl_size=4 << 30, channels=16, trit5=False),
    "ypcb":    dict(memctrl=2, ctrl_size=2 << 30, channels=8,  trit5=True),
    # The Efinix Ti375 C529 kit: one 1 GB LPDDR4x controller, four channels on it (trixium_ti375.py).
    "ti375":   dict(memctrl=1, ctrl_size=1 << 30, channels=4,  trit5=True, no_emb16=True),
}
BOARD_CHANNELS = {name: b["channels"] for name, b in BOARDS.items()}
BOARD_TRIT5    = {name: b["trit5"]    for name, b in BOARDS.items()}


def board_windows(board, channels, memctrl=None):
    """(chan_stride, window, window_reserve) of `board`'s images at `channels` engine channels
    over `memctrl` controllers (the board's by default; the YPCB also builds with one)."""
    if board == "xcu1525":
        return 0x4000_0000, None, 0x18000
    if board == "ypcb":
        from model.layout import ddr_chan_stride
        memctrl  = memctrl or BOARDS[board]["memctrl"]
        if channels % memctrl:
            raise SystemExit(f"{channels} channels do not shard over {memctrl} controllers")
        chan_mod = channels//memctrl
        stride   = ddr_chan_stride(BOARDS[board]["ctrl_size"], chan_mod)
        return stride, stride, (chan_mod - 1)*(0x10000//chan_mod)
    if board == "ti375":
        # The target's windows: channel c at HBM_WEIGHT_BASE + c*stride, no bank skew.
        from model.layout import ddr_chan_stride
        stride = ddr_chan_stride(BOARDS[board]["ctrl_size"], channels)
        return stride, stride, 0
    return None, None, 0


def image_stem(gguf_name):
    """The image name of a GGUF: bonsai-1p7b / -8b / -27b, bonsai2-2b / -4b / -9b (Bonsai 2 small)."""
    if "27B" in gguf_name:
        return "bonsai-27b"
    m = re.search(r"Bonsai-2-(\d+)B", gguf_name)
    if m:
        return f"bonsai2-{m.group(1)}b"
    return "bonsai-8b" if "8B" in gguf_name else "bonsai-1p7b"


def cmd_convert(args):
    from model.convert import convert
    if args.channels is None:
        args.channels = BOARD_CHANNELS[args.board]
    if args.trit5 is None:
        args.trit5 = BOARD_TRIT5[args.board]
    chan_stride, window, window_reserve = board_windows(args.board, args.channels, getattr(args, "memctrl", None))
    suffix = BOARD_SUFFIX[args.board]
    memctrl = getattr(args, "memctrl", None)
    if memctrl and args.board in BOARDS and memctrl != BOARDS[args.board]["memctrl"]:
        suffix = f"-m{memctrl}{suffix}"      # Windows over fewer controllers than the board's: its own image.
    out = args.out
    if out is None:
        name = os.path.basename(args.gguf)
        stem = image_stem(name)
        out  = os.path.join(REPO, "models", "bons", f"{stem}-n{args.channels}{suffix}.img")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tokenizer = args.tokenizer
    if tokenizer is None:
        candidate = os.path.join(os.path.dirname(args.gguf), "tokenizer.json")
        tokenizer = candidate if os.path.exists(candidate) else None
    manifest = out.rsplit(".", 1)[0] + ".json"
    convert(args.gguf, args.channels, out, manifest, tokenizer, args.layers, args.check,
            emb16=not (args.no_emb16 or BOARDS.get(args.board, {}).get("no_emb16")),
            chan_stride=chan_stride, window=window,
            window_reserve=window_reserve, trit5=args.trit5)
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
# Ternary-Bonsai-2 2B/4B/9B (Qwen3.5 hybrids, like the 27B): the dense-trit PTQ1_0 files (the same
# weights as PQ2_0, smaller downloads); they have no "unpacked" repo, so the tokenizer comes from
# the Qwen3.5 base model (the 27B's tokenizer too).
MODELS = {
    # name   GGUF repo                 files                                  tokenizer repo
    "1.7b": ("Ternary-Bonsai-1.7B",   ["Ternary-Bonsai-1.7B-PQ2_0.gguf"],    None),
    "8b":   ("Ternary-Bonsai-8B",     ["Ternary-Bonsai-8B-PQ2_0.gguf"],      None),
    "2b":   ("Ternary-Bonsai-2-2B",   ["Ternary-Bonsai-2-2B-PTQ1_0.gguf"],   "Qwen/Qwen3.5-2B"),
    "4b":   ("Ternary-Bonsai-2-4B",   ["Ternary-Bonsai-2-4B-PTQ1_0.gguf"],   "Qwen/Qwen3.5-4B"),
    "9b":   ("Ternary-Bonsai-2-9B",   ["Ternary-Bonsai-2-9B-PTQ1_0.gguf"],   "Qwen/Qwen3.5-9B"),
}
DEFAULT_MODELS = ["1.7b", "8b"]
MODELS_FREE_GB = 8 # Disk guard.

# The tiny random models the tests and the simulation use (seconds): (channels, config); a
# "-t5" config is the same model with 5-trits-per-byte weights (the YPCB's images), at the
# channel counts the YPCB simulations run: 4 (the board), 2 and 1 (the small-board profiles).
TINY_MODELS = [(2, "tiny"), (4, "tiny"), (8, "tiny"), (16, "tiny"),
               (2, "hybrid"), (8, "hybrid"), (8, "hybrid-1k"), (4, "bonsai2"), (4, "bonsai2-512"),
               (4, "hybrid-1k-t5"), (2, "hybrid-1k-t5"), (1, "hybrid-1k-t5")]


def cmd_models(args):
    out = args.out or os.path.join(REPO, "models", "tiny" if args.tiny else "")
    if args.tiny:
        from model import tinygen
        configs = {"tiny": tinygen.BONSAI_TINY, "hybrid": tinygen.BONSAI_TINY_HYBRID,
                   "hybrid-1k": tinygen.BONSAI_TINY_HYBRID_1K, "bonsai2": tinygen.BONSAI2_TINY,
                   "bonsai2-512": tinygen.BONSAI2_TINY_512}
        for channels, name in TINY_MODELS:
            trit5 = name.endswith("-t5")
            img, _, _ = tinygen.generate(out, channels, cfg=configs[name[:-3] if trit5 else name],
                                         trit5=trit5)
            print(img)
        return
    import shutil
    from huggingface_hub import hf_hub_download
    os.makedirs(out, exist_ok=True)
    free = shutil.disk_usage(out).free/1e9
    if free < MODELS_FREE_GB:
        sys.exit(f"only {free:.0f} GB free in {out}, need >= {MODELS_FREE_GB} GB.")
    for name in args.names or DEFAULT_MODELS:
        if name not in MODELS:
            sys.exit(f"unknown model {name} (known: {', '.join(MODELS)})")
        repo, ggufs, tokenizer_repo = MODELS[name]
        local_dir = os.path.join(out, name)
        for filename in ggufs:
            print(hf_hub_download(f"prism-ml/{repo}-gguf", filename, local_dir=local_dir))
        for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
            print(hf_hub_download(tokenizer_repo or f"prism-ml/{repo}-unpacked", filename, local_dir=local_dir))


def cmd_deploy(args):
    from software.lab.deploy import run
    sys.exit(run(args))


def cmd_validate(args):
    from software.lab.validate import run
    sys.exit(run(args))


def cmd_driver(args):
    from software.host.driver import main as driver_main
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
        from software.lab.deploy import board_config
        # The deploy keeps the tokenizers at the same models/<model>/ paths there.
        conf      = board_config(args.board)
        tokenizer = os.path.relpath(os.path.abspath(args.tokenizer), REPO)
        if tokenizer.startswith(".."):
            sys.exit(f"--remote: {args.tokenizer} is not under the tree (the deploy copies models/<model>/tokenizer.bin).")
        remote = (f"cd {shlex.quote(conf['dir'])} && ./user/trixium_chat -t {shlex.quote(args.tty)} "
                  f"-z {shlex.quote(tokenizer)} " + " ".join(shlex.quote(f) for f in flags))
        os.execvp("ssh", ["ssh", "-t", conf["target"], remote])
    exe = os.path.join(REPO, "software", "user", "trixium_chat")
    if not os.path.exists(exe):
        sys.exit(f"{exe} not built: make -C software/user BUILD_DIR=../../build/<build>")
    os.execv(exe, [exe, "-t", args.tty, "-z", args.tokenizer] + flags)


def cmd_console(args):
    from software.host.console import main as console_main
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
    if argv[:1] == ["demo"]:
        # demo passes everything through to software/lab/demo.py (its own options).
        from software.lab.demo import main as demo_main
        return demo_main(argv[1:])

    parser = argparse.ArgumentParser(prog="trixium", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Convert.
    p = subparsers.add_parser("convert", help="GGUF -> BONS image (streams; ~13 s, ~2 GB for the 1.7B).")
    p.add_argument("gguf",                              help="Source GGUF.")
    p.add_argument("--channels",  default=None, type=int, help="Engine channels of the bitstream (default: the board's, 16; ypcb: 8).")
    p.add_argument("--memctrl", type=int, default=None, help="Controllers the image's windows shard over (default: the board's; the YPCB also builds with 1).")
    p.add_argument("--board",     default="fk33",       choices=sorted(BOARD_SUFFIX), help="Target board's weight windows (default: fk33).")
    p.add_argument("--out",                             help="Image (default: models/bons/bonsai-<model>-n<channels>[-<board>].img).")
    p.add_argument("--tokenizer",                       help="tokenizer.json hashed into the image (default: next to the GGUF).")
    p.add_argument("--layers",    type=int,             help="Truncate to the first K layers.")
    p.add_argument("--check",     action="store_true",  help="Read two tensors back and compare.")
    p.add_argument("--no-emb16",  action="store_true",  help="Leave out the int16 embedding table (-622 MB, ~2.6%% slower per token).")
    p.add_argument("--trit5",     action=argparse.BooleanOptionalAction, default=None, help="Ternary weights 5 trits per byte (bitstreams built with --trit5; default: the board's, on for ypcb).")
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
    p = subparsers.add_parser("models", help="Download PrismML models (default: the 1.7B + 8B, ~2.6 GB) or generate the tiny test models.")
    p.add_argument("names", nargs="*",           help=f"Models to download (default: {' '.join(DEFAULT_MODELS)}; known: {' '.join(MODELS)}).")
    p.add_argument("--tiny", action="store_true", help="Generate the tiny random models (tests, simulation) instead.")
    p.add_argument("--out",                       help="Output directory (default: models/, models/tiny/ with --tiny).")
    p.set_defaults(fn=cmd_models)

    # Deploy.
    from software.lab import deploy, validate
    p = subparsers.add_parser("deploy", help="Deploy a build on a lab board: timing gate, configure, boot, tokens, bench.")
    deploy.add_arguments(p)
    p.set_defaults(fn=cmd_deploy)

    # Validate.
    p = subparsers.add_parser("validate", help="Release validation: bitstreams, tests, simulation, boards (hours).")
    validate.add_arguments(p)
    p.set_defaults(fn=cmd_validate)

    # Driver.
    p = subparsers.add_parser("driver", help="Load/unload trixium.ko + liteuart.ko (through sudo).")
    p.add_argument("action", choices=["load", "unload"],                                         help="Load or unload.")
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
    subparsers.add_parser("demo", help="Record the README's chat + smi demo GIF on a lab board (-h for its options).")

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
