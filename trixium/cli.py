#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""`bonsai`: the one entry point for the model path.

    bonsai convert   models/1.7b/Ternary-Bonsai-1.7B-PQ2_0.gguf  # -> models/bons/bonsai-1p7b-n16.img
    bonsai convert   models/8b/Ternary-Bonsai-8B-PQ2_0.gguf      # -> models/bons/bonsai-8b-n16.img
    bonsai verify    models/bons/bonsai-1p7b-n16.img
    bonsai tokenizer models/1.7b/tokenizer.json                 # -> models/1.7b/tokenizer.bin
    bonsai deploy    n16_212_v15b                               # bitstream + weights + boot
    bonsai chat      [--tty /dev/ttyLXU0] [-n 256]              # on the machine with the card
    bonsai console   "stats" "bench 64"                         # raw firmware commands

`deploy` drives the lab board from the workstation (tools/fk33_bringup_measure.sh with
tools/board.env); `chat` and `console` run where the card's /dev/ttyLXU0 is.
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, cwd=REPO):
    return subprocess.call(cmd, cwd=cwd)


def cmd_convert(a):
    from trixium.model.convert import convert
    out = a.out
    if out is None:
        stem = "bonsai-8b" if "8B" in os.path.basename(a.gguf) else "bonsai-1p7b"
        out = os.path.join(REPO, "models", "bons", f"{stem}-n{a.channels}.img")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tokenizer = a.tokenizer
    if tokenizer is None:
        cand = os.path.join(os.path.dirname(a.gguf), "tokenizer.json")
        tokenizer = cand if os.path.exists(cand) else None
    manifest = out.rsplit(".", 1)[0] + ".json"
    convert(a.gguf, a.channels, out, manifest, tokenizer, a.layers, a.check,
            emb16=not a.no_emb16)
    print(f"{out}\n{manifest}")


def cmd_verify(a):
    import struct
    with open(a.image, 'rb') as f:
        prefix = f.read(8)
    if prefix == struct.pack('<4sI', b'BONS', 4):
        from trixium.model.bons4 import Reader
        image = Reader(a.image)
        image.verify()
        print(f'BONS v4: {len(image.entries)} tensors, all payload SHA256 checks pass')
        return
    from trixium.model.convert import verify
    sys.exit(0 if verify(a.image) else 1)


def cmd_pack(a):
    import json
    from trixium.model.pack_native import pack
    result = pack(a.gguf, a.out, tokenizer=a.tokenizer, check=a.check,
                  progress=lambda name: print(f'  {name}', flush=True))
    print(json.dumps(result, indent=2))


def cmd_place(a):
    import json
    from pathlib import Path
    from trixium.capabilities import Capabilities
    from trixium.model.bons4 import Reader
    from trixium.placement import plan_weights
    image = Reader(a.image)
    image.verify()
    caps = Capabilities.from_bytes(Path(a.capabilities).read_bytes())
    plan = plan_weights(image, caps, topology_only=a.topology_only)
    with open(a.out, 'x') as f:
        json.dump(plan.to_dict(), f, indent=2)
        f.write('\n')
    print(f'{a.out}: weights-only placement, {len(plan.spans)} spans; no device I/O')


def cmd_tokenizer(a):
    out = a.out or os.path.join(os.path.dirname(a.tokenizer_json), "tokenizer.bin")
    sys.exit(_run([sys.executable, os.path.join(REPO, "tools", "export_tokenizer.py"),
                   a.tokenizer_json, "-o", out]))


def cmd_deploy(a):
    sys.exit(_run([os.path.join(REPO, "tools", "fk33_bringup_measure.sh"), a.build]))


def cmd_chat(a):
    flags = ["-n", str(a.n)]
    if a.temp or a.top_p != 1.0 or a.repeat != 1.0:
        flags += ["-S", f"{a.temp},{a.top_p},{a.repeat}"]
    if a.seed:
        flags += ["-s", str(a.seed)]
    if a.system:
        flags += ["-y", a.system]
    if a.firmware:
        flags += ["-k", a.firmware]
    if a.image:
        flags += ["-m", a.image]
    if a.remote:
        # From the workstation: the chat runs on the board host over ssh (tools/board.env),
        # in the tree the deploy script maintains there.
        env = _board_env()
        import shlex
        remote = (f"cd {shlex.quote(env['BOARD_DIR'])} && ./user/bonsai_chat -t {shlex.quote(a.tty)} "
                  + " ".join(shlex.quote(f) for f in flags))
        os.execvp("ssh", ["ssh", "-t", f"{env['BOARD_USER']}@{env['BOARD_HOST']}", remote])
    exe = os.path.join(REPO, "trixium", "software", "user", "bonsai_chat")
    if not os.path.exists(exe):
        sys.exit(f"{exe} not built: make -C trixium/software/user BUILD_DIR=../../../build/<build>")
    os.execv(exe, [exe, "-t", a.tty, "-z", a.tokenizer] + flags)


def _board_env():
    env = {}
    for line in open(os.path.join(REPO, "tools", "board.env")):
        line = line.split("#", 1)[0].strip()
        if "=" in line and not line.startswith(("H=", "REPO=")):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def cmd_console(a):
    cmd = [sys.executable, os.path.join(REPO, "tools", "pcie_console.py"), "-t", a.tty,
           "-w", str(a.wait), "--quiet-after", str(a.quiet)]
    for c in a.commands:
        cmd += ["-c", c]
    sys.exit(_run(cmd))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bonsai", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert", help="GGUF -> BONS image (streams; ~13 s, ~2 GB for the 1.7B)")
    p.add_argument("gguf")
    p.add_argument("--channels", type=int, default=16, help="engine channels of the bitstream (16)")
    p.add_argument("--out", help="default models/bons/bonsai-<model>-n<channels>.img")
    p.add_argument("--tokenizer", help="tokenizer.json (default: next to the GGUF); hashed into the image")
    p.add_argument("--layers", type=int, help="truncate to the first K layers")
    p.add_argument("--check", action="store_true", help="re-read and CRC every tensor")
    p.add_argument("--no-emb16", action="store_true",
                   help="leave out the int16 embedding table (-622 MB, ~2.6%% slower per token)")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("pack", help="lossless GGUF -> board-independent BONS v4 (not a legacy upload image)")
    p.add_argument("gguf")
    p.add_argument("--out", required=True, help="new output file; refuses overwrite")
    p.add_argument("--tokenizer", help="tokenizer file to hash (otherwise explicitly unpinned)")
    p.add_argument("--check", action="store_true", help="verify every written payload SHA256")
    p.set_defaults(fn=cmd_pack)

    p = sub.add_parser("place", help="offline BONS v4 whole-row placement; weights only, no upload")
    p.add_argument("image")
    p.add_argument("--capabilities", required=True, help="BNCP capability blob")
    p.add_argument("--out", required=True, help="new placement JSON; refuses overwrite")
    p.add_argument("--topology-only", action="store_true", help="capacity estimate even without region I/O hardware")
    p.set_defaults(fn=cmd_place)

    p = sub.add_parser("verify", help="verify every tensor of a v2/v3/v4 image")
    p.add_argument("image")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("tokenizer", help="tokenizer.json -> tokenizer.bin for the C chat tool")
    p.add_argument("tokenizer_json")
    p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_tokenizer)

    p = sub.add_parser("deploy", help="configure the lab board with a build and boot it (tools/board.env)")
    p.add_argument("build", nargs="?", default="n16_212_v15b", help="build/<name> (default n16_212_v15b)")
    p.set_defaults(fn=cmd_deploy)

    p = sub.add_parser("chat", help="chat REPL on the machine with the card")
    p.add_argument("--tty", default="/dev/ttyLXU0")
    p.add_argument("--tokenizer", default=os.path.join(REPO, "models", "1.7b", "tokenizer.bin"))
    p.add_argument("-n", type=int, default=256, help="reply cap in tokens (replies end at <|im_end|>)")
    p.add_argument("--temp", type=float, default=0.0, help="sampling temperature (0 = greedy)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--repeat", type=float, default=1.0, help="repetition penalty (1.1 breaks greedy loops)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--system", help="system prompt (a system turn ahead of the conversation)")
    p.add_argument("--remote", action="store_true",
                   help="run the chat on the board host over ssh (tools/board.env)")
    p.add_argument("-k", "--firmware", help="boot this firmware first (bonsai.bin)")
    p.add_argument("-m", "--image", help="upload this weight image first")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("console", help="send firmware console commands and print the replies")
    p.add_argument("commands", nargs="+")
    p.add_argument("--tty", default="/dev/ttyLXU0")
    p.add_argument("--wait", type=float, default=60.0)
    p.add_argument("--quiet", type=float, default=3.0)
    p.set_defaults(fn=cmd_console)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
