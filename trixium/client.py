#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Python client for a running TriXium card.

Talks to the firmware console over the PCIe UART (/dev/ttyLXU0 through the liteuart driver,
or any tty the console is on), tokenizes on the host with the HuggingFace tokenizer, and
uses the firmware's chat protocol (`tokclr` / `tok <ids>` / `genb <n>`) so a conversation
continues on-card without resending history.

    from trixium.client import Bonsai
    with Bonsai() as b:
        b.hdrload()
        print(b.generate("The capital of France is", max_tokens=32))
        for piece in b.stream("and Germany's?"):      # continues the conversation
            print(piece, end="", flush=True)
        print(b.command("stats"))

Requires pyserial and tokenizers (`pip install -e .` from the repository root).
"""

import re
import time

DEFAULT_TTY       = "/dev/ttyLXU0"
DEFAULT_TOKENIZER = "models/1.7b/tokenizer.json"
PROMPT            = b"bonsai> "
PBUF_TOKENS       = 512        # firmware/main.c PBUF_N: the on-card conversation buffer.
LINE_MAX          = 850        # Firmware console line buffer is 1024 B; leave slack.


class BonsaiError(RuntimeError):
    pass


class Bonsai:
    def __init__(self, tty=DEFAULT_TTY, tokenizer=DEFAULT_TOKENIZER, chat_template=True,
                 timeout=600.0):
        """chat_template=False sends the text raw (completion style, like bonsai_chat)."""
        import serial                          # Imported here: only the client needs it.
        from tokenizers import Tokenizer
        self.ser = serial.Serial(tty, 115200, timeout=0.05)
        self.tok = Tokenizer.from_file(tokenizer)
        self.chat_template = chat_template
        self.timeout = timeout
        self.turns = 0                        # Tokens currently in the on-card buffer.
        self.im_end = self.tok.token_to_id("<|im_end|>") if chat_template else None
        self._closed = True                   # Did the last reply end with <|im_end|>?
        self._sync()

    # -- console -------------------------------------------------------------------------------

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _sync(self):
        """Get to a fresh prompt: send an empty line, drain to the next `bonsai> `."""
        self.ser.reset_input_buffer()
        self.ser.write(b"\n")
        self.ser.flush()
        self._read_until(PROMPT, timeout=30.0)

    def _read_until(self, needle, timeout, on_bytes=None):
        buf = bytearray()
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = self.ser.read(4096)
            if chunk:
                buf += chunk
                if on_bytes:
                    on_bytes(chunk)
                if needle in buf:
                    return bytes(buf)
        raise BonsaiError(f"timeout waiting for {needle!r}; got {bytes(buf[-200:])!r}")

    # The firmware's UART RX ring is 128 bytes and its ISR drops bytes when it is full; the
    # console loop drains one byte per iteration and echoes each through a TX ring the host
    # driver empties on a poll timer. A long `tok` line written in one burst loses characters
    # (two ids merge into one past the vocabulary -> the engine fetches a row that does not
    # exist -> ENGSTALL, reconfigure the FPGA; 2026-09-11). So lines go out in chunks, each
    # acknowledged by its echo, byte for byte, before the next.
    PACE_CHUNK = 48

    def _write_paced(self, data):
        """Write `data`, waiting for each chunk's echo. Returns bytes read past the echo
        (the start of the reply, when a read overshoots)."""
        surplus = b""
        for i in range(0, len(data), self.PACE_CHUNK):
            chunk = data[i:i + self.PACE_CHUNK]
            self.ser.write(chunk)
            self.ser.flush()
            got = surplus
            deadline = time.time() + 3.0
            while len(got) < len(chunk) and time.time() < deadline:
                got += self.ser.read(len(chunk) - len(got))
            surplus = got[len(chunk):]
        return surplus

    def command(self, line, timeout=None):
        """Run one firmware console command; returns its output (without the echo/prompt)."""
        surplus = self._write_paced(line.encode() + b"\n")
        out = surplus + self._read_until(PROMPT, timeout or self.timeout)
        text = out.decode(errors="replace")
        return text.rsplit("bonsai>", 1)[0].replace("\r", "").strip("\n")

    def hdrload(self):
        out = self.command("hdrload", timeout=60.0)
        if "inference ready" not in out:
            raise BonsaiError(f"hdrload did not report 'inference ready':\n{out}")
        return out

    def stats(self, reset=False):
        return self.command("stats reset" if reset else "stats")

    # -- generation ----------------------------------------------------------------------------

    def reset(self):
        """Clear the on-card conversation."""
        self.command("tokclr")
        self.turns = 0
        self._closed = True

    def _encode(self, text):
        if self.chat_template:
            # Qwen3 chat turn; the empty <think> block asks for a direct answer (thinking
            # mode otherwise spends the whole budget inside <think>...</think>). On a
            # continuing conversation the previous reply is closed first: the card keeps
            # it in the buffer, and one cut short by max_tokens has no <|im_end|>.
            prefix = "" if self.turns == 0 else ("" if self._closed else "<|im_end|>") + "\n"
            text = (f"{prefix}<|im_start|>user\n{text}<|im_end|>\n"
                    f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return self.tok.encode(text).ids

    def _send_ids(self, ids):
        if self.turns + len(ids) > PBUF_TOKENS:
            raise BonsaiError(f"conversation would exceed the on-card buffer ({PBUF_TOKENS} "
                              "tokens); call reset()")
        i = 0
        while i < len(ids):
            line = "tok"
            while i < len(ids) and len(line) < LINE_MAX:
                line += f" {ids[i]}"
                i += 1
            out = self.command(line, timeout=10.0)
            m = re.search(r"TOK (\d+)", out)
            if not m:
                raise BonsaiError(f"tok not acknowledged: {out!r}")
        self.turns = int(m.group(1))

    def stream(self, text, max_tokens=64):
        """Append `text` to the conversation and yield decoded pieces as tokens arrive."""
        if self.turns == 0:
            self.command("tokclr")
        self._send_ids(self._encode(text))
        # The firmware stops at <|im_end|> (emitted, kept in the buffer) - without it the
        # model runs on past its answer until max_tokens.
        stop = f" {self.im_end}" if self.im_end is not None else ""
        pending = self._write_paced(f"genb {max_tokens}{stop}\n".encode()).decode(errors="replace")
        ids = []
        decoded_upto = 0
        deadline = time.time() + self.timeout
        done = False
        while not done and time.time() < deadline:
            chunk = self.ser.read(4096)
            if not chunk:
                continue
            pending += chunk.decode(errors="replace")
            if "GEN DONE" in pending:
                done = True
            # Token ids stream after "GEN:" as " %d" pieces.
            m = re.search(r"GEN:((?:\s+\d+)*)", pending)
            if m:
                ids = [int(t) for t in m.group(1).split()]
                # Decode incrementally; hold back the last id in case it is a partial UTF-8
                # sequence the next id completes.
                keep = len(ids) if done else max(0, len(ids) - 1)
                if keep > decoded_upto:
                    text_all = self.tok.decode(ids[:keep])
                    text_prev = self.tok.decode(ids[:decoded_upto]) if decoded_upto else ""
                    decoded_upto = keep
                    piece = text_all[len(text_prev):]
                    if piece:
                        yield piece
        if not done:
            raise BonsaiError("generation timed out")
        self.turns += len(ids)
        self._closed = bool(ids) and ids[-1] == self.im_end
        if PROMPT.decode() not in pending.split("GEN DONE", 1)[1]:   # Usually in the same chunk.
            self._read_until(PROMPT, timeout=5.0)

    def generate(self, text, max_tokens=64):
        """Append `text` to the conversation; return the generated text."""
        return "".join(self.stream(text, max_tokens))
