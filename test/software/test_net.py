#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""firmware/net.c + web.c on the host (net_stub.c): ARP, ICMP echo, the TCP handshake, the page,
a streamed /chat reply, /reset, retransmission after a lost segment, checksums of every frame."""

import ctypes
import os
import struct
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FW   = os.path.join(REPO, "firmware")

BOARD_MAC = bytes.fromhex("10e2d5000001")
BOARD_IP  = 0xC0A80133                          # <board-host>
HOST_MAC  = bytes.fromhex("020000000099")
HOST_IP   = 0xC0A80164                          # <board-host>


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    d = tmp_path_factory.mktemp("net")
    page = open(os.path.join(FW, "web_page.html"), "rb").read()
    with open(d / "web_page.h", "w") as f:
        f.write("static const char web_page[] = {" + ",".join(str(b) for b in page) + ",0};\n")
    so = d / "libnet.so"
    # web.c's "web_page.h" resolves next to web.c first: compile a copy beside this page, not the
    # one a firmware build generated in firmware/.
    web_c = d / "web.c"
    web_c.write_bytes(open(os.path.join(FW, "web.c"), "rb").read())
    subprocess.check_call(["gcc", "-O1", "-Wall", "-Werror", "-shared", "-fPIC", f"-I{d}", f"-I{FW}",
                           os.path.join(FW, "net.c"), str(web_c),
                           os.path.join(REPO, "test", "software", "net_stub.c"), "-o", str(so)])
    L = ctypes.CDLL(str(so))
    L.stub_last_line.restype = ctypes.c_char_p
    L.stub_set_now.argtypes = [ctypes.c_uint64]
    L.web_init(ctypes.create_string_buffer(BOARD_MAC, 6), BOARD_IP)
    L.page = page
    return L


# Frames -------------------------------------------------------------------------------------------

def csum(b):
    if len(b) % 2:
        b += b"\0"
    s = sum(struct.unpack(f"!{len(b)//2}H", b))
    while s >> 16:
        s = (s & 0xffff) + (s >> 16)
    return ~s & 0xffff

def ip_packet(proto, payload, src=HOST_IP, dst=BOARD_IP):
    h = struct.pack("!BBHHHBBHII", 0x45, 0, 20 + len(payload), 1, 0x4000, 64, proto, 0, src, dst)
    h = h[:10] + struct.pack("!H", csum(h)) + h[12:]
    return BOARD_MAC + HOST_MAC + b"\x08\x00" + h + payload

def tcp_segment(sport, seq, ack, flags, data=b"", window=65535, opts=b""):
    off = (20 + len(opts)) // 4
    t = struct.pack("!HHIIBBHHH", sport, 80, seq, ack, off << 4, flags, window, 0, 0) + opts + data
    pseudo = struct.pack("!IIBBH", HOST_IP, BOARD_IP, 0, 6, len(t))
    t = t[:16] + struct.pack("!H", csum(pseudo + t)) + t[18:]
    return ip_packet(6, t)

def send(L, frame):
    L.net_rx(ctypes.create_string_buffer(frame, len(frame)), len(frame))

def frames(L):
    out = []
    buf = ctypes.create_string_buffer(1600)
    for i in range(L.stub_count()):
        n = L.stub_frame(i, buf)
        out.append(buf.raw[:n])
    L.stub_clear()
    return out

def parse_tcp(f):
    assert f[:6] == HOST_MAC and f[6:12] == BOARD_MAC and f[12:14] == b"\x08\x00"
    ip = f[14:]
    ihl = (ip[0] & 0xf)*4
    total = struct.unpack("!H", ip[2:4])[0]
    assert csum(ip[:ihl]) == 0, "IP header checksum"
    assert ip[9] == 6
    t = ip[ihl:total]
    pseudo = struct.pack("!IIBBH", BOARD_IP, HOST_IP, 0, 6, len(t))
    assert csum(pseudo + t) == 0, "TCP checksum"
    sport, dport, seq, ack, off, flags, win = struct.unpack("!HHIIBBH", t[:16])
    return dict(seq=seq, ack=ack, flags=flags, data=t[(off >> 4)*4:], win=win, dport=dport, opts=t[20:(off >> 4)*4])

FIN, SYN, RST, PSH, ACK = 1, 2, 4, 8, 16


class Client:
    """A TCP client driving the board's stack: handshake, request, acknowledges everything."""
    def __init__(self, L, port):
        self.L, self.port, self.seq = L, port, 1000
        send(L, tcp_segment(port, self.seq, 0, SYN, opts=b"\x02\x04\x05\xb4"))
        (sa,) = [parse_tcp(f) for f in frames(L)]
        assert sa["flags"] == SYN | ACK and sa["ack"] == self.seq + 1 and sa["opts"][:2] == b"\x02\x04"
        self.seq += 1
        self.ack = sa["seq"] + 1
        send(L, tcp_segment(port, self.seq, self.ack, ACK))
        assert frames(L) == []

    def request(self, data):
        send(self.L, tcp_segment(self.port, self.seq, self.ack, ACK | PSH, data))
        self.seq += len(data)

    def collect(self, rounds=200):
        """Run the server, acknowledge each segment in order; returns (payload, fin)."""
        payload, fin = b"", False
        for _ in range(rounds):
            self.L.web_service()
            segs = [parse_tcp(f) for f in frames(self.L)]
            for s in segs:
                assert s["dport"] == self.port
                if s["seq"] == self.ack and s["data"]:
                    payload += s["data"]
                    self.ack += len(s["data"])
                if s["flags"] & FIN and s["seq"] + len(s["data"]) == self.ack:
                    self.ack += 1
                    fin = True
            if segs:
                send(self.L, tcp_segment(self.port, self.seq, self.ack, ACK | (FIN if fin else 0)))
                if fin:
                    self.seq += 1
                    frames(self.L)
                    return payload, fin
        return payload, fin


# Tests --------------------------------------------------------------------------------------------

def test_arp_reply(lib):
    req = b"\xff"*6 + HOST_MAC + b"\x08\x06" + struct.pack("!HHBBH", 1, 0x0800, 6, 4, 1) + \
          HOST_MAC + struct.pack("!I", HOST_IP) + b"\0"*6 + struct.pack("!I", BOARD_IP)
    send(lib, req)
    (r,) = frames(lib)
    assert r[:6] == HOST_MAC and r[12:14] == b"\x08\x06"
    assert struct.unpack("!H", r[20:22])[0] == 2 and r[22:28] == BOARD_MAC
    # Someone else's address: no answer.
    send(lib, req[:-4] + struct.pack("!I", BOARD_IP + 1))
    assert frames(lib) == []

def test_icmp_echo(lib):
    body = struct.pack("!BBHHH", 8, 0, 0, 0x1234, 1) + b"trixium-ping"
    body = body[:2] + struct.pack("!H", csum(body)) + body[4:]
    send(lib, ip_packet(1, body))
    (r,) = frames(lib)
    ip = r[14:]
    icmp = ip[20:struct.unpack("!H", ip[2:4])[0]]
    assert icmp[0] == 0 and csum(icmp) == 0 and icmp[8:] == b"trixium-ping"

def test_page(lib):
    c = Client(lib, 40001)
    c.request(b"GET / HTTP/1.1\r\nHost: <board-host>\r\n\r\n")
    payload, fin = c.collect()
    head, body = payload.split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 200 OK") and fin
    assert body == lib.page

def test_chat_streams_and_reset(lib):
    c = Client(lib, 40002)
    msg = "Bonjour ! Qu'est-ce qu'un FPGA ?".encode()
    c.request(b"POST /chat HTTP/1.1\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n" % len(msg) + msg)
    payload, fin = c.collect()
    head, body = payload.split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 200 OK") and fin
    text, stats = body.split(b"\x1e")
    assert text.decode() == "Bonjour ! Je suis TriXium." and b'"tokens":5' in stats
    assert lib.stub_last_line().decode() == msg.decode()
    r0 = lib.stub_resets()
    c = Client(lib, 40003)
    c.request(b"POST /reset HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
    payload, _ = c.collect()
    assert payload.startswith(b"HTTP/1.1 200 OK") and lib.stub_resets() == r0 + 1

def test_body_in_a_second_segment(lib):
    c = Client(lib, 40004)
    msg = b"hello"
    turns = lib.stub_turns()
    c.request(b"POST /chat HTTP/1.1\r\nContent-Length: 5\r\n\r\n")
    lib.web_service()
    frames(lib)
    assert lib.stub_turns() == turns              # Waits for the body.
    c.request(msg)
    payload, fin = c.collect()
    assert b"Bonjour" in payload and lib.stub_last_line() == b"hello"

def test_retransmission(lib):
    c = Client(lib, 40005)
    c.request(b"GET / HTTP/1.1\r\n\r\n")
    lib.web_service()
    first = [s for s in (parse_tcp(f) for f in frames(lib)) if s["data"]]
    assert first and first[0]["seq"] == c.ack
    # Nothing acknowledged: past the RTO the stack sends from the first unacknowledged byte again.
    lib.stub_set_now(400000)
    lib.net_poll()
    again = [s for s in (parse_tcp(f) for f in frames(lib)) if s["data"]]
    assert again and again[0]["seq"] == c.ack and again[0]["data"].startswith(first[0]["data"])
    lib.stub_set_now(0)
    payload, fin = c.collect()
    assert payload.split(b"\r\n\r\n", 1)[1] == lib.page and fin

def test_unknown_segment_gets_reset(lib):
    send(lib, tcp_segment(40999, 5, 77, ACK, b"x"))
    (r,) = [parse_tcp(f) for f in frames(lib)]
    assert r["flags"] & RST and r["seq"] == 77
