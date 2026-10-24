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
    L.net_ip.restype = ctypes.c_uint32
    L.web_init(ctypes.create_string_buffer(BOARD_MAC, 6), BOARD_IP)
    # No DHCP server here: past the wait the static address (the DISCOVERs dropped).
    L.stub_set_now(T0)
    L.net_poll()
    frames(L)
    assert L.net_ip() == BOARD_IP and not L.net_ip_dhcp()
    L.page = page
    L.so = str(so)
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
T0 = 5_000_000                                  # Past the 4 s DHCP wait: the clock never goes back.


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

    def collect(self, rounds=200, first=None):
        """Run the server, acknowledge each segment in order (from `first`, segments already
        taken off the wire, if given); returns (payload, fin)."""
        payload, fin = b"", False
        for i in range(rounds):
            if i == 0 and first is not None:
                segs = first
            else:
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
    lib.stub_set_now(T0 + 400000)
    lib.net_poll()
    again = [parse_tcp(f) for f in frames(lib)]
    data = [s for s in again if s["data"]]
    assert data and data[0]["seq"] == c.ack and data[0]["data"].startswith(first[0]["data"])
    payload, fin = c.collect(first=again)
    assert payload.split(b"\r\n\r\n", 1)[1] == lib.page and fin

def test_unknown_segment_gets_reset(lib):
    send(lib, tcp_segment(40999, 5, 77, ACK, b"x"))
    (r,) = [parse_tcp(f) for f in frames(lib)]
    assert r["flags"] & RST and r["seq"] == 77


# DHCP and mDNS ------------------------------------------------------------------------------------

def udp_frame(sport, dport, payload, src_ip, dst_ip, dst_mac=None, src_mac=HOST_MAC):
    u = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    h = struct.pack("!BBHHHBBHII", 0x45, 0, 20 + len(u), 1, 0, 64, 17, 0, src_ip, dst_ip)
    h = h[:10] + struct.pack("!H", csum(h)) + h[12:]
    return (dst_mac or BOARD_MAC) + src_mac + b"\x08\x00" + h + u

def parse_udp(f):
    ip = f[14:]
    ihl = (ip[0] & 0xf)*4
    total = struct.unpack("!H", ip[2:4])[0]
    assert ip[9] == 17 and csum(ip[:ihl]) == 0
    u = ip[ihl:total]
    src, dst = struct.unpack("!II", ip[12:20])
    pseudo = struct.pack("!IIBBH", src, dst, 0, 17, len(u))
    assert csum(pseudo + u) in (0, 0xffff) or struct.unpack("!H", u[6:8])[0] == 0, "UDP checksum"
    sport, dport = struct.unpack("!HH", u[:4])
    return dict(eth_dst=f[:6], src=src, dst=dst, ttl=ip[8], sport=sport, dport=dport, data=u[8:])

def dhcp_options(m):
    opts, o = {}, 240
    while o < len(m) and m[o] != 255:
        if m[o] == 0:
            o += 1; continue
        opts[m[o]] = m[o + 2:o + 2 + m[o + 1]]
        o += 2 + m[o + 1]
    return opts

def dhcp_reply(req, mtype, yiaddr, server, lease):
    m = bytearray(300)
    m[0] = 2; m[1] = 1; m[2] = 6
    m[4:8] = req[4:8]
    m[16:20] = struct.pack("!I", yiaddr)
    m[28:34] = req[28:34]
    m[236:240] = struct.pack("!I", 0x63825363)
    o = 240
    for code, val in ((53, bytes([mtype])), (54, struct.pack("!I", server)), (51, struct.pack("!I", lease)), (1, b"\xff\xff\xff\x00")):
        m[o] = code; m[o + 1] = len(val); m[o + 2:o + 2 + len(val)] = val; o += 2 + len(val)
    m[o] = 255
    return udp_frame(67, 68, bytes(m), server, 0xffffffff, dst_mac=b"\xff"*6)

@pytest.fixture
def fresh(lib, tmp_path):
    """A second instance of the stack (its own globals): a boot on a LAN with a DHCP server."""
    import shutil
    so = tmp_path / "libnet2.so"
    shutil.copy(lib.so, so)
    L = ctypes.CDLL(str(so))
    L.net_ip.restype = ctypes.c_uint32
    L.stub_set_now.argtypes = [ctypes.c_uint64]
    L.stub_set_now(1000)
    L.web_init(ctypes.create_string_buffer(BOARD_MAC, 6), BOARD_IP)
    return L

def test_dhcp_lease_and_renewal(fresh):
    L = fresh
    (d,) = [parse_udp(f) for f in frames(L)]
    assert d["eth_dst"] == b"\xff"*6 and d["src"] == 0 and d["dst"] == 0xffffffff and (d["sport"], d["dport"]) == (68, 67)
    disc = d["data"]
    assert disc[0] == 1 and disc[28:34] == BOARD_MAC and dhcp_options(disc)[53] == b"\x01"
    assert dhcp_options(disc)[12] == b"trixium" and L.net_ip() == 0
    SERVER, LEASED = 0x0A000001, 0x0A00002A                       # 10.0.0.1, 10.0.0.42
    send(L, dhcp_reply(disc, 2, LEASED, SERVER, 120))               # OFFER
    (r,) = [parse_udp(f) for f in frames(L)]
    o = dhcp_options(r["data"])
    assert o[53] == b"\x03" and o[50] == struct.pack("!I", LEASED) and o[54] == struct.pack("!I", SERVER)
    send(L, dhcp_reply(r["data"], 5, LEASED, SERVER, 120))          # ACK
    assert L.net_ip() == LEASED and L.net_ip_dhcp() and L.net_ip_changed()
    # Past the 4 s wait nothing falls back; past half the lease (60 s) the renewal goes out.
    L.stub_set_now(10_000_000); L.net_poll()
    assert L.net_ip() == LEASED and frames(L) == []
    L.stub_set_now(61_000_000); L.net_poll()
    (n,) = [parse_udp(f) for f in frames(L)]
    assert dhcp_options(n["data"])[53] == b"\x03" and n["data"][12:16] == struct.pack("!I", LEASED) and n["src"] == LEASED

def test_dhcp_nak_restarts(fresh):
    L = fresh
    (d,) = [parse_udp(f) for f in frames(L)]
    send(L, dhcp_reply(d["data"], 2, 0x0A00002A, 0x0A000001, 120))
    (r,) = [parse_udp(f) for f in frames(L)]
    send(L, dhcp_reply(r["data"], 6, 0, 0x0A000001, 0))             # NAK: DISCOVER again
    (again,) = [parse_udp(f) for f in frames(L)]
    assert dhcp_options(again["data"])[53] == b"\x01" and L.net_ip() == 0

def mdns_query(name, qtype=1, sport=5353, qid=0):
    q = struct.pack("!HHHHHH", qid, 0, 1, 0, 0, 0)
    for label in name.split("."):
        q += bytes([len(label)]) + label.encode()
    return q + b"\0" + struct.pack("!HH", qtype, 1)

def test_mdns_answers_its_name(lib):
    send(lib, udp_frame(5353, 5353, mdns_query("TriXium.local"), HOST_IP, 0xe00000fb, dst_mac=bytes.fromhex("01005e0000fb")))
    (r,) = [parse_udp(f) for f in frames(lib)]
    assert r["dst"] == 0xe00000fb and r["eth_dst"] == bytes.fromhex("01005e0000fb") and r["ttl"] == 255
    m = r["data"]
    assert struct.unpack("!HHHH", m[2:10]) == (0x8400, 0, 1, 0)
    assert m[-4:] == struct.pack("!I", BOARD_IP)
    # Another name, or a response: nothing.
    send(lib, udp_frame(5353, 5353, mdns_query("other.local"), HOST_IP, 0xe00000fb, dst_mac=bytes.fromhex("01005e0000fb")))
    assert frames(lib) == []
    # A plain resolver's query (not from 5353): a unicast answer with its id and question.
    send(lib, udp_frame(40000, 5353, mdns_query("trixium.local", qid=0x1234), HOST_IP, 0xe00000fb, dst_mac=bytes.fromhex("01005e0000fb")))
    (u,) = [parse_udp(f) for f in frames(lib)]
    assert u["dst"] == HOST_IP and u["dport"] == 40000 and u["data"][:2] == b"\x12\x34"
    assert struct.unpack("!H", u["data"][4:6])[0] == 1 and u["data"][-4:] == struct.pack("!I", BOARD_IP)
