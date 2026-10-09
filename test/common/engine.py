#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Migen testbench helpers shared by the engine, sequencer and scratchpad tests: scratchpad
read-back and the vector unit's CSR program upload/submit (DUTs exposing `scratchpad` and `vu`)."""

# Helpers ------------------------------------------------------------------------------------------

def vec_csr_upload(dut, words):
    yield dut.vu.prog_adr.storage.eq(0)
    yield dut.vu.prog_adr.re.eq(1)
    yield
    yield dut.vu.prog_adr.re.eq(0)
    for wd in words:
        yield dut.vu.prog.storage.eq(wd)
        yield dut.vu.prog.re.eq(1)
        yield
    yield dut.vu.prog.re.eq(0)
    yield


def vec_csr_submit(dut, args):
    arg_csrs = [dut.vu.arg0, dut.vu.arg1, dut.vu.arg2, dut.vu.arg3,
                dut.vu.arg4, dut.vu.arg5, dut.vu.arg6, dut.vu.arg7]
    for i, a in enumerate(args):
        yield arg_csrs[i].storage.eq(a & 0xFFFFFFFF)
    yield dut.vu.pc.storage.eq(0)
    yield dut.vu.submit.storage.eq(1)
    yield dut.vu.submit.re.eq(1)
    yield
    yield dut.vu.submit.re.eq(0)
    yield


def read_sp(dut, start, length):
    data = bytearray()
    for wd in range(start//32, (start + length + 31)//32):
        v = (yield dut.scratchpad.mem[wd])
        data += int(v).to_bytes(32, "little")
    return bytes(data[:length])
