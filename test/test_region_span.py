# SPDX-License-Identifier: BSD-2-Clause
from dataclasses import replace

from migen import run_simulation
import pytest

from trixium.capabilities import Access, Reservation, ReservationKind
from trixium.gateware.region_span import RegionSpanValidator
from test_native_reader import region


def test_full_width_span_resolution_and_reservations():
    regions = [region(7, base=1 << 32, size=1 << 33, endpoint=9, access=Access.WRITE),
               region(99, base=(1 << 64)-4096, size=4096, endpoint=8, access=Access.WRITE),
               region(21, endpoint=4, access=Access.READ)]
    reservations = [Reservation(7, ReservationKind.CPU, 0, 256),
                    Reservation(7, ReservationKind.SYSTEM, 4096, 32)]
    dut = RegionSpanValidator(regions, reservations, [(9, 34), (8, 64), (9, 34)], access=Access.WRITE)
    cases = [(7, 256, 32, 0, 0, (1 << 32)+256),
             (7, 1 << 32, 64, 0, 0, 1 << 33),
             (7, (1 << 33)-32, 32, 0, 0, (12 << 30)-32),
             (99, 4096-32, 32, 0, 1, (1 << 64)-32),
             (7, 0, 32, 4, 0, 0), (7, 4064, 64, 4, 0, 0),
             (7, 256, 0, 1, 0, 0), (7, 257, 32, 1, 0, 0),
             (7, (1 << 33)-32, 64, 2, 0, 0), (7, (1 << 64)-32, 64, 2, 0, 0),
             (99, 4096-32, 64, 2, 0, 0), (99, 32, (1 << 64)-32, 2, 0, 0),
             (21, 0, 32, 1, 0, 0), (13, 0, 32, 1, 0, 0)]
    def run():
        for rid, offset, length, error, port, physical in cases:
            yield dut.region.eq(rid)
            yield dut.offset.eq(offset)
            yield dut.length.eq(length)
            yield
            assert (yield dut.error) == error, (rid, offset, length, (yield dut.error))
            assert bool((yield dut.valid)) == (error == 0)
            if not error:
                assert (yield dut.port) == port
                assert (yield dut.address) == physical
    run_simulation(dut, run())


def test_span_rejects_address_truncation_to_physical_port():
    dut = RegionSpanValidator([region(access=Access.WRITE)], [], [(0, 32)], access=Access.WRITE)
    def run():
        yield dut.region.eq(7)
        yield dut.offset.eq(1 << 32)
        yield dut.length.eq(32)
        yield
        assert (yield dut.error) == 2
    run_simulation(dut, run())


def test_span_constructor_policy_checks():
    good = region(access=Access.WRITE)
    bad = Reservation(8, ReservationKind.CPU, 0, 32)
    for regions, reservations, ports, access in [
        ([], [], [(0, 33)], Access.WRITE), ([good, good], [], [(0, 33)], Access.WRITE),
        ([good, replace(good, id=8)], [], [(0, 33)], Access.WRITE),
        ([good], [bad], [(0, 33)], Access.WRITE), ([good], [], [], Access.WRITE),
        ([good], [], [(0, 11)], Access.WRITE), ([good], [], [(0, 33)], Access(0)),
    ]:
        with pytest.raises(ValueError):
            RegionSpanValidator(regions, reservations, ports, access=access)
