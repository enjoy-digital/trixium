# SPDX-License-Identifier: BSD-2-Clause
from types import SimpleNamespace

import pytest
from migen import Memory, Module, Signal, run_simulation
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import CSRStorage

from trixium.gateware.integration import BonsaiClusterConfig
from trixium.gateware.region_read_csr import BonsaiRegionReadCSR
from trixium.gateware.sequencer import BonsaiSequencer
from trixium.model.native_sequence import region_fetch_descriptor
from test_integration import write
from test_native_reader import region
from test_projection_restore_sp import pack


def descriptor(**changes):
    fields = dict(region=0xfedcba98, offset=0x123456789abcde00, length=35, dst=0x800)
    fields.update(changes)
    return region_fetch_descriptor(**fields)


def fixture(descriptors):
    dut = Module()
    dut.submodules.seq = seq = BonsaiSequencer(native_pipes=2, with_region_fetch=True)
    words = [pack(d[start:start+8]) for d in descriptors for start in (0, 8)]
    mem = Memory(256, max(2, len(words)), init=words)
    port = mem.get_port()
    dut.specials += mem, port
    dut.comb += [port.adr.eq(seq.sp_adr), seq.sp_dat.eq(port.dat_r)]
    return dut


def start(seq, count, base=0):
    yield from write(seq.base, base)
    yield from write(seq.count, count)
    yield from write(seq.start, 1)


def test_region_fetch_descriptor_and_configuration():
    assert descriptor() == [10, 0xfedcba98, 0x9abcde00, 0x12345678, 35, 0x800] + [0]*10
    for changes in ({'region': -1}, {'region': 1 << 32}, {'offset': 1 << 64}, {'offset': 1},
                    {'length': 0}, {'length': 1 << 24}, {'dst': 1 << 24}, {'dst': 4}):
        with pytest.raises(ValueError):
            descriptor(**changes)
    with pytest.raises(ValueError):
        BonsaiSequencer(with_region_fetch=True)
    with pytest.raises(ValueError):
        BonsaiClusterConfig(with_region_fetch=True)
    with pytest.raises(ValueError):
        BonsaiClusterConfig(with_region_fetch=True, with_vecunit=False,
                            native_regions=(region(),), native_endpoints=(0,))
    assert not hasattr(BonsaiSequencer(native_pipes=1), 'region_fetch_cmd')


@pytest.mark.parametrize('field,value,error', [(0, 10 | 1 << 9, 2), (0, 10 | 1 << 5, 2),
    (0, 11, 2), (4, 1 << 24, 2), (5, 1 << 24, 2), (5, 64, 4),
    (5, 32, 4)] + [(field, 1, 2) for field in range(6, 16)])
def test_region_fetch_descriptor_rejection_precedes_memory(field, value, error):
    desc = descriptor()
    desc[field] = value
    dut = fixture([desc, [0]*16])
    def run():
        yield from start(dut.seq, 2)
        for _ in range(40):
            yield
            assert not (yield dut.seq.region_fetch_cmd.valid)
        assert (yield dut.seq.region_fetch_error.status) == error
        assert (yield dut.seq.busy) and not (yield dut.seq.status.fields.dcount)
    run_simulation(dut, run())


@pytest.mark.parametrize('failure', [0, 1, 4, 0x80, 0x100])
def test_region_fetch_full_width_waits_nowait_and_failure_wins(failure):
    legacy_fetch = [1 | (1 << 9), 2, 0, 0, 0x100, 7, 0, 0, 0, 0x800] + [0]*6
    dut = fixture([legacy_fetch, descriptor()])
    seq, cmd = dut.seq, dut.seq.region_fetch_cmd
    def run():
        assert (yield seq.region_fetch_abi.status) == 0x10000
        yield seq.region_fetch_manual_pending.eq(1)
        yield from start(seq, 2)
        for _ in range(8):
            assert (yield seq.native_manual_drain) and not (yield seq.sp_active)
            yield
        yield seq.region_fetch_manual_pending.eq(0)
        for _ in range(30):
            yield
            if (yield seq.eng_valid):
                break
        else:
            raise AssertionError('legacy fetch did not dispatch')
        yield seq.eng_ready.eq(1)
        yield
        yield seq.eng_ready.eq(0)
        for _ in range(30):
            yield
            assert not (yield cmd.valid)
        yield seq.eng_done.eq(1)
        yield
        yield seq.eng_done.eq(0)
        for _ in range(20):
            yield
            if (yield cmd.valid):
                break
        else:
            raise AssertionError('region fetch did not dispatch after drain')
        for _ in range(8):
            assert (yield cmd.valid) and (yield seq.status.fields.dcount) == 1
            for name, value in dict(region=0xfedcba98, offset=0x123456789abcde00, length=35, dst=0x800).items():
                assert (yield getattr(cmd, name)) == value
            yield
        yield cmd.ready.eq(1)
        yield
        yield cmd.ready.eq(0)
        for _ in range(5):
            yield
        yield seq.region_fetch_failed.eq(failure)
        yield seq.region_fetch_done.eq(1)
        yield seq.eng_done.eq(1)
        yield
        yield seq.region_fetch_done.eq(0)
        yield seq.eng_done.eq(0)
        for _ in range(15):
            yield
        assert (yield seq.region_fetch_error.status) == failure << 8
        assert (yield seq.status.fields.dcount) == (1 if failure else 2)
        assert bool((yield seq.busy)) == bool(failure)
    run_simulation(dut, run())


def test_older_manual_read_failure_stops_list_before_fetch():
    dut = fixture([descriptor()])
    def run():
        yield dut.seq.region_fetch_manual_pending.eq(1)
        yield dut.seq.region_fetch_manual_failed.eq(1)
        yield dut.seq.region_fetch_failed.eq(4)
        yield from start(dut.seq, 1)
        for _ in range(8):
            yield
            assert not (yield dut.seq.sp_active) and not (yield dut.seq.region_fetch_cmd.valid)
        assert (yield dut.seq.region_fetch_error.status) == 0x408
        assert (yield dut.seq.busy) and not (yield dut.seq.status.fields.dcount)
    run_simulation(dut, run())


@pytest.mark.parametrize('raw_completion', [False, True])
def test_raw_or_manual_completion_does_not_replay_previous_k_capture(raw_completion):
    capture = [1 | (1 << 5), 2, 0, 0, 0x100, 1, 0, 0, 0, 0x800] + [0]*6
    # A0 = immediate zero + kreg0; expose the preserved k without private signals.
    vector = [2 | (1 << 11), 1 << 16] + [0]*14
    dut = fixture([capture] + ([descriptor()] if raw_completion else []) + [vector])
    seq = dut.seq
    def run():
        yield seq.absmax.eq(1 << 20)  # bit_length 21 - 14 = k7.
        yield from start(seq, 3 if raw_completion else 2)
        for _ in range(30):
            yield
            if (yield seq.eng_valid):
                break
        assert (yield seq.eng_valid)
        yield seq.eng_ready.eq(1)
        yield
        yield seq.eng_ready.eq(0)
        yield seq.eng_done.eq(1)
        yield
        yield seq.eng_done.eq(0)
        if raw_completion:
            for _ in range(30):
                yield
                if (yield seq.region_fetch_cmd.valid):
                    break
            assert (yield seq.region_fetch_cmd.valid)
            yield seq.region_fetch_cmd.ready.eq(1)
            yield
            yield seq.region_fetch_cmd.ready.eq(0)
            yield seq.absmax.eq(1 << 30)  # A raw fetch must not recapture k17.
            for _ in range(5):
                yield
            yield seq.region_fetch_done.eq(1)
            yield seq.eng_done.eq(1)
            yield
            yield seq.region_fetch_done.eq(0)
            yield seq.eng_done.eq(0)
        for _ in range(30):
            yield
            if (yield seq.vec_start):
                break
        assert (yield seq.vec_start) and (yield seq.vec_args[0]) == 7
        yield seq.vec_done.eq(1)
        yield
        yield seq.vec_done.eq(0)
        for _ in range(15):
            yield
        assert not (yield seq.busy)
        if not raw_completion:
            # A second list waits for an older manual read; eng_done during
            # SBDRAIN must not use the first list's still-latched cap_en.
            yield seq.region_fetch_manual_pending.eq(1)
            yield from start(seq, 1, base=64)
            assert (yield seq.native_manual_drain)
            yield seq.absmax.eq(1 << 30)
            yield seq.eng_done.eq(1)
            yield
            yield seq.eng_done.eq(0)
            for _ in range(5):
                yield
            yield seq.region_fetch_manual_pending.eq(0)
            for _ in range(30):
                yield
                if (yield seq.vec_start):
                    break
            assert (yield seq.vec_start) and (yield seq.vec_args[0]) == 7
    run_simulation(dut, run())


def test_region_read_csr_owner_mux_manual_drain_reject_and_accounting():
    dut = Module()
    fields = [('region', 32), ('offset', 64), ('length', 24), ('dst', 24)]
    engine = SimpleNamespace(region_read_cmd=stream.Endpoint(fields), region_read_error=Signal(8),
        region_read_failed=Signal(), region_read_done=Signal(), region_read_busy=Signal(), region_read_beats=Signal(24))
    control = SimpleNamespace(rst_trigger=Signal(), rst_busy=Signal())
    seq = SimpleNamespace(region_fetch_cmd=stream.Endpoint(fields), region_fetch_done=Signal(),
        region_fetch_failed=Signal(9), region_fetch_manual_pending=Signal(), region_fetch_manual_failed=Signal(),
        busy=Signal(), native_manual_drain=Signal(), start=CSRStorage(1, name='seq_start'))
    dut.submodules.csr = csr = BonsaiRegionReadCSR(engine, control, seq=seq)
    def run():
        manual = dict(region=7, offset=(1 << 32)+128, length=3, dst=0x800)
        for name, value in manual.items():
            yield from write(getattr(csr, name), value)
        yield from write(csr.start, 1)
        yield seq.busy.eq(1)
        yield seq.native_manual_drain.eq(1)
        for _ in range(5):
            yield
            assert (yield seq.region_fetch_manual_pending)
            for name, value in manual.items():
                assert (yield getattr(engine.region_read_cmd, name)) == value
        yield engine.region_read_cmd.ready.eq(1)
        yield
        yield
        yield engine.region_read_cmd.ready.eq(0)
        assert (yield csr.accepted.status) == 1 and not (yield csr.status.fields.pending)
        yield engine.region_read_done.eq(1)
        yield
        yield engine.region_read_done.eq(0)
        yield
        assert (yield csr.completed.status) == 1 and not (yield seq.region_fetch_manual_pending)
        yield seq.native_manual_drain.eq(0)
        sequenced = dict(region=42, offset=(1 << 48)+256, length=35, dst=0x2000)
        for name, value in sequenced.items():
            yield getattr(seq.region_fetch_cmd, name).eq(value)
        yield seq.region_fetch_cmd.valid.eq(1)
        yield from write(csr.start, 1)  # Rejected, never queued/replayed.
        assert (yield csr.error.status) == 256 and not (yield csr.status.fields.pending)
        assert not (yield seq.region_fetch_failed)
        for _ in range(5):
            yield
            assert (yield engine.region_read_cmd.valid)
            for name, value in sequenced.items():
                assert (yield getattr(engine.region_read_cmd, name)) == value
        yield engine.region_read_cmd.ready.eq(1)
        yield
        assert (yield seq.region_fetch_cmd.ready)
        yield seq.region_fetch_cmd.valid.eq(0)
        yield engine.region_read_cmd.ready.eq(0)
        yield engine.region_read_done.eq(1)
        yield
        assert (yield seq.region_fetch_done)
        yield engine.region_read_done.eq(0)
        yield seq.busy.eq(0)
        for _ in range(5):
            yield
        assert (yield csr.accepted.status) == (yield csr.completed.status) == 1
        assert not (yield engine.region_read_cmd.valid)
        yield control.rst_trigger.eq(1)
        yield
        yield control.rst_trigger.eq(0)
        yield
        assert not (yield csr.error.status) and not (yield seq.region_fetch_manual_pending)
        # Simultaneous list start and new manual read is rejected even before busy.
        yield seq.start.re.eq(1)
        yield seq.start.storage.eq(1)
        yield from write(csr.start, 1)
        yield seq.start.re.eq(0)
        yield
        assert (yield csr.error.status) == 256 and not (yield engine.region_read_cmd.valid)
        assert (yield csr.accepted.status) == 1
    run_simulation(dut, run())
