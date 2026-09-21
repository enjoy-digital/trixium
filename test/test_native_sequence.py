# SPDX-License-Identifier: BSD-2-Clause
import pytest
from migen import Memory, Module, run_simulation

from trixium.gateware.sequencer import BonsaiSequencer
from trixium.model.native_sequence import native_pipe_descriptor, native_projection_descriptor
from test_activation_sp import pack
from test_integration import write


def projection(**changes):
    args = dict(kind=0, rows=3, rem=1, groups=68, bias=17, rshift=9, xsrc=0x800, dst=0x8000)
    args.update(changes)
    return native_projection_descriptor(**args)


def fixture(descriptors, base=0, pipes=2):
    dut = Module()
    dut.submodules.seq = seq = BonsaiSequencer(native_pipes=pipes)
    words = [pack(desc[start:start+8], 32) for desc in descriptors for start in (0, 8)]
    mem = Memory(256, max(2, len(words)), init=words)
    port = mem.get_port()
    dut.specials += mem, port
    dut.comb += [port.adr.eq(seq.sp_adr - base//32), seq.sp_dat.eq(port.dat_r)]
    return dut


def start(seq, count, base=0):
    yield from write(seq.base, base)
    yield from write(seq.count, count)
    yield from write(seq.start, 1)


def test_native_sequence_encoders_and_configuration():
    cfg = native_pipe_descriptor(pipe=1, region=0xfedcba98, offset=0xfeed0123456789ab, stride=0x123456789abcdef0)
    assert cfg == [6, 1, 0xfedcba98, 0x456789ab, 0xfeed0123, 0x9abcdef0, 0x12345678] + [0]*9
    assert projection() == [7, 0, 3 | 1 << 16, 0, 0, 0, 68 << 8, 17 | 9 << 16, 0x800, 0x8000] + [0]*6
    for changes in ({'pipe': -1}, {'pipe': 32}, {'region': 1 << 32}, {'offset': 1 << 64}, {'stride': -1}):
        args = dict(pipe=0, region=7, offset=0, stride=3808)
        args.update(changes)
        with pytest.raises(ValueError):
            native_pipe_descriptor(**args)
    for changes in ({'kind': 2}, {'groups': 0}, {'groups': 256}, {'rows': 0, 'rem': 0}, {'rows': 65536},
                    {'rem': 32}, {'bias': 128}, {'rshift': 64}, {'xsrc': 1}, {'dst': 3},
                    {'qstride': 65536}, {'dual': 2}, {'interleaved': -1}):
        with pytest.raises(ValueError):
            projection(**changes)
    for pipes in (-1, 32):
        with pytest.raises(ValueError):
            BonsaiSequencer(native_pipes=pipes)
    with pytest.raises(ValueError):
        BonsaiSequencer(native_pipes=1, scratchpad_size=512*1024)


@pytest.mark.parametrize('kind,groups,cols', [(0, 68, 8704), (0, 136, 17408), (1, 255, 8160)])
def test_native_sequence_full_width_config_backpressure_and_columns(kind, groups, cols):
    cfg = native_pipe_descriptor(pipe=1, region=0xfedcba98, offset=0xfeed0123456789ab, stride=0x123456789abcdef0)
    dut = fixture([cfg, projection(kind=kind, groups=groups)])
    seq = dut.seq
    def run():
        assert (yield seq.native_abi.status) == 0x10000
        yield seq.native_manual_pending.eq(1)
        yield seq.native_engine_busy.eq(1)
        yield from start(seq, 2)
        for _ in range(5):
            assert (yield seq.busy) and (yield seq.native_manual_drain)
            assert not (yield seq.sp_active) and not (yield seq.eng_valid)
            yield
        yield seq.native_manual_pending.eq(0)
        for _ in range(5):
            yield
            assert not (yield seq.sp_active)
        yield seq.native_engine_busy.eq(0)
        for _ in range(30):
            yield
            if (yield seq.native_cfg.valid):
                break
        else:
            raise AssertionError('native configuration did not dispatch')
        for _ in range(5):
            assert (yield seq.native_cfg.valid) and not (yield seq.eng_valid)
            for field, value in (('chan', 1), ('region', 0xfedcba98), ('offset', 0xfeed0123456789ab), ('stride', 0x123456789abcdef0)):
                assert (yield getattr(seq.native_cfg, field)) == value
            yield
        yield seq.native_cfg.ready.eq(1)
        yield
        yield seq.native_cfg.ready.eq(0)
        for _ in range(30):
            yield
            if (yield seq.eng_valid):
                break
        else:
            raise AssertionError('native projection did not dispatch')
        for _ in range(5):
            assert (yield seq.eng_native) and (yield seq.eng_cols) == cols
            assert (yield seq.eng_valid) and (yield seq.status.fields.dcount) == 1
            yield
        yield seq.eng_ready.eq(1)
        yield
        yield seq.eng_ready.eq(0)
        for _ in range(5):
            yield
            assert not (yield seq.eng_valid) and (yield seq.busy)
        yield seq.eng_done.eq(1)
        yield
        yield seq.eng_done.eq(0)
        for _ in range(10):
            yield
        assert not (yield seq.busy) and (yield seq.status.fields.dcount) == 2
        assert not (yield seq.native_error.status)
    run_simulation(dut, run())


@pytest.mark.parametrize('field,value', [(0, 7 | 1 << 9), (1, 2), (1, 1 << 10), (1, 1 << 31),
    (2, 1 << 21), (3, 8704), (4, 1), (5, 1), (6, 1), (7, 1 << 7), (7, 1 << 22), (7, 1 << 28),
    (8, 1 << 24), (9, 1 << 24), (10, 1), (15, 1)])
def test_native_projection_rejects_reserved_fields_before_engine_access(field, value):
    words = projection()
    words[field] = value
    dut = fixture([words])
    def run():
        yield from start(dut.seq, 1)
        for _ in range(25):
            assert not (yield dut.seq.eng_valid) and not (yield dut.seq.native_cfg.valid)
            yield
        assert (yield dut.seq.native_error.status) == 2
        assert (yield dut.seq.busy) and not (yield dut.seq.status.fields.dcount)
    run_simulation(dut, run())


@pytest.mark.parametrize('field,value,error', [(0, 6 | 1 << 9, 2), (1, 2, 8), (1, 1 << 24, 8), (7, 1, 2), (15, 1, 2)])
def test_native_config_rejects_invalid_pipe_and_reserved_fields(field, value, error):
    words = native_pipe_descriptor(pipe=0, region=7, offset=0, stride=3808)
    words[field] = value
    dut = fixture([words])
    def run():
        yield from start(dut.seq, 1)
        for _ in range(25):
            assert not (yield dut.seq.eng_valid) and not (yield dut.seq.native_cfg.valid)
            yield
        assert (yield dut.seq.native_error.status) == error
        assert (yield dut.seq.busy) and not (yield dut.seq.status.fields.dcount)
    run_simulation(dut, run())


@pytest.mark.parametrize('dual,dst,qstride', [(False, 0x1000, 0), (True, 0x800, 512)])
def test_native_projection_protects_complete_output_spans(dual, dst, qstride):
    dut = fixture([projection(dual=dual, dst=dst, qstride=qstride)], base=0x1000)
    def run():
        yield from start(dut.seq, 1, base=0x1000)
        for _ in range(25):
            assert not (yield dut.seq.eng_valid)
            yield
        assert (yield dut.seq.native_error.status) == 4
    run_simulation(dut, run())


def test_native_projection_protects_later_descriptor_before_dispatch():
    dut = fixture([projection(dst=64), [0]*16])
    def run():
        yield from start(dut.seq, 2)
        for _ in range(25):
            assert not (yield dut.seq.eng_valid)
            yield
        assert (yield dut.seq.native_error.status) == 4
        assert (yield dut.seq.busy) and not (yield dut.seq.status.fields.dcount)
    run_simulation(dut, run())


@pytest.mark.parametrize('base,count', [(1, 1), (0xffffe0, 1), (0, 65535), (256*1024 - 32, 1)])
def test_native_sequence_checks_list_capacity_before_fetch(base, count):
    seq = BonsaiSequencer(native_pipes=2)
    def run():
        yield from start(seq, count, base)
        for _ in range(10):
            assert not (yield seq.sp_active) and not (yield seq.native_cfg.valid) and not (yield seq.eng_valid)
            yield
        assert (yield seq.native_error.status) == 1
        assert (yield seq.busy) and not (yield seq.status.fields.dcount)
    run_simulation(seq, run())


@pytest.mark.parametrize('at_start', [False, True])
def test_native_sequence_propagates_failure_without_completing(at_start):
    dut = fixture([projection()])
    seq = dut.seq
    def run():
        yield seq.native_failed.eq(at_start)
        yield seq.native_engine_busy.eq(at_start)
        yield from start(seq, 1)
        if not at_start:
            yield seq.eng_ready.eq(1)
            for _ in range(30):
                yield
                if (yield seq.eng_valid):
                    break
            else:
                raise AssertionError('projection did not dispatch')
            yield
            yield seq.eng_done.eq(1)
            yield seq.native_failed.eq(1)  # Failure must win over completion.
        for _ in range(10):
            yield
        assert (yield seq.native_error.status) == 256
        assert (yield seq.busy) and not (yield seq.status.fields.dcount)
    run_simulation(dut, run())


@pytest.mark.parametrize('kind,rows,rem,groups,dual,interleaved', [
    (0, 2, 1, 68, False, False), (1, 3, 1, 5, True, True), (0, 0, 1, 1, False, True)])
def test_shared_native_sequence_exact_projection_modes_and_physical_addresses(kind, rows, rem, groups, dual, interleaved):
    from types import SimpleNamespace
    from migen.sim import passive
    from trixium.gateware.integration import BonsaiClusterConfig
    from test_integration import SoC
    from test_native_reader import region
    from test_native_engine import make_engine_case
    from test_hadamard_integration import install_list
    dut = SoC(BonsaiClusterConfig(with_dual=dual, scratchpad_size=64*1024,
        native_regions=(region(7, size=1 << 20), region(42, base=1 << 32, size=1 << 20)),
        native_endpoints=(0, 0)), n=2)
    case = make_engine_case(SimpleNamespace(axis=dut.ports), kind, rows, rem, groups, dual, interleaved)
    cmd = case['cmd']
    cmd['xsrc'], cmd['dst'] = 0x800, 0x8004
    descriptors = [native_pipe_descriptor(pipe=i, region=descriptor[0], offset=descriptor[1], stride=descriptor[2])
                   for i, descriptor in enumerate(case['descriptors']) if rows or i < rem]
    descriptors.append(native_projection_descriptor(kind=kind, rows=rows, rem=rem, groups=groups,
        bias=cmd['bias'], rshift=cmd['rshift'], xsrc=cmd['xsrc'], dst=cmd['dst'], dual=dual,
        qstride=cmd['qstride'], interleaved=interleaved))
    observed = [[], []]
    @passive
    def monitor():
        while True:
            for i, port in enumerate(dut.ports):
                if (yield port.ar.valid) and (yield port.ar.ready):
                    address, beats = (yield port.ar.addr), (yield port.ar.len) + 1
                    observed[i].extend(address + 32*b for b in range(beats))
            yield
    def run():
        for channel, address, beats in case['reads']:
            for b in range(beats):
                data = bytes(case['memory'].get(address + 32*b + j, 0xa5) for j in range(32))
                yield dut.models[channel].mem[((address + 32*b) % (256*1024))//32].eq(int.from_bytes(data, 'little'))
        xb = case['xb']
        for pos in range(0, len(xb), 32):
            yield dut.scratchpad.mem[(cmd['xsrc'] + pos)//32].eq(int.from_bytes(xb[pos:pos+32], 'little'))
        yield from install_list(dut, descriptors)
        accepted = False
        for _ in range(5000):
            assert not (yield dut.seq.native_error.status)
            if (yield dut.engine.cmd.valid) and (yield dut.engine.cmd.ready):
                assert (yield dut.engine.cmd.native) and (yield dut.engine.cmd.cols) == cmd['cols']
                assert (yield dut.engine.cmd.bias) == cmd['bias'] and (yield dut.engine.cmd.rshift) == cmd['rshift']
                accepted = True
            if not (yield dut.seq.busy):
                break
            yield
        else:
            raise AssertionError('shared native list did not complete')
        assert accepted and (yield dut.seq.status.fields.dcount) == len(descriptors)
        assert (yield dut.engine_csr.done_count.status) == 1
        for head, results in enumerate(case['expected']):
            for row, expected in enumerate(results):
                address = cmd['dst'] + 4*(head*cmd['qstride'] + row)
                data = yield dut.scratchpad.mem[address//32]
                got = (data >> (8*(address % 32))) & 0xffffffff
                assert got == int(expected) & 0xffffffff
        for _ in range(5):
            yield
    run_simulation(dut, [run(), monitor()], clocks={'sys': 10, 'eng': 10})
    for channel in range(2):
        assert observed[channel] == [address + 32*b for i, address, beats in case['reads']
                                     if i == channel for b in range(beats)]
