# SPDX-License-Identifier: BSD-2-Clause
"""Passive, coherent memory traffic measurement; no backpressure or clock crossings.

AXI sources count accepted logical bytes (WSTRB for writes). LiteDRAM sources
count issued controller-word payload, including masked/unused byte lanes and ALL
controller clients. Neither includes refresh/command signaling bytes on DRAM pins.
Every source registers events before accumulation, keeping counter carry chains
off protocol paths. All sources and the CSR bank must share sys clock/reset.
"""
from functools import reduce
from operator import or_

from migen import Array, Constant, If, Mux, Signal
from litex.gen import LiteXModule
from litex.soc.interconnect.csr import CSRStatus, CSRStorage

METRICS = ('read_bytes', 'write_bytes', 'cmd_stalls', 'data_stalls', 'errors')
AXI_LOGICAL = 1
LITEDRAM_PAYLOAD = 2


def _sum_tree(values):
    values = list(values)
    if not values:
        return Constant(0, 1)
    while len(values) > 1:
        values = [values[i] + values[i + 1] if i + 1 < len(values) else values[i]
                  for i in range(0, len(values), 2)]
    return values[0]


class BonsaiTrafficSource(LiteXModule):
    def __init__(self, kind, index, word_bytes, mask, increments):
        increments = list(increments)
        if not 0 < kind < 256 or not 0 <= mask < 32:
            raise ValueError('traffic kind/mask out of range')
        if not 0 <= index < 256 or not 0 < word_bytes < 65536:
            raise ValueError('traffic descriptor index/word width out of range')
        if len(increments) != len(METRICS):
            raise ValueError('incomplete traffic source')
        self.info = kind | (index << 8) | (word_bytes << 16)
        self.mask = mask
        self.counters = []
        for name, increment in zip(METRICS, increments):
            event = Signal(len(increment), name='traffic_event_' + name)
            count = Signal(64, name='traffic_' + name)
            self.sync += [event.eq(increment), count.eq(count + event)]
            self.counters.append(count)


class BonsaiAXITraffic(BonsaiTrafficSource):
    """Observe one AXI boundary; all bytes delivered on R and active WSTRB bytes.

    cmd_stalls = cycles with either AR or AW backpressured; data_stalls = cycles
    with either R or W backpressured. Simultaneous stalls count once per class.
    errors = accepted non-OKAY R beats plus non-OKAY B responses (can be two/cycle).
    """
    def __init__(self, axi, index=0):
        if getattr(axi, 'clock_domain', 'sys') != 'sys':
            raise ValueError('traffic monitor requires a sys-domain AXI boundary')
        width = axi.data_width // 8
        read = axi.r.valid & axi.r.ready
        write = axi.w.valid & axi.w.ready
        increments = [
            Mux(read, width, 0),
            Mux(write, _sum_tree(axi.w.strb), 0),
            (axi.ar.valid & ~axi.ar.ready) | (axi.aw.valid & ~axi.aw.ready),
            (axi.r.valid & ~axi.r.ready) | (axi.w.valid & ~axi.w.ready),
            (read & (axi.r.resp != 0)) +
            (axi.b.valid & axi.b.ready & (axi.b.resp != 0)),
        ]
        BonsaiTrafficSource.__init__(self, AXI_LOGICAL, index, width, 0x1f, increments)


class BonsaiLiteDRAMTraffic(BonsaiTrafficSource):
    """Controller-issued payload, covering CPU AND accelerator clients.

    Bank rdata_valid/wdata_ready are asserted when a column command is issued;
    the native crossbar adds data latency afterwards. A masked 512-bit write
    still consumes a whole 64-byte DRAM word. cmd_stalls observes upstream bank
    request backpressure, not all bank scheduling/refresh idle time. There is no
    response/error metric at this boundary: mask explicitly excludes both.
    """
    def __init__(self, interface, index=0):
        banks = [getattr(interface, 'bank' + str(i)) for i in range(interface.nbanks)]
        width = interface.data_width // 8
        increments = [
            _sum_tree(b.rdata_valid for b in banks)*width,
            _sum_tree(b.wdata_ready for b in banks)*width,
            reduce(or_, (b.valid & ~b.ready for b in banks)),
            Constant(0, 1), Constant(0, 1),
        ]
        BonsaiTrafficSource.__init__(self, LITEDRAM_PAYLOAD, index, width, 0x07, increments)


class BonsaiTrafficCSR(LiteXModule):
    """One strobe freezes every source on the same edge; selector only reads it.

    Epoch and valid invalidate snapshots crossing/inside an engine soft reset.
    Counters themselves are sys-domain and include reset-drain traffic. Register
    event latency is identical across sources; quiesce for two clocks to include
    the final edge's event. Host/firmware must serialize use of the snapshot bank.
    """
    def __init__(self, sources):
        sources = list(sources)
        if not 1 <= len(sources) <= 255:
            raise ValueError('traffic CSR requires 1..255 sources')
        if len({s.info & 0xffff for s in sources}) != len(sources):
            raise ValueError('duplicate traffic source kind/index')
        self.reset_event = Signal()
        self.reset_busy = Signal()
        self.version = CSRStatus(32, reset=1)
        self.count = CSRStatus(16, reset=len(sources))
        self.snapshot = CSRStorage(1, description='Write 1: freeze all memory counters atomically.')
        self.select = CSRStorage(8, description='Source index in the frozen snapshot.')
        self.valid = CSRStatus(1, description='Snapshot captured outside reset/drain and selector in range.')
        self.epoch = CSRStatus(64, description='Engine reset generation of snapshot.')
        self.cycles = CSRStatus(64, description='Free-running sys cycles at snapshot.')
        self.info = CSRStatus(32, description='kind[7:0], source index[15:8], word bytes[31:16].')
        self.mask = CSRStatus(5, description='Supported metrics: read/write bytes, cmd/data stalls, errors.')
        status = []
        for name in METRICS:
            reg = CSRStatus(64, name=name, description='Frozen ' + name + ' of selected source.')
            setattr(self, name, reg)
            status.append(reg)
        self.live_cycles = Signal(64)
        self.live_epoch = Signal(64)
        self.snapshots = [[Signal(64) for _ in METRICS] for _ in sources]
        captured_valid = Signal()
        selected = self.select.storage
        in_range = selected < len(sources)
        self.sync += [
            self.live_cycles.eq(self.live_cycles + 1),
            If(self.reset_event, self.live_epoch.eq(self.live_epoch + 1)),
            If(self.snapshot.re & self.snapshot.storage,
                captured_valid.eq(~self.reset_busy & ~self.reset_event),
                self.epoch.status.eq(self.live_epoch),
                self.cycles.status.eq(self.live_cycles),
                *[saved.eq(live) for source, snapshot in zip(sources, self.snapshots)
                  for live, saved in zip(source.counters, snapshot)],
            ),
        ]
        self.comb += [
            self.valid.status.eq(captured_valid & in_range),
            self.info.status.eq(Mux(in_range, Array(s.info for s in sources)[selected], 0)),
            self.mask.status.eq(Mux(in_range, Array(s.mask for s in sources)[selected], 0)),
            *[reg.status.eq(Mux(in_range, Array(s[i] for s in self.snapshots)[selected], 0))
              for i, reg in enumerate(status)],
        ]


def add_memory_traffic(soc, axi_ports, dram_interfaces=()):
    """Optional target instrumentation, after add_bonsai_cluster().

    AXI indices identify the supplied accelerator-port order (not physical HBM
    pseudo-channel numbers). DDR indices identify controller order. ztraffic
    sorts after existing CSR banks, so enabling it does not move their addresses.
    HBM AXI alone provides no claim about physical HBM-controller payload.
    """
    axi_ports, dram_interfaces = list(axi_ports), list(dram_interfaces)
    names = ['ztraffic'] + ['traffic_axi' + str(i) for i in range(len(axi_ports))]
    names += ['traffic_ddr' + str(i) for i in range(len(dram_interfaces))]
    if any(hasattr(soc, name) for name in names):
        raise ValueError('memory traffic monitors already installed')
    if not 1 <= len(axi_ports) + len(dram_interfaces) <= 255:
        raise ValueError('memory traffic requires 1..255 sources')
    sources = []
    for label, interfaces, cls in [('axi', axi_ports, BonsaiAXITraffic),
                                    ('ddr', dram_interfaces, BonsaiLiteDRAMTraffic)]:
        for i, interface in enumerate(interfaces):
            source = cls(interface, index=i)
            setattr(soc, 'traffic_' + label + str(i), source)
            sources.append(source)
    soc.ztraffic = BonsaiTrafficCSR(sources)
    soc.comb += [soc.ztraffic.reset_event.eq(soc.engine_csr.rst_trigger),
                 soc.ztraffic.reset_busy.eq(soc.engrst.busy)]
