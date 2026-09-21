#
# This file is part of TriXium.
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Reusable accelerator integration for LiteX SoCs and simulation.

The target supplies memory AXI ports, the sys clock/reset, physical constraints,
vendor floating-point implementations, CPU memory map and an optional upload
stream. This module installs the same compute/reset/scratchpad wiring everywhere.
It intentionally installs modules at the SoC root to preserve public CSR bank
names. Moving construction here DOES change internal Migen names; existing routed
images remain rollback artifacts, not proof of timing for the extracted design.

Currently eng uses the sys clock with an independent soft reset. Connecting ports
in another clock domain requires an explicit target-side CDC adapter.
"""
from dataclasses import dataclass

from migen import ClockDomainsRenamer

from trixium.gateware.scratchpad import BonsaiScratchpad
from trixium.gateware.engreset import BonsaiEngineResetDomain
from trixium.gateware.vector import BonsaiVectorUnit
from trixium.gateware.sequencer import BonsaiSequencer
from trixium.gateware.matvec import BonsaiMatVecEngine
from trixium.gateware.csr_front import BonsaiEngineCSR
from trixium.gateware.loader import BonsaiHBMLoader


def native_read_options(regions, reservations, endpoints, *, write=False):
    """Opt-in memory policy for the experimental native engine, not capability bits.

    The target supplies physical facts/reservations once. Grant READ to the
    bounded native path without mutating its conservative public capability
    records. Optionally grant WRITE for the checked region-upload path as well;
    this is still not a complete public region-I/O/runtime capability.
    """
    from dataclasses import replace
    from trixium.capabilities import Access
    access = Access.READ | (Access.WRITE if write else Access(0))
    return dict(native_regions=tuple(replace(r, access=access) for r in regions),
                native_reservations=tuple(reservations), native_endpoints=tuple(endpoints))


@dataclass(frozen=True)
class BonsaiClusterConfig:
    with_vecunit: bool = True
    with_dual: bool = False
    with_dnrec: bool = False
    chan_stride: int = 0
    chan_mod: int = 0
    chan_skew: int = 0
    scratchpad_size: int = 256*1024
    loader_data_width: int = 128
    with_sp_csr: bool = True
    native_regions: tuple = ()
    native_reservations: tuple = ()
    native_endpoints: tuple = ()

    def __post_init__(self):
        for name in ('native_regions', 'native_reservations', 'native_endpoints'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.native_regions and (self.native_reservations or self.native_endpoints):
            raise ValueError('native reservations/endpoints require a native region table')
        if self.with_dnrec and not self.with_vecunit:
            raise ValueError("DeltaNet recurrence needs the vector scratchpad arbiter")
        for name in ("chan_stride", "chan_mod"):
            value = getattr(self, name)
            if value < 0 or value & (value - 1):
                raise ValueError(name + " must be zero or a power of two")
        if self.chan_skew < 0 or (self.chan_skew and not self.chan_mod):
            raise ValueError("nonnegative chan_skew requires chan_mod")
        if self.chan_stride % 32 or self.chan_skew % 32:
            raise ValueError("memory windows must be aligned to a 32-byte beat")
        if self.scratchpad_size < 32 or self.scratchpad_size % 32:
            raise ValueError("scratchpad must contain a whole number of 32-byte beats")
        if self.loader_data_width not in (32, 64, 128, 256):
            raise ValueError("upload stream width must divide 256 bits")


def add_bonsai_cluster(soc, memory_ports, config=BonsaiClusterConfig()):
    """Install one cluster on ``soc``; leave memory adapters/SoC bus/DMA to target.

    All memory ports are 256-bit AXI in sys. Legacy commands and upload/store
    retain stride/mod/skew placement. Optional native projections use explicit
    region/offset/stride descriptors on the same reader and MAC per pipe; optional
    checked region uploads share the existing writers. Sequencer lowering for
    that layout is not yet implemented. Ports are
    storage endpoints, not an assertion about board capacity.
    """
    ports = list(memory_ports)
    if not 1 <= len(ports) <= 31:
        raise ValueError("legacy cluster requires 1..31 memory ports")
    if config.chan_mod and len(ports) % config.chan_mod:
        raise ValueError("memory ports must divide evenly over controllers")
    if config.native_regions and len(config.native_endpoints) != len(ports):
        raise ValueError('native endpoint mapping must name every compute port')
    for port in ports:
        if port.data_width != 256 or port.address_width < 32:
            raise ValueError("cluster memory ports require 256-bit AXI, >=32-bit addresses")
        if getattr(port, "clock_domain", "sys") != "sys":
            raise ValueError("memory ports need target-side CDC into sys")
    for name in ("scratchpad", "engrst", "engine", "engine_csr", "loader", "vecunit", "seq", "dnunit", "zznative", "zzregion"):
        if hasattr(soc, name):
            raise ValueError("cluster would overwrite existing SoC module: " + name)

    soc.scratchpad = BonsaiScratchpad(size=config.scratchpad_size)
    soc.engrst = BonsaiEngineResetDomain(ports)
    axis = soc.engrst.masters
    soc.add_constant("BONSAI_HAS_ENGRESET")

    vecunit = seq = None
    if config.with_vecunit:
        vecunit = soc.vecunit = ClockDomainsRenamer("eng")(BonsaiVectorUnit())
        seq = soc.seq = ClockDomainsRenamer("eng")(BonsaiSequencer())
        soc.comb += [
            vecunit.seq_en.eq(seq.busy),
            vecunit.seq_start.eq(seq.vec_start),
            vecunit.seq_pc.eq(seq.vec_pc),
            seq.vec_done.eq(vecunit.done),
            seq.vec_res6.eq(vecunit.res6.status),
            # Descriptor w0[4] selects vec_res6 OR engine absmax. Do not bypass it.
            vecunit.seq_seed7.eq(seq.vec_seed7),
        ]
        soc.comb += [vecunit.seq_args[i].eq(seq.vec_args[i]) for i in range(8)]
        for name in ("BONSAI_HAS_VECUNIT", "BONSAI_HAS_VECUNIT2", "BONSAI_VEC_LOOPS2",
                     "BONSAI_SEQ_LUT", "BONSAI_HAS_SEQ"):
            soc.add_constant(name)
        soc.add_constant("BONSAI_VEC_VREGS", vecunit.vregs)

    sp_client = vecunit
    if config.with_dnrec:
        from trixium.gateware.deltanet import BonsaiDeltaNetUnit, BonsaiScratchpadClientMux
        soc.dnunit = ClockDomainsRenamer("eng")(BonsaiDeltaNetUnit())
        soc.spmux = BonsaiScratchpadClientMux(vecunit, soc.dnunit)
        sp_client = soc.spmux
        soc.add_constant("BONSAI_HAS_DNREC")

    placement = dict(chan_stride=config.chan_stride, chan_mod=config.chan_mod,
                     chan_skew=config.chan_skew)
    native = (dict(native_regions=config.native_regions, native_reservations=config.native_reservations,
                   native_endpoints=config.native_endpoints) if config.native_regions else {})
    soc.engine = ClockDomainsRenamer("eng")(
        BonsaiMatVecEngine(soc.scratchpad, axis, vecunit=sp_client,
                          seq=seq, dual=config.with_dual, **placement, **native))
    soc.engine_csr = BonsaiEngineCSR(soc.engine, seq=seq)
    if config.native_regions:
        from trixium.gateware.native_csr import BonsaiNativeCSR
        soc.zznative = BonsaiNativeCSR(soc.engine, soc.engine_csr)
        soc.add_constant('BONSAI_HAS_NATIVE_ENGINE')
    soc.comb += [
        soc.engrst.trigger.eq(soc.engine_csr.rst_trigger),
        soc.engine_csr.rst_busy.eq(soc.engrst.busy),
        soc.engine_csr.rst_stuck.eq(soc.engrst.stuck),
    ]
    if config.with_dual:
        soc.add_constant("BONSAI_HAS_DUALQ")

    from trixium.capabilities import Access
    region_upload = (dict(regions=config.native_regions, reservations=config.native_reservations,
                          endpoints=config.native_endpoints)
                     if any(r.access & Access.WRITE for r in config.native_regions) else {})
    soc.loader = ClockDomainsRenamer("eng")(
        BonsaiHBMLoader(axis, data_width=config.loader_data_width, **placement, **region_upload))
    if region_upload:
        from trixium.gateware.region_csr import BonsaiRegionUploadCSR
        soc.zzregion = BonsaiRegionUploadCSR(soc.loader, soc.engine_csr)
        soc.add_constant('BONSAI_HAS_REGION_UPLOAD')
    soc.comb += [
        soc.engine.store_cmd.connect(soc.loader.store_cmd),
        soc.engine.store_data.connect(soc.loader.store_data),
        soc.engine.store_busy.eq(soc.loader.store_busy),
    ]
    units_idle = ~soc.engine.sp_busy
    if sp_client is not None:
        units_idle = units_idle & ~sp_client.busy
    if seq is not None:
        units_idle = units_idle & ~seq.busy
    soc.comb += soc.scratchpad.units_idle.eq(units_idle)
    for name in ("BONSAI_HAS_KVSTORE", "BONSAI_HAS_TOPK"):
        soc.add_constant(name)
    soc.add_constant("BONSAI_SP_SB_SLOT", 1)
    soc.add_constant("BONSAI_SP_SB_RSLOT", 1)
    if config.with_sp_csr:
        soc.sp_csr = soc.scratchpad.add_csr_port()
