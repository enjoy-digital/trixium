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
from functools import reduce
from operator import or_

from migen import C, ClockDomainsRenamer

from trixium.core.scratchpad import BonsaiScratchpad, BonsaiScratchpadArbiter
from trixium.core.reset import BonsaiEngineResetDomain
from trixium.core.vector import BonsaiVectorUnit
from trixium.core.sequencer import BonsaiSequencer
from trixium.core.engine import BonsaiMatVecEngine
from trixium.core.csr import BonsaiEngineCSR
from trixium.memory.loader import BonsaiHBMLoader


@dataclass(frozen=True)
class BonsaiClusterConfig:
    with_vecunit: bool = True
    chan_stride: int = 0
    chan_mod: int = 0
    chan_skew: int = 0
    scratchpad_size: int = 256*1024
    loader_data_width: int = 128
    with_sp_csr: bool = True
    append_sp_csr: bool = False  # New peek bank after existing banks (targets that had no sp_csr).
    with_hadamard: bool = False
    with_activation_quantize: bool = False
    with_projection_restore: bool = False
    with_silu: bool = False
    with_norm: bool = False
    with_conv: bool = False
    scratchpad_xpm: bool = False # Xilinx: the scratchpad as an explicit XPM block RAM.
    with_dnstream: bool = False  # Engine op 5: the DeltaNet stream unit on every channel.
    dnstream_options: tuple = () # (name, value) overrides of the unit's arguments (tests).
    with_attnstream: bool = False  # Engine op 6: the attention passes (scores, weighted V).
    attnstream_options: tuple = () # (name, value) overrides of the unit's arguments.

    def __post_init__(self):
        if (self.with_hadamard or self.with_activation_quantize or self.with_projection_restore or self.with_silu or self.with_norm or self.with_conv) and (not self.with_vecunit or self.scratchpad_size > 256*1024):
            raise ValueError("Resident clients need the vector/sequencer arbiter and at most 256 KiB scratchpad")
        if self.append_sp_csr and not self.with_sp_csr:
            raise ValueError('append_sp_csr requires a scratchpad CSR port')
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

    All memory ports are 256-bit AXI in sys. Commands and upload/store use the
    stride/mod/skew placement. Ports are storage endpoints, not an assertion about
    board capacity.
    """
    ports = list(memory_ports)
    if not 1 <= len(ports) <= 31:
        raise ValueError("legacy cluster requires 1..31 memory ports")
    if config.chan_mod and len(ports) % config.chan_mod:
        raise ValueError("memory ports must divide evenly over controllers")
    for port in ports:
        if port.data_width != 256 or port.address_width < 32:
            raise ValueError("cluster memory ports require 256-bit AXI, >=32-bit addresses")
        if getattr(port, "clock_domain", "sys") != "sys":
            raise ValueError("memory ports need target-side CDC into sys")
    for name in ("scratchpad", "sp_csr", "zzxsp", "engrst", "engine", "engine_csr", "loader", "vecunit", "seq", "hadamard", "activation_quantize", "projection_restore", "silu", "norm", "conv", "sparb"):
        if hasattr(soc, name):
            raise ValueError("cluster would overwrite existing SoC module: " + name)

    soc.scratchpad = BonsaiScratchpad(size=config.scratchpad_size, xpm=config.scratchpad_xpm)
    sp_csr = None
    if config.with_sp_csr:
        sp_csr = soc.scratchpad.add_csr_port()
        setattr(soc, 'zzxsp' if config.append_sp_csr else 'sp_csr', sp_csr)
    soc.engrst = BonsaiEngineResetDomain(ports)
    axis = soc.engrst.masters
    soc.add_constant("BONSAI_HAS_ENGRESET")

    vecunit = seq = None
    if config.with_vecunit:
        vecunit = soc.vecunit = ClockDomainsRenamer("eng")(BonsaiVectorUnit())
        seq = soc.seq = ClockDomainsRenamer("eng")(BonsaiSequencer(
            with_hadamard=config.with_hadamard, with_activation_quantize=config.with_activation_quantize,
            with_projection_restore=config.with_projection_restore, with_silu=config.with_silu,
            with_norm=config.with_norm, with_conv=config.with_conv,
            scratchpad_size=config.scratchpad_size))
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
                     "BONSAI_SEQ_LUT", "BONSAI_HAS_SEQ", "BONSAI_SEQ_OVERLAP"):
            soc.add_constant(name)
        soc.add_constant("BONSAI_VEC_VREGS", vecunit.vregs)

    # Scratchpad clients besides the vector unit, in attach order: the LAST busy one owns
    # the port (the priority the former nested BonsaiScratchpadClientMux chain gave). One
    # flat BonsaiScratchpadArbiter is built after all of them, so the BRAM address/data/
    # enable see log2(clients) levels instead of one serial mux per unit (v11 failed setup
    # on that chain with seven clients; the arbiter is proven equal to it).
    sp_clients = []
    base_busy = [vecunit.busy] if vecunit is not None else []
    extensions = []
    if config.with_hadamard:
        from trixium.units.hadamard import BonsaiHadamardScratchpad
        soc.hadamard = ClockDomainsRenamer("eng")(BonsaiHadamardScratchpad(config.scratchpad_size))
        sp_clients.append(soc.hadamard)
        extensions.append(("hadamard", soc.hadamard))
        soc.add_constant("BONSAI_HAS_HADAMARD")
        soc.add_constant("BONSAI_HADAMARD_BLOCK", 1024)
    if config.with_activation_quantize:
        from trixium.units.activation import BonsaiActivationScratchpad
        soc.activation_quantize = ClockDomainsRenamer("eng")(BonsaiActivationScratchpad(config.scratchpad_size))
        sp_clients.append(soc.activation_quantize)
        extensions.append(("activation_quantize", soc.activation_quantize))
        soc.add_constant("BONSAI_HAS_ACTIVATION_QUANTIZE")
    if config.with_projection_restore:
        from trixium.units.restore import BonsaiProjectionRestoreScratchpad
        soc.projection_restore = ClockDomainsRenamer("eng")(BonsaiProjectionRestoreScratchpad(config.scratchpad_size))
        sp_clients.append(soc.projection_restore)
        extensions.append(("projection_restore", soc.projection_restore))
        soc.comb += [
            soc.projection_restore.argmax_clear.eq(seq.restore_argmax_clear),
            seq.restore_argmax_valid_in.eq(soc.projection_restore.argmax.best_valid),
            seq.restore_argmax_idx_in.eq(soc.projection_restore.argmax.best_idx),
            seq.restore_argmax_val_in.eq(soc.projection_restore.argmax.best_val),
        ]
        soc.add_constant("BONSAI_HAS_PROJECTION_RESTORE")
    if config.with_silu:
        from trixium.units.silu import BonsaiSiluScratchpad
        soc.silu = ClockDomainsRenamer("eng")(BonsaiSiluScratchpad(config.scratchpad_size))
        sp_clients.append(soc.silu)
        extensions.append(("silu", soc.silu))
        soc.comb += seq.silu_unit_fixups.eq(soc.silu.fixups)
        soc.add_constant("BONSAI_HAS_SILU")
    if config.with_norm:
        from trixium.units.norm import BonsaiNormScratchpad
        soc.norm = ClockDomainsRenamer("eng")(BonsaiNormScratchpad(config.scratchpad_size))
        sp_clients.append(soc.norm)
        extensions.append(("norm", soc.norm))
        soc.add_constant("BONSAI_HAS_NORM")
    if config.with_conv:
        from trixium.units.conv import BonsaiConvScratchpad
        soc.conv = ClockDomainsRenamer("eng")(BonsaiConvScratchpad(config.scratchpad_size))
        sp_clients.append(soc.conv)
        extensions.append(("conv", soc.conv))
        soc.add_constant("BONSAI_HAS_CONV")

    placement = dict(chan_stride=config.chan_stride, chan_mod=config.chan_mod,
                     chan_skew=config.chan_skew)
    sp_client = vecunit
    if sp_clients:
        if vecunit is None:  # The chain had the same requirement (it wrapped the vector unit).
            raise ValueError("scratchpad units need the vector unit as the arbiter's base client")
        soc.sparb = BonsaiScratchpadArbiter(vecunit, sp_clients)
        sp_client = soc.sparb
    soc.engine = ClockDomainsRenamer("eng")(
        BonsaiMatVecEngine(soc.scratchpad, axis, vecunit=sp_client,
                          seq=seq, **placement,
                          dnstream=config.with_dnstream and (config.dnstream_options or True),
                          attnstream=config.with_attnstream and (config.attnstream_options or True)))
    if config.with_attnstream:
        at = soc.engine.attnstream
        soc.add_constant("BONSAI_HAS_ATTNSTREAM")
        soc.add_constant("BONSAI_ATTN_LANES", at.lanes)
        soc.add_constant("BONSAI_ATTN_GROUPS", at.groups)
        soc.add_constant("BONSAI_ATTN_HD_MAX", 8*at.hb_max)
        soc.add_constant("BONSAI_ATTN_T_MAX", at.t_max)
    if config.with_dnstream:
        soc.add_constant("BONSAI_HAS_DNSTREAM")
        soc.add_constant("BONSAI_DNSTREAM_LANES", len(axis))
        soc.add_constant("BONSAI_DNSTREAM_ZERO")     # op 5 bias bit 0: state taken as +0.0
    soc.engine_csr = BonsaiEngineCSR(soc.engine, seq=seq)
    for name, unit in extensions:
        from migen import Signal
        available = Signal()
        other_idle = ~reduce(or_, base_busy, C(0, 1))
        for other_name, other in extensions:
            if other_name != name:
                other_idle = other_idle & ~other.busy
        command = getattr(seq, name + "_cmd")
        soc.comb += [
            # An engine op in flight blocks a unit unless the sequencer dispatches it with
            # OVERLAP (sequencer.py w0[9]): the per-cycle arbitration shares the ports.
            available.eq((~soc.engine.sp_busy | seq.overlap) & other_idle &
                         ~soc.scratchpad.sbw_pending & ~soc.engine_csr.rst_busy &
                         ~soc.engine_csr.rst_trigger),
            command.connect(unit.cmd, omit={"valid", "ready"}),
            unit.cmd.valid.eq(command.valid & available), command.ready.eq(unit.cmd.ready & available),
            getattr(seq, name + "_done").eq(unit.done), getattr(seq, name + "_failed").eq(unit.error),
            unit.cancel.eq(soc.engine_csr.rst_trigger | soc.engine_csr.rst_busy),
        ]
    soc.comb += [
        soc.engrst.trigger.eq(soc.engine_csr.rst_trigger),
        soc.engine_csr.rst_busy.eq(soc.engrst.busy),
        soc.engine_csr.rst_stuck.eq(soc.engrst.stuck),
    ]

    soc.loader = ClockDomainsRenamer("eng")(
        BonsaiHBMLoader(axis, data_width=config.loader_data_width, **placement,
                        with_mstore=config.with_dnstream))
    soc.comb += [
        soc.engine.store_cmd.connect(soc.loader.store_cmd),
        soc.engine.store_data.connect(soc.loader.store_data),
        soc.engine.store_busy.eq(soc.loader.store_busy),
    ]
    if config.with_dnstream:
        soc.comb += soc.engine.mstore_cmd.connect(soc.loader.mstore_cmd)
        soc.comb += [d.connect(w) for d, w in zip(soc.engine.mstore_data, soc.loader.mstore_data)]
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
