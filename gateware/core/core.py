#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""TriXiumCore: the accelerator as one LiteX module, shared by every target and the simulation.

The target supplies the memory AXI ports, the sys clock/reset, physical constraints, the CPU
memory map and the upload stream. The core's `eng` domain runs on the sys clock with its own
soft reset (engine reset/drain); ports in another clock domain need a target-side CDC.
"""

from operator import or_
from functools import reduce
from dataclasses import dataclass

from migen import C, ClockDomainsRenamer, Signal

from litex.gen import LiteXModule

from gateware.core.scratchpad import Scratchpad, ScratchpadArbiter
from gateware.core.reset import EngineResetDomain
from gateware.core.vector import VectorUnit
from gateware.core.sequencer import Sequencer
from gateware.core.engine import MatVecEngine
from gateware.core.csr import EngineCSR
from gateware.memory.loader import Loader
from gateware.units.hadamard import HadamardScratchpad
from gateware.units.activation import ActivationScratchpad
from gateware.units.restore import ProjectionRestoreScratchpad
from gateware.units.silu import SiluScratchpad
from gateware.units.norm import NormScratchpad
from gateware.units.conv import ConvScratchpad

# TriXium Config -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class TriXiumConfig:
    with_vecunit: bool             = True
    chan_stride: int               = 0
    chan_mod: int                  = 0
    chan_skew: int                 = 0
    scratchpad_size: int           = 256*1024
    loader_data_width: int         = 128
    with_sp_csr: bool              = True
    with_hadamard: bool            = False
    with_activation_quantize: bool = False
    with_projection_restore: bool  = False
    with_silu: bool                = False
    with_norm: bool                = False
    with_conv: bool                = False
    norm_rope: bool                = True  # Unit 12's RoPE stage (off: CPU RoPE, norm ABI 1.1).
    hadamard_lanes: int            = 32    # Unit 1 butterflies per clock (8, 16, 32).
    silu_ways: int                 = 4     # Unit 11 kernels in lockstep (elements per clock/2).
    scratchpad_xpm: bool           = False # Xilinx: the scratchpad as an explicit XPM block RAM.
    with_dnstream: bool            = False # Engine op 5: the DeltaNet stream unit on every channel.
    dnstream_options: tuple        = ()    # (name, value) overrides of the unit's arguments.
    with_attnstream: bool          = False # Engine op 6: the attention passes (scores, weighted V).
    with_trit5: bool               = False # Ternary codes 5 trits per byte (Trit5Unpacker per channel).
    with_prefetch: bool            = False # Engine op 7: a matvec's weight reads issued ahead.
    reader_fifo_depth: int         = 512   # Beats buffered per channel reader (the prefetch's reach).
    reader_max_bursts: int         = 8     # 16-beat bursts a channel reader keeps outstanding.
    with_axpy: bool                = True  # Engine op 4 (weighted-V accumulate, the Qwen3 attention): the
                                           # pipes' shifters, accumulators and drain; off in hybrid builds.
    with_vec_fused: bool           = False # Vector unit: fused clamp/round instructions (asm.fuse).
    attnstream_options: tuple      = ()    # (name, value) overrides of the unit's arguments.

    def __post_init__(self):
        if (self.with_hadamard or self.with_activation_quantize or self.with_projection_restore
                or self.with_silu or self.with_norm or self.with_conv) and \
                self.scratchpad_size > 256*1024:
            raise ValueError("Resident clients need at most a 256 KiB scratchpad")
        if self.chan_mod < 0 or self.chan_mod & (self.chan_mod - 1):
            raise ValueError("chan_mod must be zero or a power of two")
        # Without chan_mod the STORE/upload paths shift the channel index by the stride; with it
        # they add a constant per window, so any beat-aligned stride works.
        if self.chan_stride < 0 or (not self.chan_mod and self.chan_stride & (self.chan_stride - 1)):
            raise ValueError("chan_stride must be zero or a power of two without chan_mod")
        if self.chan_skew < 0 or (self.chan_skew and not self.chan_mod):
            raise ValueError("nonnegative chan_skew requires chan_mod")
        if self.chan_stride % 32 or self.chan_skew % 32:
            raise ValueError("memory windows must be aligned to a 32-byte beat")
        if self.scratchpad_size < 32 or self.scratchpad_size % 32:
            raise ValueError("scratchpad must contain a whole number of 32-byte beats")
        if self.loader_data_width not in (32, 64, 128, 256):
            raise ValueError("upload stream width must divide 256 bits")

# TriXium Core -------------------------------------------------------------------------------------

class TriXiumCore(LiteXModule):
    """The TriXium accelerator: engine, sequencer, vector unit, scratchpad units and loader.

    Standard interfaces, the whole contract with a target:
    - ``memory_ports``: one 256-bit AXI master per engine channel, in sys (the target does
      any clock-domain crossing and maps each port onto its memory: DDR4, HBM, a model).
    - ``bus``: a 32-bit wishbone slave onto the scratchpad (the CPU's window).
    - ``upload``: a 128-bit stream sink for weight uploads (the host's PCIe DMA).
    - ``constants``: the firmware-visible feature constants, for ``soc.add_constant``.

    The CSRs (engine, sequencer, vector unit, loader, scratchpad peek) are this module's.
    Commands and upload/store use the stride/mod/skew channel placement.
    """
    def __init__(self, memory_ports, config=TriXiumConfig()):
        self.constants = {}
        ports = list(memory_ports)
        if not 1 <= len(ports) <= 31:
            raise ValueError("the core needs 1..31 memory ports")
        if config.chan_mod and len(ports) % config.chan_mod:
            raise ValueError("memory ports must divide evenly over controllers")
        for port in ports:
            if port.data_width != 256 or port.address_width < 32:
                raise ValueError("cluster memory ports require 256-bit AXI, >=32-bit addresses")
            if getattr(port, "clock_domain", "sys") != "sys":
                raise ValueError("memory ports need target-side CDC into sys")

        # # #

        # Scratchpad / Engine reset domain.
        self.scratchpad = Scratchpad(size=config.scratchpad_size, xpm=config.scratchpad_xpm)
        sp_csr = None
        if config.with_sp_csr:
            sp_csr = self.scratchpad.add_csr_port()
            self.sp_csr = sp_csr
        self.engrst = EngineResetDomain(ports)
        axis = self.engrst.masters
        self.add_constant("TRIXIUM_HAS_ENGRESET")

        # Sequencer (the descriptor lists of both paths) and the vector unit. A hybrid-only
        # bitstream drops the vector unit (with_vecunit=False: 34k LUTs on the K7, which the
        # hybrid path never runs - its lists hold no vector descriptor); the sequencer's vector
        # ports then stay idle and the firmware declines K-segmented matvecs and non-hybrid
        # models (TRIXIUM_HAS_VECUNIT absent).
        vecunit = None
        seq = self.seq = ClockDomainsRenamer("eng")(Sequencer(
            with_hadamard            = config.with_hadamard,
            with_activation_quantize = config.with_activation_quantize,
            with_projection_restore  = config.with_projection_restore,
            with_silu                = config.with_silu,
            with_norm                = config.with_norm,
            with_conv                = config.with_conv,
            norm_rope                = config.norm_rope,
            scratchpad_size          = config.scratchpad_size))
        for name in ("TRIXIUM_HAS_SEQ", "TRIXIUM_SEQ_OVERLAP"):
            self.add_constant(name)
        if config.with_vecunit:
            vecunit = self.vecunit = ClockDomainsRenamer("eng")(VectorUnit(fused=config.with_vec_fused))
            self.comb += [
                vecunit.seq_en.eq(seq.busy),
                vecunit.seq_start.eq(seq.vec_start),
                vecunit.seq_pc.eq(seq.vec_pc),
                seq.vec_done.eq(vecunit.done),
                seq.vec_res6.eq(vecunit.res6.status),
                # Descriptor w0[4] selects vec_res6 OR engine absmax. Do not bypass it.
                vecunit.seq_seed7.eq(seq.vec_seed7),
            ]
            self.comb += [vecunit.seq_args[i].eq(seq.vec_args[i]) for i in range(8)]
            for name in ("TRIXIUM_HAS_VECUNIT", "TRIXIUM_HAS_VECUNIT2", "TRIXIUM_VEC_LOOPS2",
                         "TRIXIUM_SEQ_LUT"):
                self.add_constant(name)
            self.add_constant("TRIXIUM_VEC_VREGS", vecunit.vregs)
            if config.with_vec_fused:
                self.add_constant("TRIXIUM_VEC_FUSED") # firmware/Makefile: asm --fused 1.

        # Scratchpad units: scratchpad clients besides the vector unit, in attach order (the
        # last busy one owns the port). One flat ScratchpadArbiter is built after all of them,
        # so the BRAM address/data/enable see log2(clients) levels instead of one serial mux
        # per unit (timing).
        sp_clients = []
        base_busy  = [vecunit.busy] if vecunit is not None else []
        extensions = []
        if config.with_hadamard:
            self.hadamard = ClockDomainsRenamer("eng")(
                HadamardScratchpad(config.scratchpad_size, lanes=config.hadamard_lanes))
            sp_clients.append(self.hadamard)
            extensions.append(("hadamard", self.hadamard))
            self.add_constant("TRIXIUM_HAS_HADAMARD")
            self.add_constant("TRIXIUM_HADAMARD_BLOCK", 1024)
        if config.with_activation_quantize:
            self.activation_quantize = ClockDomainsRenamer("eng")(
                ActivationScratchpad(config.scratchpad_size))
            sp_clients.append(self.activation_quantize)
            extensions.append(("activation_quantize", self.activation_quantize))
            self.add_constant("TRIXIUM_HAS_ACTIVATION_QUANTIZE")
        if config.with_projection_restore:
            self.projection_restore = ClockDomainsRenamer("eng")(
                ProjectionRestoreScratchpad(config.scratchpad_size))
            sp_clients.append(self.projection_restore)
            extensions.append(("projection_restore", self.projection_restore))
            self.comb += [
                self.projection_restore.argmax_clear.eq(seq.restore_argmax_clear),
                seq.restore_argmax_valid_in.eq(self.projection_restore.argmax.best_valid),
                seq.restore_argmax_idx_in.eq(self.projection_restore.argmax.best_idx),
                seq.restore_argmax_val_in.eq(self.projection_restore.argmax.best_val),
            ]
            self.add_constant("TRIXIUM_HAS_PROJECTION_RESTORE")
        if config.with_silu:
            self.silu = ClockDomainsRenamer("eng")(SiluScratchpad(config.scratchpad_size,
                ways=config.silu_ways))
            sp_clients.append(self.silu)
            extensions.append(("silu", self.silu))
            self.comb += seq.silu_unit_fixups.eq(self.silu.fixups)
            self.add_constant("TRIXIUM_HAS_SILU")
        if config.with_norm:
            self.norm = ClockDomainsRenamer("eng")(NormScratchpad(config.scratchpad_size,
                with_rope=config.norm_rope))
            sp_clients.append(self.norm)
            extensions.append(("norm", self.norm))
            self.add_constant("TRIXIUM_HAS_NORM")
        if config.with_conv:
            self.conv = ClockDomainsRenamer("eng")(ConvScratchpad(config.scratchpad_size))
            sp_clients.append(self.conv)
            extensions.append(("conv", self.conv))
            self.add_constant("TRIXIUM_HAS_CONV")

        # Matvec engine.
        placement = dict(chan_stride=config.chan_stride, chan_mod=config.chan_mod,
                         chan_skew=config.chan_skew)
        sp_client = vecunit
        if sp_clients:
            self.sparb = ScratchpadArbiter(vecunit, sp_clients)   # base None without the unit
            sp_client = self.sparb
        self.engine = ClockDomainsRenamer("eng")(MatVecEngine(
            scratchpad = self.scratchpad,
            hbm_axis   = axis,
            vecunit    = sp_client,
            seq        = seq,
            dnstream   = config.with_dnstream   and (config.dnstream_options   or True),
            attnstream = config.with_attnstream and (config.attnstream_options or True),
            trit5      = config.with_trit5,
            prefetch   = config.with_prefetch,
            reader_fifo_depth = config.reader_fifo_depth,
            reader_max_bursts = config.reader_max_bursts,
            axpy       = config.with_axpy,
            **placement))
        if config.with_trit5:
            self.add_constant("TRIXIUM_HAS_TRIT5")
        if config.with_axpy:
            self.add_constant("TRIXIUM_HAS_AXPY")
        if config.with_prefetch:
            self.add_constant("TRIXIUM_HAS_PREFETCH")
            self.add_constant("TRIXIUM_READER_FIFO_DEPTH", config.reader_fifo_depth)
        if config.with_attnstream:
            at = self.engine.attnstream
            self.add_constant("TRIXIUM_HAS_ATTNSTREAM")
            self.add_constant("TRIXIUM_ATTN_LANES",  at.lanes)
            self.add_constant("TRIXIUM_ATTN_GROUPS", at.groups)
            self.add_constant("TRIXIUM_ATTN_HD_MAX", 8*at.hb_max)
            self.add_constant("TRIXIUM_ATTN_T_MAX",  at.t_max)
            self.add_constant("TRIXIUM_ATTN_KV_SLICE") # Record kv0: one command per kv-head slice.
        if config.with_dnstream:
            self.add_constant("TRIXIUM_HAS_DNSTREAM")
            self.add_constant("TRIXIUM_DNSTREAM_LANES", len(axis))
            self.add_constant("TRIXIUM_DNSTREAM_SUBLANES", self.engine.dnstream.sublanes)
            self.add_constant("TRIXIUM_DNSTREAM_ZERO") # Op 5 bias bit 0: state taken as +0.0.
            self.add_constant("TRIXIUM_DNSTREAM_WOFF") # Op 5 writes back at wbase + dstride beats.
        self.engine_csr = EngineCSR(self.engine, seq=seq)

        # Scratchpad unit dispatch: a unit's command is accepted only while it may own the port.
        for name, unit in extensions:
            available = Signal()
            other_idle = ~reduce(or_, base_busy, C(0, 1))
            for other_name, other in extensions:
                if other_name != name:
                    other_idle = other_idle & ~other.busy
            command = getattr(seq, name + "_cmd")
            self.comb += [
                # An engine op in flight blocks a unit unless the sequencer dispatches it with
                # OVERLAP (sequencer.py w0[9]): the per-cycle arbitration shares the ports.
                available.eq((~self.engine.sp_busy | seq.overlap) & other_idle &
                             ~self.scratchpad.sbw_pending & ~self.engine_csr.rst_busy &
                             ~self.engine_csr.rst_trigger),
                command.connect(unit.cmd, omit={"valid", "ready"}),
                unit.cmd.valid.eq(command.valid & available),
                command.ready.eq(unit.cmd.ready & available),
                getattr(seq, name + "_done").eq(unit.done),
                getattr(seq, name + "_failed").eq(unit.error),
                unit.cancel.eq(self.engine_csr.rst_trigger | self.engine_csr.rst_busy),
            ]
        self.comb += [
            self.engrst.trigger.eq(self.engine_csr.rst_trigger),
            self.engine_csr.rst_busy.eq(self.engrst.busy),
            self.engine_csr.rst_stuck.eq(self.engrst.stuck),
        ]

        # Loader.
        self.loader = ClockDomainsRenamer("eng")(Loader(
            hbm_axis    = axis,
            data_width  = config.loader_data_width,
            with_mstore = config.with_dnstream,
            **placement))
        self.comb += [
            self.engine.store_cmd.connect(self.loader.store_cmd),
            self.engine.store_data.connect(self.loader.store_data),
            self.engine.store_busy.eq(self.loader.store_busy),
        ]
        if config.with_dnstream:
            self.comb += self.engine.mstore_cmd.connect(self.loader.mstore_cmd)
            self.comb += [d.connect(w) for d, w in zip(self.engine.mstore_data, self.loader.mstore_data)]
        units_idle = ~self.engine.sp_busy
        if sp_client is not None:
            units_idle = units_idle & ~sp_client.busy
        if seq is not None:
            units_idle = units_idle & ~seq.busy
        self.comb += self.scratchpad.units_idle.eq(units_idle)
        for name in ("TRIXIUM_HAS_KVSTORE", "TRIXIUM_HAS_TOPK"):
            self.add_constant(name)
        self.add_constant("TRIXIUM_SP_SB_SLOT",  1)
        self.add_constant("TRIXIUM_SP_SB_RSLOT", 1)

        # Standard interfaces.
        self.bus    = self.scratchpad.add_wishbone_port()
        self.upload = self.loader.sink

    def add_constant(self, name, value=None):
        """Record a firmware-visible constant (exported by the target with soc.add_constant)."""
        assert name not in self.constants, name
        self.constants[name] = value
