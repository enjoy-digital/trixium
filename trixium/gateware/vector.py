#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiVectorUnit: microcoded 16-lane SIMD engine executing trixium.model.asm binaries.

Bit-exact contract: the asm.py INTERPRETER is the golden model - this RTL must produce
identical scratchpad contents and scalar results for the same binary and arguments.

Implementation (V1, 2026-08-29): a single-issue, in-order, ONE-INSTRUCTION-PER-CYCLE
pipeline. The previous unit was a non-overlapped FSM (EX0/EX1/EX1B/EX2, 4 cycles per
instruction) and the activation ops were ~48% of the token on the 1.7B; the pipeline keeps
the ISA, the microcode and every port, and is ~3.4x on the same programs.

  F  : pc -> imem (distributed LUTRAM, async read) -> insn_d. The one-entry loop buffer
       (loop_first) feeds D on a taken back-edge, so loops cost 0 bubbles.
  D  : decode, regfile reads (V[va], V[vb] with D-side forwarding; S[sa], S[sd]),
       scoreboard; ALL scalar effects commit at the end of D: MOVSI/ARG/SADDI/SMOV/SADD/
       SSHLV/SNORME, the load/store address post-increment, LOOPS/LOOPE, DONE, bad_op.
  E1 : operand forward from the instruction in E2 (2:1 x 512), lane ALU + 32x32 DSP
       multiply, one-hot writeback select -> ex1_r; reduction trees stage 1; LUT ROM
       address launch; loads drive the scratchpad read address (sp_rd_req, ONE cycle);
       stores pack their data.
  E2 : wb2 = load data / multiply / ex1_r -> V (write port 2); reduction stage 2; LUT
       ROM read + operand select (e_a, e_b - e_a, frac); stores drive the write port and
       HOLD here while the integrator withholds sp_wr_gnt (F/D/E1 freeze, E3/E4 drain).
  E3 : LUT interpolation multiply-add -> lut_r; reductions read S[sd] late and write it.
  E4 : lut_r -> V (write port 3).
  FINISH: after DONE has drained E1..E4, latch res5/6/7 and pulse done.

Latencies (producer's D to the earliest dependent D): ALU/multiply 1, load 2, LUT 4,
scalar 1, reductions into S: back-to-back reductions to the same S need no stall (the late
E3 read sees the previous E3 write); any other touch of a pending S stalls. Forwarding is
youngest-first: E1 operand mux (ex1_r/mul of the instruction in E2) > D-side wb2 (E2) >
D-side wb3 (E3 LUT) > regfile. Every architectural effect commits exactly once.

Standalone the cycle count is deterministic (sp_wr_gnt resets to 1), so tools/bench_vecunit
stays the hardware cycle contract; under concurrent engine traffic the store hold adds
cycles but never changes numerics (identical data is re-driven).

Scratchpad access goes through the integrator's per-cycle arbitration (docs/overlap-design.md):
sp_rd (256-bit read, 1-cycle latency, claimed for exactly the load's E1 cycle) and sp_wr
(256-bit write, 32-bit lane enables, claimed for the store's E2 cycle)."""

from functools import reduce
from operator import or_, and_

from migen import *

from litex.gen import *

from litex.soc.interconnect.csr import *

from trixium.model import asm as vasm
from trixium.model import bitexact as bx

LANES = 16

# BonsaiVectorUnit ---------------------------------------------------------------------------------

class BonsaiVectorUnit(LiteXModule):
    def __init__(self, imem_depth=None, vregs=16):
        # Kept in step with model.asm.IMEM_DEPTH, which asserts the programs fit: an
        # overflowing upload wraps and silently overwrites the program at offset 0.
        if imem_depth is None:
            from trixium.model.asm import IMEM_DEPTH
            imem_depth = IMEM_DEPTH
        # Scratchpad ports (muxed onto the engine's fill/gather ports by the integrator).
        self.sp_rd_adr = Signal(13)  # 256-bit beat index.
        self.sp_rd_dat = Signal(256) # Valid 1 cycle after adr.
        self.sp_wr_adr = Signal(13)
        self.sp_wr_we  = Signal(8)   # 32-bit lane enables.
        self.sp_wr_dat = Signal(256)
        # Per-cycle port claims (docs/overlap-design.md). rd_req is the load's E1 cycle
        # (address on the port; dat_r next cycle reflects that address whoever owns the
        # port then). wr_req is the store's E2 cycle; a withheld wr_gnt HOLDS the store in
        # E2 with identical data re-driven, so numerics cannot change. wr_gnt resets to 1:
        # unwired DUTs (unit tests, the vecunit bench) are unchanged.
        self.sp_rd_req = Signal()
        self.sp_wr_req = Signal()
        self.sp_wr_gnt = Signal(reset=1)

        self.busy = Signal()
        self.done = Signal() # Pulse at program completion (sequencer handshake).

        # Direct-drive port (descriptor sequencer): overrides the CSR path while seq_en.
        self.seq_en    = Signal()
        self.seq_start = Signal()
        self.seq_pc    = Signal(16)
        self.seq_args  = [Signal(32, name=f"vseq_arg{i}") for i in range(8)]
        self.seq_seed7 = Signal(48) # S7 initial value under seq_en (prev prog's res6).

        # CSRs.
        self.prog      = CSRStorage(32, description="Instruction upload (auto-incrementing address).")
        self.prog_adr  = CSRStorage(16, description="Upload address (write before a burst).")
        self.pc        = CSRStorage(16, description="Program start PC.")
        self.arg0 = CSRStorage(32); self.arg1 = CSRStorage(32)
        self.arg2 = CSRStorage(32); self.arg3 = CSRStorage(32)
        self.arg4 = CSRStorage(32); self.arg5 = CSRStorage(32)
        self.arg6 = CSRStorage(32); self.arg7 = CSRStorage(32)
        self.submit    = CSRStorage(1, description="Write 1 to run the program at pc.")
        self.status    = CSRStatus(fields=[
            CSRField("busy",   size=1),
            CSRField("bad_op", size=1, description="Sticky: unknown opcode executed."),
        ])
        self.done_count = CSRStatus(32)
        self.res5 = CSRStatus(48, description="S5 after DONE.")
        self.res6 = CSRStatus(48, description="S6 after DONE.")
        self.res7 = CSRStatus(48, description="S7 after DONE.")

        # # #

        OP = vasm.OP

        def is_in(sig, names):
            return reduce(or_, [sig == OP[n] for n in names])

        # Instruction memory (CSR-uploaded). Async read: F is pc -> LUTRAM -> insn_d, so
        # decode launches from a flop (the integrator constrains it to distributed RAM).
        imem = Memory(32, imem_depth)
        self.specials += imem
        imem_w = imem.get_port(write_capable=True)
        imem_r = imem.get_port(async_read=True)
        self.specials += imem_w, imem_r
        self.autocsr_exclude = set() # (Memories here are fine: not named .mem.)

        up_adr = Signal(16)
        self.sync += [
            If(self.prog_adr.re,
                up_adr.eq(self.prog_adr.storage),
            ).Elif(self.prog.re,
                up_adr.eq(up_adr + 1),
            )
        ]
        self.comb += [
            imem_w.adr.eq(up_adr),
            imem_w.dat_w.eq(self.prog.storage),
            imem_w.we.eq(self.prog.re),
        ]

        # LUT ROMs: {entry[i], entry[i+1]} pairs, replicated per lane for 1-cycle interp.
        # Synchronous read: address launched in E1, data during E2.
        def make_rom(table, edge):
            vals = [int(v) for v in table] + [int(edge)]
            init = [(vals[i] | (vals[i+1] << 17)) for i in range(256)]
            ports = []
            for l in range(LANES):
                rom = Memory(34, 256, init=init)
                self.specials += rom
                p = rom.get_port()
                self.specials += p
                ports.append(p)
            return ports

        rom_e2  = make_rom(bx.exp2_lut(),  1 << 16)
        rom_rec = make_rom(bx.recip_lut(), 1 << 14)
        rom_rsq = make_rom(bx.rsqrt_lut(), int(round((2**15)/2.0)))

        # Register files.
        # ISA rev 2 (2026-09-03): 16 vector registers so the single-chain hot programs can
        # be unrolled x2; the encoding moves with it (model.asm docstring). vregs=8 keeps
        # the original encoding for reference builds.
        assert vregs in (8, 16), vregs
        self.vregs = vregs
        VBITS = 4 if vregs == 16 else 3
        V = Array([Signal(32*LANES, name=f"vr{i}") for i in range(vregs)])
        S = Array([Signal((48, True), name=f"sr{i}") for i in range(8)])

        def lane(sig, i):
            return sig[32*i:32*(i+1)]

        # Op classes ------------------------------------------------------------------------
        ALU_OPS = ["VADD", "VSUB", "VMIN", "VMAX", "VSHRI", "VSHLI", "VSHLV", "VRNE",
                   "VCLAMP8", "VCLAMP16", "VABS", "VCLZN", "VPACK", "BCAST", "VSHLS"]
        MUL_OPS = ["VMUL", "VMULH15", "VMULS"]
        LD_OPS  = ["LDV", "LDW"]
        ST_OPS  = ["STV", "STB", "STW"]
        LUT_OPS = ["LUTE2", "LUTRECIP", "LUTRSQRT"]
        RED_OPS = ["REDSUM", "REDSSQ", "REDMAX", "REDSMAX"]
        SCW_OPS = ["MOVSI", "ARG", "SADDI", "SMOV", "SADD", "SSHLV", "SNORME"] # S[sd] writers
        VEC_WB  = ALU_OPS + MUL_OPS + LD_OPS + LUT_OPS
        USES_VA = ["VADD", "VSUB", "VMUL", "VMULH15", "VMIN", "VMAX", "VSHRI", "VSHLI",
                   "VSHLV", "VRNE", "VCLAMP8", "VCLAMP16", "VABS", "VCLZN", "VPACK",
                   "LUTE2", "LUTRECIP", "LUTRSQRT", "VSHLS", "VMULS",
                   "STV", "STB", "STW", "REDSUM", "REDSSQ", "REDMAX", "REDSMAX"]
        USES_VB = ["VADD", "VSUB", "VMUL", "VMULH15", "VMIN", "VMAX", "VSHLV", "VPACK"]
        USES_SA = ["SMOV", "SADD", "SSHLV", "SNORME", "LOOPS", "LOOPS2", "LDV", "LDW",
                   "STV", "STB", "STW", "VSHLS", "VMULS", "BCAST"]
        USES_SD = SCW_OPS # Non-reduction readers/writers of S[sd] (reductions read late).

        # F / D state -----------------------------------------------------------------------
        pc         = Signal(16)     # Next word to fetch.
        fetch_en   = Signal()
        insn_d     = Signal(32)
        d_pc       = Signal(16)
        d_valid    = Signal()
        use_lf     = Signal()       # D consumes loop_first instead of insn_d.
        loop_first = Signal(32)
        loop_start = Signal(16)
        loop_cnt   = Signal(32)
        lf_arm     = Signal()       # The word after a LOOPS is the loop's first instruction.
        # Second (outer) loop level: independent registers, same mechanism.
        use_lf2     = Signal()
        loop_first2 = Signal(32)
        loop_start2 = Signal(16)
        loop_cnt2   = Signal(32)
        lf2_arm     = Signal()
        running    = Signal()
        bad_op     = Signal()
        finish     = Signal()

        self.comb += imem_r.adr.eq(pc)

        d_word = Signal(32)
        self.comb += d_word.eq(Mux(use_lf, loop_first, Mux(use_lf2, loop_first2, insn_d)))
        d_op  = Signal(6)
        d_vd  = Signal(VBITS)
        d_va  = Signal(VBITS)
        d_vb  = Signal(VBITS)
        d_sd  = Signal(3)
        d_sa  = Signal(3)
        d_imm = Signal((11, True))
        self.comb += d_op.eq(d_word[26:32])
        if vregs == 8:
            self.comb += [
                d_vd.eq(d_word[23:26]),
                d_va.eq(d_word[20:23]),
                d_vb.eq(d_word[17:20]),
                d_sd.eq(d_word[14:17]),
                d_sa.eq(d_word[11:14]),
                d_imm.eq(d_word[0:11]),
            ]
        else:
            # Scalar-only IMM11 ops carry their immediate where the vreg fields sit.
            d_imm11 = Signal()
            self.comb += [
                d_imm11.eq(is_in(d_op, list(vasm.IMM11_OPS))),
                d_vd.eq(d_word[22:26]),
                d_va.eq(d_word[18:22]),
                d_vb.eq(d_word[14:18]),
                d_sd.eq(d_word[11:14]),
                d_sa.eq(d_word[8:11]),
                d_imm.eq(Mux(d_imm11, d_word[15:26],
                             Cat(d_word[0:8], Replicate(d_word[7], 3)))),
            ]
        d_is_load  = Signal()
        d_is_store = Signal()
        d_is_lut   = Signal()
        d_is_red   = Signal()
        d_wr_v     = Signal()
        d_uses_va  = Signal()
        d_uses_vb  = Signal()
        d_uses_sa  = Signal()
        d_uses_sd  = Signal()
        d_is_known = Signal()
        self.comb += [
            d_is_load.eq(is_in(d_op, LD_OPS)),
            d_is_store.eq(is_in(d_op, ST_OPS)),
            d_is_lut.eq(is_in(d_op, LUT_OPS)),
            d_is_red.eq(is_in(d_op, RED_OPS)),
            d_wr_v.eq(is_in(d_op, VEC_WB)),
            d_uses_va.eq(is_in(d_op, USES_VA)),
            d_uses_vb.eq(is_in(d_op, USES_VB)),
            d_uses_sa.eq(is_in(d_op, USES_SA)),
            d_uses_sd.eq(is_in(d_op, USES_SD)),
            d_is_known.eq(is_in(d_op, vasm.OPCODES)),
        ]

        # D-stage scalar reads and scalar-op results.
        sa_d = Signal((48, True))
        sd_d = Signal((48, True))
        self.comb += [
            sa_d.eq(S[d_sa]),
            sd_d.eq(S[d_sd]),
        ]
        # Args (32-bit CSRs, sign-extended into the 48-bit scalar file by ARG). The
        # sequencer's direct-drive values replace the CSR storage while seq_en is high.
        csr_args = [self.arg0.storage, self.arg1.storage, self.arg2.storage,
                    self.arg3.storage, self.arg4.storage, self.arg5.storage,
                    self.arg6.storage, self.arg7.storage]
        arg_eff = [Signal(32, name=f"arg_eff{i}") for i in range(8)]
        for i in range(8):
            self.comb += arg_eff[i].eq(Mux(self.seq_en, self.seq_args[i], csr_args[i]))
        args_arr = Array(arg_eff)
        arg_sel = Signal(32)
        self.comb += arg_sel.eq(args_arr[d_imm[0:3]])

        # Scalar SSHLV: 48-bit variable shift of S[sd] by S[sa] (clamped +-47), wraps to 48.
        ssh_amt = Signal((7, True))
        c47n = Signal((48, True), reset=-47)
        self.comb += ssh_amt.eq(Mux(sa_d > 47, 47, Mux(sa_d < c47n, c47n, sa_d[0:7])))
        ssh_amt_pos = Signal(6)
        ssh_amt_neg = Signal(6)
        self.comb += [
            ssh_amt_pos.eq(Mux(ssh_amt >= 0, ssh_amt[0:6], 0)),
            ssh_amt_neg.eq(Mux(ssh_amt < 0, (-ssh_amt)[0:6], 0)),
        ]
        ssh_full  = Signal((48 + 64, True)) # Full-width left result, then truncate.
        ssh_right = Signal((48, True))
        sshlv_res = Signal((48, True))
        self.comb += [
            ssh_full.eq(sd_d << ssh_amt_pos),
            ssh_right.eq(sd_d >> ssh_amt_neg),
            sshlv_res.eq(Mux(ssh_amt >= 0, ssh_full[0:48], ssh_right)),
        ]
        # Scalar SNORME: even shift landing S[sa] in [2^30, 2^32); 0 for S[sa] <= 0.
        snorme_res = Signal((48, True))
        sn_cases = If(sa_d <= 0, snorme_res.eq(0))
        for p in range(46, -1, -1):
            sn_cases = sn_cases.Elif(sa_d[p],
                snorme_res.eq((30 - p) if (p % 2 == 0) else (31 - p)))
        self.comb += sn_cases

        # E1 record ---------------------------------------------------------------------------
        e1_valid    = Signal()
        e1_op       = Signal(6)
        e1_vd       = Signal(VBITS)
        e1_va       = Signal(VBITS)
        e1_vb       = Signal(VBITS)
        e1_sd       = Signal(3)
        e1_imm      = Signal((11, True))
        e1_va_r     = Signal(32*LANES)
        e1_vb_r     = Signal(32*LANES)
        e1_sa_r     = Signal((48, True))
        e1_is_load  = Signal()
        e1_is_store = Signal()
        e1_is_lut   = Signal()
        e1_is_red   = Signal()
        e1_wr_v     = Signal()
        e1_f = {n: Signal(name=f"e1_f_{n.lower()}") for n in VEC_WB + ST_OPS + RED_OPS}
        # Decoded at D, registered: no opcode compare in the E1 cone (n16 timing #3c had
        # the scalar-B select decoded from e1_op fanning out to every lane's operand mux).
        e1_scalar_b = Signal()
        e1_redssq   = Signal()
        e1_is_ldw   = Signal()
        e1_is_mul   = Signal()

        # E2 record ---------------------------------------------------------------------------
        e2_valid    = Signal()
        e2_op       = Signal(6)
        e2_vd       = Signal(VBITS)
        e2_sd       = Signal(3)
        e2_is_load  = Signal()
        e2_is_ldw   = Signal()
        e2_is_store = Signal()
        e2_is_lut   = Signal()
        e2_is_red   = Signal()
        e2_is_mul   = Signal()
        e2_is_mulh  = Signal()
        e2_wr_v     = Signal()
        e2_f_lut    = {n: Signal(name=f"e2_f_{n.lower()}") for n in LUT_OPS}
        e2_f_red    = {n: Signal(name=f"e2_f_{n.lower()}") for n in RED_OPS}
        ex1_r       = Signal(32*LANES)
        mul_r       = [Signal((64, True), name=f"mul_r{i}") for i in range(LANES)]
        red1_sum    = [Signal((48, True), name=f"red1_sum{k}") for k in range(4)]
        red1_max    = [Signal(32, name=f"red1_max{k}") for k in range(4)]
        red1_smax   = [Signal((32, True), name=f"red1_smax{k}") for k in range(4)]
        e2_st_adr   = Signal(13)
        e2_st_we    = Signal(8)
        e2_st_dat   = Signal(256)
        e2_fr7      = [Signal(7, name=f"e2_fr7_{i}") for i in range(LANES)]
        e2_fr6      = [Signal(6, name=f"e2_fr6_{i}") for i in range(LANES)]
        e2_fq7      = [Signal(7, name=f"e2_fq7_{i}") for i in range(LANES)]

        # E3 record ---------------------------------------------------------------------------
        e3_valid    = Signal()
        e3_vd       = Signal(VBITS)
        e3_sd       = Signal(3)
        e3_is_lut   = Signal()
        e3_is_red   = Signal()
        e3_f_red    = {n: Signal(name=f"e3_f_{n.lower()}") for n in RED_OPS}
        # LUT interpolation is split across E2/E3 (n16 timing #3c: ROM read + subtract +
        # unregistered DSP multiply + add + select was a 16-level, 5.9 ns cone). E2 reads
        # the ROM and selects (e_a, e_b - e_a, frac) for the op; E3 multiplies and adds;
        # the result lands in lut_r at the E3/E4 edge and is written from E4. LUT latency
        # is 4 (a dependent one behind stalls three cycles instead of two).
        e3_lea      = [Signal((18, True), name=f"e3_lea{i}") for i in range(LANES)]
        e3_ld       = [Signal((19, True), name=f"e3_ld{i}") for i in range(LANES)]
        e3_lfr      = [Signal(8, name=f"e3_lfr{i}") for i in range(LANES)]
        red2_sum    = Signal((48, True))
        red2_ssq    = [Signal((64, True), name=f"red2_ssq{k}") for k in range(4)]
        red2_max    = Signal(32)
        red2_smax   = Signal((32, True))

        # E4 record (LUT writeback only) --------------------------------------------------------
        e4_valid    = Signal()
        e4_vd       = Signal(VBITS)
        e4_is_lut   = Signal()
        lut_r       = Signal(32*LANES)

        # Hazards / pipeline control ----------------------------------------------------------
        def vd_match(stage_vd):
            return (d_uses_va & (stage_vd == d_va)) | (d_uses_vb & (stage_vd == d_vb))
        def red_pending(x):
            return ((e1_valid & e1_is_red & (e1_sd == x))
                    | (e2_valid & e2_is_red & (e2_sd == x))
                    | (e3_valid & e3_is_red & (e3_sd == x)))
        stall_ld   = Signal()
        stall_mul  = Signal()
        stall_lut  = Signal()
        stall_waw  = Signal()
        stall_s    = Signal()
        stall_mem  = Signal()
        done_wait  = Signal()
        stall_d    = Signal()
        hold_e2    = Signal()
        d_go       = Signal()
        d_is_done  = Signal()
        d_is_loops = Signal()
        d_is_loope = Signal()
        loope_taken = Signal()
        d_is_loops2 = Signal()
        d_is_loope2 = Signal()
        loope2_taken = Signal()
        self.comb += [
            d_is_done.eq(d_op == OP["DONE"]),
            d_is_loops.eq(d_op == OP["LOOPS"]),
            d_is_loope.eq(d_op == OP["LOOPE"]),
            stall_ld.eq(e1_valid & e1_is_load & vd_match(e1_vd)),
            # Multiply results are NOT forwarded into E1 (n16 timing #3d: the DSP output ->
            # mulh select -> forward mux -> ALU -> select cone was the worst path, 15 levels).
            # A dependent one behind waits a cycle and takes the D-side wb2 path instead.
            stall_mul.eq(e1_valid & e1_is_mul & vd_match(e1_vd)),
            stall_lut.eq((e1_valid & e1_is_lut & vd_match(e1_vd))
                         | (e2_valid & e2_is_lut & vd_match(e2_vd))
                         | (e3_valid & e3_is_lut & vd_match(e3_vd))),
            # WAW: a LUT op writes at E4, everything else at E2. A non-LUT writer exactly
            # one behind a LUT op to the same vd would land first and be clobbered; hold
            # it one cycle so the two writes fall on the same edge (E2 wins there).
            stall_waw.eq(d_wr_v & ~d_is_lut & e1_valid & e1_is_lut & (e1_vd == d_vd)),
            stall_s.eq((d_uses_sa & red_pending(d_sa)) | (d_uses_sd & red_pending(d_sd))),
            # A load one behind a store to the same beat would read the RAM on the
            # edge the store writes it (read-during-write across ports): wait one cycle.
            stall_mem.eq(d_is_load & e1_valid & e1_is_store
                         & (e1_sa_r[5:18] == sa_d[5:18])),
            done_wait.eq(d_is_done & (e1_valid | e2_valid | e3_valid | e4_valid)),
            stall_d.eq(d_valid & (stall_ld | stall_mul | stall_lut | stall_waw | stall_s
                                  | stall_mem | done_wait)),
            hold_e2.eq(e2_valid & e2_is_store & ~self.sp_wr_gnt),
            d_go.eq(d_valid & ~stall_d & ~hold_e2),
            loope_taken.eq(d_go & d_is_loope & (loop_cnt > 1)),
            d_is_loops2.eq(d_op == OP["LOOPS2"]),
            d_is_loope2.eq(d_op == OP["LOOPE2"]),
            loope2_taken.eq(d_go & d_is_loope2 & (loop_cnt2 > 1)),
        ]

        start = Signal()
        self.comb += start.eq(Mux(self.seq_en, self.seq_start,
                                  self.submit.re & self.submit.storage))
        start_pc = Signal(16)
        self.comb += start_pc.eq(Mux(self.seq_en, self.seq_pc, self.pc.storage))

        # E2 writeback value (also the D-side forward source) -----------------------------------
        ld_val = Signal(32*LANES)
        for i in range(LANES):
            ldv_l = Cat(self.sp_rd_dat[16*i:16*(i+1)], Replicate(self.sp_rd_dat[16*i+15], 16))
            ldw_l = self.sp_rd_dat[32*i:32*(i+1)] if i < 8 else C(0, 32)
            self.comb += lane(ld_val, i).eq(Mux(e2_is_ldw, ldw_l, ldv_l))
        mul_val = Signal(32*LANES)
        for i in range(LANES):
            self.comb += lane(mul_val, i).eq(Mux(e2_is_mulh, mul_r[i][15:47], mul_r[i][0:32]))
        wb2_nl = Signal(32*LANES) # Non-load writeback: what E1 may forward.
        wb2    = Signal(32*LANES)
        self.comb += [
            wb2_nl.eq(Mux(e2_is_mul, mul_val, ex1_r)),
            wb2.eq(Mux(e2_is_load, ld_val, wb2_nl)),
        ]

        # D-side operand capture with forwarding: wb2 (E2) > lut_r (E4 LUT) > regfile.
        def d_operand(idx):
            v = Signal(32*LANES)
            self.comb += v.eq(
                Mux(e2_valid & e2_wr_v & ~e2_is_lut & (e2_vd == idx), wb2,
                Mux(e4_valid & e4_is_lut & (e4_vd == idx), lut_r,
                    V[idx])))
            return v
        va_d = d_operand(d_va)
        vb_d = d_operand(d_vb)

        # E1 datapath -------------------------------------------------------------------------
        fwd_a = Signal()
        fwd_b = Signal()
        e2_fwd_ok = Signal()
        self.comb += [
            e2_fwd_ok.eq(e2_valid & e2_wr_v & ~e2_is_load & ~e2_is_lut & ~e2_is_mul),
            fwd_a.eq(e2_fwd_ok & (e2_vd == e1_va)),
            fwd_b.eq(e2_fwd_ok & (e2_vd == e1_vb)),
        ]
        a_flat = Signal(32*LANES)
        b_flat = Signal(32*LANES)
        # Forward from ex1_r ONLY (n16 #4: e2_fwd_ok already excludes muls, but wb2_nl
        # still carried mul_val into this mux as a never-selected input - static timing
        # cannot see the exclusion, so the whole DSP -> mulh select -> forward -> ALU ->
        # one-hot cone stayed the worst family, 1024 endpoints at -1.14. Bit-identical:
        # for every case fwd_a/fwd_b can select, wb2_nl equals ex1_r.)
        self.comb += [
            a_flat.eq(Mux(fwd_a, ex1_r, e1_va_r)),
            b_flat.eq(Mux(fwd_b, ex1_r, e1_vb_r)),
        ]
        a_l, b_l = [], []
        for i in range(LANES):
            a = Signal((32, True))
            b = Signal((32, True))
            self.comb += [
                a.eq(lane(a_flat, i)),
                # Scalar-B forms (VSHLS/VMULS): b comes from S[sa] captured at D.
                b.eq(Mux(e1_scalar_b, e1_sa_r[0:32], lane(b_flat, i))),
            ]
            a_l.append(a)
            b_l.append(b)

        # Multipliers: capture at the E1 edge, results valid in E2. b muxed to a for REDSSQ.
        mul_c = []
        for i in range(LANES):
            b_in = Signal((32, True))
            m    = Signal((64, True))
            self.comb += [
                b_in.eq(Mux(e1_redssq, a_l[i], b_l[i])),
                m.eq(a_l[i]*b_in),
            ]
            mul_c.append(m)

        # Reductions, stage 1 (16 -> 4) from the E1 operands.
        abs_l = []
        for i in range(LANES):
            s = Signal(32)
            self.comb += s.eq(Mux(a_l[i] < 0, -a_l[i], a_l[i]))
            abs_l.append(s)
        def max2(x, y, signed):
            m = Signal((32, True)) if signed else Signal(32)
            self.comb += m.eq(Mux(x > y, x, y))
            return m
        red1_sum_c  = []
        red1_max_c  = []
        red1_smax_c = []
        for k in range(4):
            s = Signal((48, True))
            self.comb += s.eq(a_l[4*k] + a_l[4*k+1] + a_l[4*k+2] + a_l[4*k+3])
            red1_sum_c.append(s)
            red1_max_c.append(max2(max2(abs_l[4*k], abs_l[4*k+1], False),
                                   max2(abs_l[4*k+2], abs_l[4*k+3], False), False))
            red1_smax_c.append(max2(max2(a_l[4*k], a_l[4*k+1], True),
                                    max2(a_l[4*k+2], a_l[4*k+3], True), True))

        # Lanewise ALU results.
        res = {name: Signal(32*LANES, name=f"res_{name.lower()}") for name in ALU_OPS}
        c31n  = Signal((32, True), reset=-31)
        c8p   = Signal((32, True), reset=127)
        c8n   = Signal((32, True), reset=-127)
        c8p_v = Signal((32, True), reset=127)
        c8n_v = Signal((32, True), reset=-127)
        c16p   = Signal((32, True), reset=32767)
        c16n   = Signal((32, True), reset=-32767)
        c16p_v = Signal((32, True), reset=32767)
        c16n_v = Signal((32, True), reset=-32767)
        am5   = Signal(5)
        am5m1 = Signal(5) # am5 - 1 clamped at 0 (both Mux branches evaluate in simulation).
        self.comb += [
            am5.eq(e1_imm[0:5]),
            am5m1.eq(Mux(am5 != 0, am5 - 1, 0)),
        ]
        for i in range(LANES):
            a = a_l[i]
            b = b_l[i]
            self.comb += [
                lane(res["VADD"], i).eq(a + b),
                lane(res["VSUB"], i).eq(a - b),
                lane(res["VMIN"], i).eq(Mux(a < b, a, b)),
                lane(res["VMAX"], i).eq(Mux(a > b, a, b)),
                lane(res["VSHRI"], i).eq(a >> am5),
                lane(res["VSHLI"], i).eq(a << am5),
                # Clamp bounds as dedicated signed signals: literal negative constants
                # in comparisons can emit as unsigned-typed Verilog literals, flipping
                # the comparison unsigned (Verilator-visible only; Migen sim is fine).
                lane(res["VCLAMP8"],  i).eq(Mux(a > c8p, c8p_v, Mux(a < c8n, c8n_v, a))),
                lane(res["VCLAMP16"], i).eq(Mux(a > c16p, c16p_v, Mux(a < c16n, c16n_v, a))),
                lane(res["VABS"], i).eq(abs_l[i]),
                lane(res["BCAST"], i).eq(e1_sa_r[0:32]),
            ]
            # VRNE (round-to-nearest-even right shift by imm).
            q      = Signal((32, True))
            rbit   = Signal()
            sticky = Signal()
            mask   = Signal(32)
            rner   = Signal((32, True))
            self.comb += [
                q.eq(a >> am5),
                rbit.eq(Mux(am5 != 0, (a >> am5m1) & 1, 0)),
                mask.eq(~(C(0xFFFFFFFF, 32) << am5m1)[0:32]),
                sticky.eq(Mux(am5 > 1, (a & mask) != 0, 0)),
                rner.eq(q + (rbit & (sticky | q[0]))),
                lane(res["VRNE"], i).eq(Mux(am5 == 0, a, rner)),
            ]
            # VSHLV (per-lane signed variable shift, clamp +-31); VSHLS shares it.
            amtc = Signal((6, True))
            self.comb += [
                # -31 as a signed Signal: a negative literal here emits as an unsigned
                # Verilog literal (-6'd31 = 33), turning the comparison unsigned - small
                # positive amounts then clamp to -31 and left shifts become >>31.
                amtc.eq(Mux(b > 31, 31, Mux(b < c31n, c31n, b))),
                lane(res["VSHLV"], i).eq(Mux(amtc >= 0, a << amtc[0:5], a >> (-amtc)[0:5])),
            ]
            # VCLZN: shift normalizing |a| into [2^14, 2^15); 0 for a == 0.
            nrm   = Signal((32, True))
            cases = If(abs_l[i] == 0, nrm.eq(0))
            for bitpos in range(31, -1, -1):
                cases = cases.Elif(abs_l[i][bitpos], nrm.eq(14 - bitpos))
            self.comb += cases
            self.comb += lane(res["VCLZN"], i).eq(nrm)
        for i in range(8):
            self.comb += [
                lane(res["VPACK"], i).eq(a_l[i]),
                lane(res["VPACK"], i + 8).eq(b_l[i]),
            ]
        self.comb += res["VSHLS"].eq(res["VSHLV"])
        # One-hot AND-OR select from registered flags (no decode in the E1 cone).
        ex1_val = Signal(32*LANES)
        self.comb += ex1_val.eq(reduce(or_, [Replicate(e1_f[n], 32*LANES) & res[n]
                                             for n in ALU_OPS]))

        # LUT ROM addresses (E1, from the forwarded operand) and interpolation fractions.
        fr7_c, fr6_c, fq7_c = [], [], []
        for i in range(LANES):
            x15 = Signal(15)
            m15 = Signal(15)
            xu  = Signal(32)
            self.comb += [
                x15.eq(a_flat[32*i:32*i+15]),
                m15.eq(a_flat[32*i:32*i+15] - (1 << 14)),
                xu.eq(lane(a_flat, i)),
                rom_e2[i].adr.eq(x15[7:15]),
                rom_rec[i].adr.eq(m15[6:14]),
                # MSB-split rsqrt index (see bx.rsqrt_lut): [1,2) -> 0..127 step 1/128,
                # [2,4) -> 128..255 step 1/64.
                rom_rsq[i].adr.eq(Mux(xu[31], Cat(xu[24:31], C(1, 1)), Cat(xu[23:30], C(0, 1)))),
            ]
            fr7 = Signal(7); fr6 = Signal(6); fq7 = Signal(7)
            self.comb += [
                fr7.eq(x15[0:7]),
                fr6.eq(m15[0:6]),
                fq7.eq(Mux(xu[31], xu[17:24], xu[16:23])),
            ]
            fr7_c.append(fr7); fr6_c.append(fr6); fq7_c.append(fq7)

        # Store data packing (E1).
        st16 = Signal(256)
        st8  = Signal(128)
        self.comb += [
            st16.eq(Cat(*[a_flat[32*i:32*i+16] for i in range(LANES)])),
            st8.eq(Cat(*[a_flat[32*i:32*i+8] for i in range(LANES)])),
        ]
        st_we_c  = Signal(8)
        st_dat_c = Signal(256)
        self.comb += [
            If(e1_f["STB"],
                If(e1_sa_r[4],
                    st_we_c.eq(0xF0), st_dat_c.eq(Cat(C(0, 128), st8)),
                ).Else(
                    st_we_c.eq(0x0F), st_dat_c.eq(Cat(st8, C(0, 128))),
                ),
            ).Elif(e1_f["STW"],
                st_we_c.eq(0xFF), st_dat_c.eq(a_flat[0:256]), # Lanes 0..7 as 8 x int32.
            ).Else(
                st_we_c.eq(0xFF), st_dat_c.eq(st16),
            ),
        ]

        # Scratchpad ports.
        self.comb += [
            self.sp_rd_adr.eq(e1_sa_r[5:18]),
            self.sp_rd_req.eq(e1_valid & e1_is_load),
            self.sp_wr_adr.eq(e2_st_adr),
            self.sp_wr_dat.eq(e2_st_dat),
            self.sp_wr_req.eq(e2_valid & e2_is_store),
            self.sp_wr_we.eq(Mux(e2_valid & e2_is_store, e2_st_we, 0)),
        ]

        # E2 datapath: LUT operand select and reduction stage 2 -------------------------------
        # E2 picks (e_a, d = e_b - e_a, frac) for the op; E3 does e_a + ((d*frac) >> 7).
        # The recip table's 6-bit fraction is doubled so one shift serves all three:
        # (d*(2f)) >> 7 == (d*f) >> 6 exactly (floor on both sides).
        lea_c, ld_c, lfr_c = [], [], []
        for i in range(LANES):
            def endpoints(port):
                e_a = Signal((18, True))
                e_b = Signal((18, True))
                d   = Signal((19, True))
                self.comb += [
                    e_a.eq(port.dat_r[0:17]),
                    e_b.eq(port.dat_r[17:34]),
                    d.eq(e_b - e_a),
                ]
                return e_a, d
            ea_e2,  d_e2  = endpoints(rom_e2[i])
            ea_rec, d_rec = endpoints(rom_rec[i])
            ea_rsq, d_rsq = endpoints(rom_rsq[i])
            lea = Signal((18, True), name=f"lea_c{i}")
            ld  = Signal((19, True), name=f"ld_c{i}")
            lfr = Signal(8, name=f"lfr_c{i}")
            self.comb += [
                lea.eq(Mux(e2_f_lut["LUTE2"], ea_e2, Mux(e2_f_lut["LUTRECIP"], ea_rec, ea_rsq))),
                ld.eq(Mux(e2_f_lut["LUTE2"], d_e2, Mux(e2_f_lut["LUTRECIP"], d_rec, d_rsq))),
                lfr.eq(Mux(e2_f_lut["LUTE2"], e2_fr7[i],
                       Mux(e2_f_lut["LUTRECIP"], Cat(C(0, 1), e2_fr6[i]), e2_fq7[i]))),
            ]
            lea_c.append(lea); ld_c.append(ld); lfr_c.append(lfr)
        # E3 datapath: LUT interpolation multiply-add.
        lut_val = Signal(32*LANES)
        for i in range(LANES):
            pr  = Signal((19 + 8, True), name=f"lut_pr{i}")
            out = Signal((32, True), name=f"lut_out{i}")
            self.comb += [
                pr.eq(e3_ld[i]*e3_lfr[i]),
                out.eq(e3_lea[i] + (pr >> 7)),
                lane(lut_val, i).eq(out),
            ]
        red2_sum_c  = Signal((48, True))
        red2_max_c  = Signal(32)
        red2_smax_c = Signal((32, True))
        red2_ssq_c  = [Signal((64, True), name=f"red2_ssq_c{k}") for k in range(4)]
        self.comb += [
            red2_sum_c.eq(red1_sum[0] + red1_sum[1] + red1_sum[2] + red1_sum[3]),
            red2_max_c.eq(max2(max2(red1_max[0], red1_max[1], False),
                               max2(red1_max[2], red1_max[3], False), False)),
            red2_smax_c.eq(max2(max2(red1_smax[0], red1_smax[1], True),
                                max2(red1_smax[2], red1_smax[3], True), True)),
        ]
        for k in range(4):
            self.comb += red2_ssq_c[k].eq(mul_r[4*k] + mul_r[4*k+1] + mul_r[4*k+2] + mul_r[4*k+3])

        # E3: reductions read S[sd] late. REDMAX compares as the interpreter does - the
        # 48-bit SIGNED accumulator against the zero-extended absmax (the FSM unit compared
        # the low 32 bits unsigned, which differed for a negative accumulator; no stock
        # program reaches that, the random hazard stress does).
        ssq3 = Signal((64, True))
        sd3  = Signal((48, True))
        red2_max_s = Signal((48, True))
        self.comb += [
            ssq3.eq(red2_ssq[0] + red2_ssq[1] + red2_ssq[2] + red2_ssq[3]),
            sd3.eq(S[e3_sd]),
            red2_max_s.eq(red2_max),
        ]

        # Status.
        self.comb += [
            self.busy.eq(running),
            self.status.fields.busy.eq(running),
            self.status.fields.bad_op.eq(bad_op),
            self.done.eq(finish),
        ]

        # Sequential: F/D advance, D commits, E1/E2/E3 records -------------------------------
        self.sync += [
            # --- F / D ---
            If(start,
                pc.eq(start_pc),
                fetch_en.eq(1),
                running.eq(1),
                d_valid.eq(0),
                use_lf.eq(0),
                lf_arm.eq(0),
                use_lf2.eq(0),
                lf2_arm.eq(0),
                bad_op.eq(0),
                *[S[i].eq(0) for i in range(7)],
                S[7].eq(Mux(self.seq_en, self.seq_seed7, 0)),
            ).Elif(~hold_e2 & ~stall_d,
                If(loope_taken,
                    pc.eq(loop_start + 1),
                    use_lf.eq(1),
                    use_lf2.eq(0),
                    d_pc.eq(loop_start),
                    d_valid.eq(fetch_en),
                ).Elif(loope2_taken,
                    pc.eq(loop_start2 + 1),
                    use_lf2.eq(1),
                    use_lf.eq(0),
                    d_pc.eq(loop_start2),
                    d_valid.eq(fetch_en),
                ).Else(
                    insn_d.eq(imem_r.dat_r),
                    d_pc.eq(pc),
                    pc.eq(pc + 1),
                    use_lf.eq(0),
                    use_lf2.eq(0),
                    d_valid.eq(fetch_en & ~(d_go & d_is_done)),
                ),
            ),
            # Loop buffers: the word after a LOOPS/LOOPS2 is that loop's first instruction.
            If(lf_arm & d_valid,
                loop_first.eq(d_word),
                lf_arm.eq(0),
            ),
            If(lf2_arm & d_valid,
                loop_first2.eq(d_word),
                lf2_arm.eq(0),
            ),
            # --- D commits (exactly once, on d_go) ---
            If(d_go,
                If(d_op == OP["MOVSI"],  S[d_sd].eq(d_imm)),
                If(d_op == OP["ARG"],    S[d_sd].eq(Cat(arg_sel, Replicate(arg_sel[31], 16)))),
                If(d_op == OP["SADDI"],  S[d_sd].eq(sd_d + d_imm)),
                If(d_op == OP["SMOV"],   S[d_sd].eq(sa_d)),
                If(d_op == OP["SADD"],   S[d_sd].eq(sd_d + sa_d)),
                If(d_op == OP["SSHLV"],  S[d_sd].eq(sshlv_res)),
                If(d_op == OP["SNORME"], S[d_sd].eq(snorme_res)),
                If(d_is_loops,
                    If(d_imm == 0,
                        loop_cnt.eq(sa_d[0:32]),
                    ).Else(
                        loop_cnt.eq(d_imm),
                    ),
                    loop_start.eq(d_pc + 1),
                    lf_arm.eq(1),
                ),
                If(loope_taken, loop_cnt.eq(loop_cnt - 1)),
                If(d_is_loops2,
                    If(d_imm == 0,
                        loop_cnt2.eq(sa_d[0:32]),
                    ).Else(
                        loop_cnt2.eq(d_imm),
                    ),
                    loop_start2.eq(d_pc + 1),
                    lf2_arm.eq(1),
                ),
                If(loope2_taken, loop_cnt2.eq(loop_cnt2 - 1)),
                If((d_op == OP["LDV"]) | (d_op == OP["LDW"]) | (d_op == OP["STV"])
                   | (d_op == OP["STW"]),
                    S[d_sa].eq(sa_d + 32),
                ),
                If(d_op == OP["STB"], S[d_sa].eq(sa_d + 16)),
                If(d_is_done, fetch_en.eq(0), finish.eq(1)),
                If(~d_is_known, bad_op.eq(1)),
            ),
            # --- FINISH (one cycle after DONE committed; the pipe is empty by then) ---
            If(finish,
                finish.eq(0),
                running.eq(0),
                self.done_count.status.eq(self.done_count.status + 1),
                self.res5.status.eq(S[5]),
                self.res6.status.eq(S[6]),
                self.res7.status.eq(S[7]),
            ),
            # --- E1 record (from D) ---
            If(~hold_e2,
                e1_valid.eq(d_valid & ~stall_d),
                e1_op.eq(d_op),
                e1_vd.eq(d_vd), e1_va.eq(d_va), e1_vb.eq(d_vb), e1_sd.eq(d_sd),
                e1_imm.eq(d_imm),
                e1_va_r.eq(va_d),
                e1_vb_r.eq(vb_d),
                e1_sa_r.eq(sa_d),
                e1_is_load.eq(d_is_load),
                e1_is_store.eq(d_is_store),
                e1_is_lut.eq(d_is_lut),
                e1_is_red.eq(d_is_red),
                e1_wr_v.eq(d_wr_v),
                *[e1_f[n].eq(d_op == OP[n]) for n in e1_f],
                e1_scalar_b.eq((d_op == OP["VSHLS"]) | (d_op == OP["VMULS"])),
                e1_redssq.eq(d_op == OP["REDSSQ"]),
                e1_is_ldw.eq(d_op == OP["LDW"]),
                e1_is_mul.eq(is_in(d_op, MUL_OPS)),
                # --- E2 record (from E1) ---
                e2_valid.eq(e1_valid),
                e2_op.eq(e1_op),
                e2_vd.eq(e1_vd), e2_sd.eq(e1_sd),
                e2_is_load.eq(e1_is_load),
                e2_is_ldw.eq(e1_is_ldw),
                e2_is_store.eq(e1_is_store),
                e2_is_lut.eq(e1_is_lut),
                e2_is_red.eq(e1_is_red),
                e2_is_mul.eq(e1_is_mul),
                e2_is_mulh.eq(e1_op == OP["VMULH15"]),
                e2_wr_v.eq(e1_wr_v),
                *[e2_f_lut[n].eq(e1_f[n]) for n in LUT_OPS],
                *[e2_f_red[n].eq(e1_f[n]) for n in RED_OPS],
                ex1_r.eq(ex1_val),
                *[mul_r[i].eq(mul_c[i]) for i in range(LANES)],
                *[red1_sum[k].eq(red1_sum_c[k]) for k in range(4)],
                *[red1_max[k].eq(red1_max_c[k]) for k in range(4)],
                *[red1_smax[k].eq(red1_smax_c[k]) for k in range(4)],
                e2_st_adr.eq(e1_sa_r[5:18]),
                e2_st_we.eq(st_we_c),
                e2_st_dat.eq(st_dat_c),
                *[e2_fr7[i].eq(fr7_c[i]) for i in range(LANES)],
                *[e2_fr6[i].eq(fr6_c[i]) for i in range(LANES)],
                *[e2_fq7[i].eq(fq7_c[i]) for i in range(LANES)],
                # --- E3 record (from E2) ---
                e3_valid.eq(e2_valid),
                e3_vd.eq(e2_vd), e3_sd.eq(e2_sd),
                e3_is_lut.eq(e2_is_lut),
                e3_is_red.eq(e2_is_red),
                *[e3_f_red[n].eq(e2_f_red[n]) for n in RED_OPS],
                *[e3_lea[i].eq(lea_c[i]) for i in range(LANES)],
                *[e3_ld[i].eq(ld_c[i]) for i in range(LANES)],
                *[e3_lfr[i].eq(lfr_c[i]) for i in range(LANES)],
                red2_sum.eq(red2_sum_c),
                *[red2_ssq[k].eq(red2_ssq_c[k]) for k in range(4)],
                red2_max.eq(red2_max_c),
                red2_smax.eq(red2_smax_c),
            ).Else(
                # A held store freezes everything upstream and nothing downstream.
                e3_valid.eq(0),
            ),
            # --- E4 record (from E3): always advances, held store or not ---
            e4_valid.eq(e3_valid),
            e4_vd.eq(e3_vd),
            e4_is_lut.eq(e3_is_lut),
            lut_r.eq(lut_val),
            # --- V writes: E4 (LUT) first, E2 second so the younger instruction wins ---
            If(e4_valid & e4_is_lut, V[e4_vd].eq(lut_r)),
            If(e2_valid & ~hold_e2 & e2_wr_v, V[e2_vd].eq(wb2)),
            # --- E3 reductions (late S read; the scoreboard keeps every other writer off) ---
            If(e3_valid & e3_is_red,
                If(e3_f_red["REDSUM"], S[e3_sd].eq(sd3 + red2_sum)),
                If(e3_f_red["REDSSQ"], S[e3_sd].eq(sd3 + ssq3[0:48])),
                If(e3_f_red["REDMAX"] & (red2_max_s > sd3), S[e3_sd].eq(red2_max)),
                If(e3_f_red["REDSMAX"] & (red2_smax > sd3), S[e3_sd].eq(red2_smax)),
            ),
        ]
