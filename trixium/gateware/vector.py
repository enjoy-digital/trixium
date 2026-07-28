#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""BonsaiVectorUnit: microcoded 16-lane SIMD engine executing trixium.model.asm binaries.

Bit-exact contract: the asm.py INTERPRETER is the golden model - this RTL must produce
identical scratchpad contents and scalar results for the same binary and arguments.

Implementation: uniform 3-cycle instructions (FETCH -> EX1 -> EX2) - trivial to verify,
and still >100x faster than the soft-float CPU paths it replaces (~40k cycles for a
6144-element silu_mul). 16 lanes x int32 (8 vregs as flat 512-bit signals), 8 x 48-bit
scalars, one hardware loop, Q15 LUT ops with x16-replicated dual-entry ROMs.

Cycle contract per instruction:
  FETCH: imem_r.adr = pc (data valid during EX1).
  EX1  : decode from imem_r.dat_r; scratchpad read launch (sp_rd_adr from S[sa]); LUT ROM
         addresses launch (from V[va]); multiplier capture; instr registered.
  EX2  : all effects: vreg writeback, scalar updates, address post-increment, store port
         drive, pc update (loop/done).

Scratchpad access goes through externally-muxed ports (the engine's fill/gather ports -
never a 5th RAM port): sp_rd (256-bit read, 1-cycle latency) and sp_wr (256-bit write,
32-bit lane enables). The integrator grants the ports when the engine is idle."""

from functools import reduce
from operator import or_

from migen import *

from litex.gen import *

from litex.soc.interconnect.csr import *

from trixium.model import asm as vasm
from trixium.model import bitexact as bx

LANES = 16

# BonsaiVectorUnit ---------------------------------------------------------------------------------

class BonsaiVectorUnit(LiteXModule):
    def __init__(self, imem_depth=256):
        # Scratchpad ports (muxed onto the engine's fill/gather ports by the integrator).
        self.sp_rd_adr = Signal(13)  # 256-bit beat index.
        self.sp_rd_dat = Signal(256) # Valid 1 cycle after adr.
        self.sp_wr_adr = Signal(13)
        self.sp_wr_we  = Signal(8)   # 32-bit lane enables.
        self.sp_wr_dat = Signal(256)

        self.busy = Signal()

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

        # Instruction memory (CSR-uploaded).
        imem = Memory(32, imem_depth)
        self.specials += imem
        imem_w = imem.get_port(write_capable=True)
        imem_r = imem.get_port()
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
        V = Array([Signal(32*LANES, name=f"vr{i}") for i in range(8)])
        S = Array([Signal((48, True), name=f"sr{i}") for i in range(8)])

        # Decode: fields from imem output during EX1, from the registered copy during EX2.
        instr_r = Signal(32)
        in_ex1  = Signal()
        cur     = Signal(32)
        self.comb += cur.eq(Mux(in_ex1, imem_r.dat_r, instr_r))

        f_op  = Signal(6)
        f_vd  = Signal(3)
        f_va  = Signal(3)
        f_vb  = Signal(3)
        f_sd  = Signal(3)
        f_sa  = Signal(3)
        f_imm = Signal((11, True))
        self.comb += [
            f_op.eq(cur[26:32]),
            f_vd.eq(cur[23:26]),
            f_va.eq(cur[20:23]),
            f_vb.eq(cur[17:20]),
            f_sd.eq(cur[14:17]),
            f_sa.eq(cur[11:14]),
            f_imm.eq(cur[0:11]),
        ]

        pc         = Signal(16)
        loop_start = Signal(16)
        loop_cnt   = Signal(32)

        va_flat = Signal(32*LANES)
        vb_flat = Signal(32*LANES)
        sa_val  = Signal((48, True))
        sd_val  = Signal((48, True))
        self.comb += [
            va_flat.eq(V[f_va]),
            vb_flat.eq(V[f_vb]),
            sa_val.eq(S[f_sa]),
            sd_val.eq(S[f_sd]),
        ]

        def lane(sig, i):
            return sig[32*i:32*(i+1)]

        a_l = []
        b_l = []
        for i in range(LANES):
            a = Signal((32, True))
            b = Signal((32, True))
            self.comb += a.eq(va_flat[32*i:32*(i+1)])
            self.comb += b.eq(vb_flat[32*i:32*(i+1)])
            a_l.append(a)
            b_l.append(b)

        # Multipliers: capture at EX1 edge, results valid in EX2. b muxed to a for REDSSQ.
        mul_full = []
        for i in range(LANES):
            b_in = Signal((32, True))
            self.comb += b_in.eq(Mux(f_op == OP["REDSSQ"], a_l[i], b_l[i]))
            m = Signal((64, True))
            self.sync += If(in_ex1, m.eq(a_l[i]*b_in))
            mul_full.append(m)

        # Reductions (comb over EX2's operand view - V regs are stable between EX1 and EX2).
        red_sum = Signal((48, True))
        self.comb += red_sum.eq(sum(a_l))
        sq_sum = Signal((64, True))
        self.comb += sq_sum.eq(sum(mul_full))
        abs_l = []
        for i in range(LANES):
            s = Signal(32)
            self.comb += s.eq(Mux(a_l[i] < 0, -a_l[i], a_l[i]))
            abs_l.append(s)
        def tree_max(sigs):
            level = list(sigs)
            while len(level) > 1:
                nxt = []
                for j in range(0, len(level), 2):
                    if j + 1 < len(level):
                        m = Signal(32)
                        self.comb += m.eq(Mux(level[j] > level[j+1], level[j], level[j+1]))
                        nxt.append(m)
                    else:
                        nxt.append(level[j])
                level = nxt
            return level[0]
        red_max_v = tree_max(abs_l)
        def tree_smax(sigs):
            level = list(sigs)
            while len(level) > 1:
                nxt = []
                for j in range(0, len(level), 2):
                    if j + 1 < len(level):
                        m = Signal((32, True))
                        self.comb += m.eq(Mux(level[j] > level[j+1], level[j], level[j+1]))
                        nxt.append(m)
                    else:
                        nxt.append(level[j])
                level = nxt
            return level[0]
        red_smax_v = tree_smax(a_l)

        # Lanewise results --------------------------------------------------------------------
        res = {name: Signal(32*LANES, name=f"res_{name.lower()}") for name in
               ("VADD", "VSUB", "VMUL", "VMULH15", "VMIN", "VMAX", "VSHRI", "VSHLI",
                "VSHLV", "VRNE", "VCLAMP8", "VCLAMP16", "VABS", "VCLZN", "VPACK",
                "LDV", "LDW", "BCAST", "LUTE2", "LUTRECIP", "LUTRSQRT")}

        am5 = Signal(5)
        am5m1 = Signal(5) # am5 - 1 clamped at 0 (both Mux branches evaluate in simulation).
        self.comb += [
            am5.eq(f_imm[0:5]),
            am5m1.eq(Mux(am5 != 0, am5 - 1, 0)),
        ]

        for i in range(LANES):
            a = a_l[i]
            b = b_l[i]
            self.comb += [
                lane(res["VADD"], i).eq(a + b),
                lane(res["VSUB"], i).eq(a - b),
                lane(res["VMUL"], i).eq(mul_full[i][0:32]),
                lane(res["VMULH15"], i).eq(mul_full[i][15:47]),
                lane(res["VMIN"], i).eq(Mux(a < b, a, b)),
                lane(res["VMAX"], i).eq(Mux(a > b, a, b)),
                lane(res["VSHRI"], i).eq(a >> am5),
                lane(res["VSHLI"], i).eq(a << am5),
                lane(res["VCLAMP8"],  i).eq(Mux(a > 127, 127, Mux(a < -127, -127, a))),
                lane(res["VCLAMP16"], i).eq(Mux(a > 32767, 32767, Mux(a < -32767, -32767, a))),
                lane(res["VABS"], i).eq(abs_l[i]),
                lane(res["BCAST"], i).eq(sa_val[0:32]),
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
            # VSHLV (per-lane signed variable shift, clamp +-31).
            amtc = Signal((6, True))
            self.comb += [
                amtc.eq(Mux(b > 31, 31, Mux(b < -31, -31, b))),
                lane(res["VSHLV"], i).eq(Mux(amtc >= 0, a << amtc[0:5], a >> (-amtc)[0:5])),
            ]
            # VCLZN: shift normalizing |a| into [2^14, 2^15); 0 for a == 0.
            nrm   = Signal((32, True))
            cases = If(abs_l[i] == 0, nrm.eq(0))
            for bitpos in range(31, -1, -1):
                cases = cases.Elif(abs_l[i][bitpos], nrm.eq(14 - bitpos))
            self.comb += cases
            self.comb += lane(res["VCLZN"], i).eq(nrm)
            # Loads (sign-extended int16 / raw int32 in lanes 0..7).
            self.comb += lane(res["LDV"], i).eq(
                Cat(self.sp_rd_dat[16*i:16*(i+1)], Replicate(self.sp_rd_dat[16*i+15], 16)))
            if i < 8:
                self.comb += lane(res["LDW"], i).eq(self.sp_rd_dat[32*i:32*(i+1)])
            else:
                self.comb += lane(res["LDW"], i).eq(0)

        for i in range(8):
            self.comb += [
                lane(res["VPACK"], i).eq(a_l[i]),
                lane(res["VPACK"], i + 8).eq(b_l[i]),
            ]

        # LUT interpolation: ROM addresses launched in EX1 (comb from va), data in EX2.
        for i in range(LANES):
            x15 = Signal(15)
            self.comb += x15.eq(va_flat[32*i:32*i+15])
            self.comb += rom_e2[i].adr.eq(x15[7:15])
            m15 = Signal(15)
            self.comb += m15.eq(va_flat[32*i:32*i+15] - (1 << 14))
            self.comb += rom_rec[i].adr.eq(m15[6:14])
            # MSB-split rsqrt index (see bx.rsqrt_lut): [1,2) -> 0..127 step 1/128,
            # [2,4) -> 128..255 step 1/64. Subtracting the leading power of two just
            # clears that bit, so the address is a plain bit-select muxed on x[31].
            xu = Signal(32)
            self.comb += xu.eq(va_flat[32*i:32*(i+1)])
            self.comb += rom_rsq[i].adr.eq(Mux(xu[31],
                Cat(xu[24:31], C(1, 1)), Cat(xu[23:30], C(0, 1))))

            def interp(port, frac, fbits, out_lane):
                e_a = Signal((18, True))
                e_b = Signal((18, True))
                d   = Signal((19, True))
                pr  = Signal((19 + 7, True))
                self.comb += [
                    e_a.eq(port.dat_r[0:17]),
                    e_b.eq(port.dat_r[17:34]),
                    d.eq(e_b - e_a),
                    pr.eq(d*frac),
                    out_lane.eq(e_a + (pr >> fbits)),
                ]
            fr7 = Signal(7)
            self.comb += fr7.eq(x15[0:7])
            interp(rom_e2[i], fr7, 7, lane(res["LUTE2"], i))
            fr6 = Signal(6)
            self.comb += fr6.eq(m15[0:6])
            interp(rom_rec[i], fr6, 6, lane(res["LUTRECIP"], i))
            fq7 = Signal(7)
            self.comb += fq7.eq(Mux(xu[31], xu[17:24], xu[16:23]))
            interp(rom_rsq[i], fq7, 7, lane(res["LUTRSQRT"], i))

        # Vector writeback mux.
        vec_wb_ops = ["VADD", "VSUB", "VMUL", "VMULH15", "VMIN", "VMAX", "VSHRI", "VSHLI",
                      "VSHLV", "VRNE", "VCLAMP8", "VCLAMP16", "VABS", "VCLZN", "VPACK",
                      "LDV", "LDW", "BCAST", "LUTE2", "LUTRECIP", "LUTRSQRT"]
        wb_val = Signal(32*LANES)
        self.comb += Case(f_op, {OP[n]: wb_val.eq(res[n]) for n in vec_wb_ops})
        is_vec_wb = Signal()
        self.comb += is_vec_wb.eq(reduce(or_, [f_op == OP[n] for n in vec_wb_ops]))

        known_ops = set(vasm.OPCODES)
        is_known = Signal()
        self.comb += is_known.eq(reduce(or_, [f_op == OP[n] for n in vasm.OPCODES]))

        # Store data.
        st16 = Signal(256)
        st8  = Signal(128)
        self.comb += [
            st16.eq(Cat(*[va_flat[32*i:32*i+16] for i in range(LANES)])),
            st8.eq(Cat(*[va_flat[32*i:32*i+8] for i in range(LANES)])),
        ]

        # Scalar SSHLV: 48-bit variable shift of S[sd] by S[sa] (clamped +-47), wraps to 48.
        ssh_amt = Signal((7, True))
        self.comb += ssh_amt.eq(Mux(sa_val > 47, 47, Mux(sa_val < -47, -47, sa_val[0:7])))
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
            ssh_full.eq(sd_val << ssh_amt_pos),
            ssh_right.eq(sd_val >> ssh_amt_neg),
            sshlv_res.eq(Mux(ssh_amt >= 0, ssh_full[0:48], ssh_right)),
        ]

        # Scalar SNORME: even shift landing S[sa] in [2^30, 2^32); 0 for S[sa] <= 0.
        snorme_res = Signal((48, True))
        sn_cases = If(sa_val <= 0, snorme_res.eq(0))
        for p in range(46, -1, -1):
            sn_cases = sn_cases.Elif(sa_val[p],
                snorme_res.eq((30 - p) if (p % 2 == 0) else (31 - p)))
        self.comb += sn_cases

        # Args (32-bit CSRs, sign-extended into the 48-bit scalar file by ARG).
        args_arr = Array([self.arg0.storage, self.arg1.storage, self.arg2.storage,
                          self.arg3.storage, self.arg4.storage, self.arg5.storage,
                          self.arg6.storage, self.arg7.storage])
        arg_sel = Signal(32)
        self.comb += arg_sel.eq(args_arr[f_imm[0:3]])

        # Control FSM -------------------------------------------------------------------------
        running = Signal()
        bad_op  = Signal()
        self.comb += [
            self.busy.eq(running),
            self.status.fields.busy.eq(running),
            self.status.fields.bad_op.eq(bad_op),
        ]
        start = Signal()
        self.comb += start.eq(self.submit.re & self.submit.storage)

        self.fsm = fsm = FSM(reset_state="IDLE")

        self.comb += imem_r.adr.eq(pc)
        # Scratchpad read launch (EX1): address from S[sa] byte address.
        self.comb += self.sp_rd_adr.eq(sa_val[5:18])

        fsm.act("IDLE",
            If(start,
                NextValue(pc, self.pc.storage),
                NextValue(running, 1),
                NextState("FETCH"),
            )
        )
        fsm.act("FETCH",
            NextState("EX1"),
        )
        fsm.act("EX1",
            in_ex1.eq(1),
            NextState("EX2"),
        )
        self.sync += If(in_ex1, instr_r.eq(imem_r.dat_r))

        fsm.act("EX2",
            # Store port drive.
            If(f_op == OP["STV"],
                self.sp_wr_adr.eq(sa_val[5:18]),
                self.sp_wr_we.eq(0xFF),
                self.sp_wr_dat.eq(st16),
            ),
            If(f_op == OP["STB"],
                self.sp_wr_adr.eq(sa_val[5:18]),
                If(sa_val[4],
                    self.sp_wr_we.eq(0xF0),
                    self.sp_wr_dat.eq(Cat(C(0, 128), st8)),
                ).Else(
                    self.sp_wr_we.eq(0x0F),
                    self.sp_wr_dat.eq(Cat(st8, C(0, 128))),
                ),
            ),
            If(f_op == OP["STW"],
                self.sp_wr_adr.eq(sa_val[5:18]),
                self.sp_wr_we.eq(0xFF),
                self.sp_wr_dat.eq(va_flat[0:256]), # Lanes 0..7 as 8 x int32.
            ),
            # PC update.
            If(f_op == OP["DONE"],
                NextState("FINISH"),
            ).Elif((f_op == OP["LOOPE"]) & (loop_cnt > 1),
                NextValue(loop_cnt, loop_cnt - 1),
                NextValue(pc, loop_start),
                NextState("FETCH"),
            ).Else(
                NextValue(pc, pc + 1),
                NextState("FETCH"),
            ),
        )
        fsm.act("FINISH",
            NextValue(running, 0),
            NextState("IDLE"),
        )

        # EX2 effects (sync).
        self.sync += [
            If(fsm.ongoing("EX2"),
                # Vector writeback.
                If(is_vec_wb,
                    V[f_vd].eq(wb_val),
                ),
                # Scalar ops.
                If(f_op == OP["MOVSI"], S[f_sd].eq(f_imm)),
                If(f_op == OP["ARG"],   S[f_sd].eq(Cat(arg_sel, Replicate(arg_sel[31], 16)))),
                If(f_op == OP["SADDI"], S[f_sd].eq(sd_val + f_imm)),
                If(f_op == OP["SMOV"],  S[f_sd].eq(sa_val)),
                If(f_op == OP["SADD"],  S[f_sd].eq(sd_val + sa_val)),
                If(f_op == OP["SSHLV"], S[f_sd].eq(sshlv_res)),
                If(f_op == OP["SNORME"], S[f_sd].eq(snorme_res)),
                If(f_op == OP["LOOPS"],
                    If(f_imm == 0,
                        loop_cnt.eq(sa_val[0:32]),
                    ).Else(
                        loop_cnt.eq(f_imm),
                    ),
                    loop_start.eq(pc + 1),
                ),
                # Memory post-increment.
                If((f_op == OP["LDV"]) | (f_op == OP["LDW"]) | (f_op == OP["STV"])
                   | (f_op == OP["STW"]),
                    S[f_sa].eq(sa_val + 32),
                ),
                If(f_op == OP["STB"],
                    S[f_sa].eq(sa_val + 16),
                ),
                # Reductions.
                If(f_op == OP["REDSUM"], S[f_sd].eq(sd_val + red_sum)),
                If(f_op == OP["REDSSQ"], S[f_sd].eq(sd_val + sq_sum[0:48])),
                If(f_op == OP["REDMAX"],
                    If(red_max_v > sd_val[0:32],
                        S[f_sd].eq(red_max_v),
                    )
                ),
                If(f_op == OP["REDSMAX"],
                    If(red_smax_v > sd_val,
                        S[f_sd].eq(red_smax_v),
                    )
                ),
                If(~is_known, bad_op.eq(1)),
            ),
            If(start,
                bad_op.eq(0),
                *[S[i].eq(0) for i in range(8)],
            ),
            If(fsm.ongoing("FINISH"),
                self.done_count.status.eq(self.done_count.status + 1),
                self.res5.status.eq(S[5]),
                self.res6.status.eq(S[6]),
                self.res7.status.eq(S[7]),
            ),
        ]
