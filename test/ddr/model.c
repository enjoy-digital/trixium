/* SPDX-License-Identifier: BSD-2-Clause
 *
 * An eye model of a LiteDRAM DFI injector and a US+ PHY's delay CSRs, for liblitedram's
 * sdram.c compiled natively (test/ddr/test_wtrain.py).
 *
 * The model is the controller's CSR page of the XCU1525 build (sdram at 0x20000, ddrphy at
 * 0x20800 from CSR_BASE: the shim generated/csr.h and software/host/rdlevel.py's constants):
 * csr_read_simple/csr_write_simple below are what the generated accessors resolve to, so the
 * unmodified sdram.c/accessors.c and rdlevel.py (through a Bar0 that calls them) drive the same
 * registers. Per module, a write eye and a read eye in (bitslip, delay) space: a burst written
 * with the module's write bitslip/DQS/DQ inside its write eye lands, one read with the read
 * bitslip/delay inside its read eye comes back right; outside, the data is what a wrong
 * setting gives on a real lane (a wrong bitslip shifts the beats, a write off its eye never
 * lands, a read off its eye is garbled). Inside the eye but within `margin` taps of an edge,
 * bytes with many bit transitions fail: the only data-dependent behaviour, so a scan with one
 * seed pair sees a slightly wider window than one with three.
 *
 * DDR4 MPR reads are modelled (MR3 A2 set: a READ returns MPR page 0's patterns of BA[1:0] on
 * every DQ through the read eye), so liblitedram's MPR read leveling levels the reads first.
 *
 * The US+ PHY has no software reset of the DQS write delay (the CSR exists, nothing uses it):
 * the model ignores the pulse unless model_set_dqs_rst_wired(1). Delay increments are ignored
 * while EN_VTC is high (UG571), commands while the controller is not under software control.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define PHASES   4
#define BYTES    16         /* DFI data bytes per phase (64-bit DDR4, 2 beats). */
#define BEATS    (2*PHASES)
#define MODULES  8
#define DELAYS   512
#define BITSLIPS 8

/* CSR map (offsets from CSR_BASE, 0xf0000000). */
#define CSR_BASE_ADDR  0xf0000000UL
#define SDRAM_OFF      0x20000
#define DDRPHY_OFF     0x20800
#define PI_STRIDE      0x30

#define SDRAM_CONTROL        0x00
#define SDRAM_PI_BASE        0x04
#define PI_COMMAND           0x00
#define PI_COMMAND_ISSUE     0x04
#define PI_ADDRESS           0x08
#define PI_BADDRESS          0x0c
#define PI_WRDATA            0x10   /* 4 words, big ordering */
#define PI_RDDATA            0x20   /* 4 words, big ordering */

#define PHY_RST                 0x00
#define PHY_EN_VTC              0x04
#define PHY_HALF_SYS8X_TAPS     0x08
#define PHY_WLEVEL_EN           0x0c
#define PHY_WLEVEL_STROBE       0x10
#define PHY_CDLY_RST            0x14
#define PHY_CDLY_INC            0x18
#define PHY_CDLY_VALUE          0x1c
#define PHY_DLY_SEL             0x20
#define PHY_RDLY_DQ_RST         0x24
#define PHY_RDLY_DQ_INC         0x28
#define PHY_RDLY_DQ_BITSLIP_RST 0x2c
#define PHY_RDLY_DQ_BITSLIP     0x30
#define PHY_WDLY_DQ_RST         0x34
#define PHY_WDLY_DQ_INC         0x38
#define PHY_WDLY_DQS_RST        0x3c
#define PHY_WDLY_DQS_INC        0x40
#define PHY_WDLY_DQS_INC_COUNT  0x44
#define PHY_WDLY_DQ_BITSLIP_RST 0x48
#define PHY_WDLY_DQ_BITSLIP     0x4c
#define PHY_RDPHASE             0x50
#define PHY_WRPHASE             0x54

#define DFII_CONTROL_SEL 0x01
#define CMD_CS     0x01
#define CMD_WE     0x02
#define CMD_CAS    0x04
#define CMD_RAS    0x08
#define CMD_WRDATA 0x10
#define CMD_RDDATA 0x20

struct module_state {
	int wbs, dqs, dq;      /* write bitslip, DQS and DQ delays */
	int rbs, rd;           /* read bitslip and delay */
	int dqs_count;         /* the PHY's wdly_dqs_inc_count (9 bits, never reset) */
};

struct write_eye {
	int set;               /* 0: nothing lands */
	int wbs;               /* the write bitslip that lands the burst in place */
	int dqs_lo, dqs_hi;    /* DQS positions (absolute taps) inside tDQSS */
	int off_c, off_hw;     /* the DQ-DQS offset window, circular: centre, half-width */
	int margin;            /* marginal taps inside each edge */
};

struct read_eye {
	int set;
	int rbs;               /* the read bitslip of the window */
	int c, hw;             /* the delay window: centre, half-width; it spills into the next
	                          (previous) bitslip past delay 511 (before 0) */
	int margin;
};

static struct {
	uint32_t control, en_vtc, dly_sel, rdphase, wrphase, wlevel_en, cdly, phy_rst;
	uint32_t command[PHASES], address[PHASES], baddress[PHASES];
	uint32_t wrdata[PHASES][4], rddata[PHASES][4];
	uint8_t  mem[PHASES][BYTES];          /* the scratch burst: row 0, bank 0 */
	struct module_state mod[MODULES];
	struct write_eye    weye[MODULES];
	struct read_eye     reye[MODULES];
	int dqs_rst_wired;
	int mpr;                              /* MR3 A2: reads return the MPR patterns */
	unsigned long reads, writes, unknown, checks;
	int memtest_fail, memtest_calls;
} M;

/* Eyes ------------------------------------------------------------------------------------------*/

static int circ_dist(int a, int b) {
	int d = ((a - b) % DELAYS + DELAYS) % DELAYS;
	return d > DELAYS/2 ? DELAYS - d : d;
}

static int transitions(uint8_t v) {
	return __builtin_popcount((v ^ (v >> 1)) & 0x7f);
}

/* The write side of module m: 0 outside the eye, 1 inside, 2 inside but marginal. */
static int write_zone(int m) {
	const struct write_eye *e = &M.weye[m];
	const struct module_state *s = &M.mod[m];
	int off;
	if (!e->set || s->wbs != e->wbs || s->dqs < e->dqs_lo || s->dqs > e->dqs_hi)
		return 0;
	off = ((s->dq - s->dqs) % DELAYS + DELAYS) % DELAYS;
	if (circ_dist(off, e->off_c) > e->off_hw)
		return 0;
	if ((s->dqs - e->dqs_lo < e->margin) || (e->dqs_hi - s->dqs < e->margin) ||
	    (e->off_hw - circ_dist(off, e->off_c) < e->margin))
		return 2;
	return 1;
}

/* The read side: the delay unwrapped around the window's bitslip. */
static int read_zone(int m) {
	const struct read_eye *e = &M.reye[m];
	const struct module_state *s = &M.mod[m];
	int u, d;
	if (!e->set)
		return 0;
	if (s->rbs == e->rbs)
		u = s->rd;
	else if (s->rbs == (e->rbs + 1) % BITSLIPS)
		u = s->rd + DELAYS;
	else if (s->rbs == (e->rbs + BITSLIPS - 1) % BITSLIPS)
		u = s->rd - DELAYS;
	else
		return 0;
	d = u > e->c ? u - e->c : e->c - u;
	if (d > e->hw)
		return 0;
	return (e->hw - d < e->margin) ? 2 : 1;
}

/* Data path ---------------------------------------------------------------------------------------*/

/* Beat k of module m: phase k/2, the positive edge byte m then the negative edge byte m+8. */
static uint8_t *beat(uint8_t data[PHASES][BYTES], int m, int k) {
	return &data[k/2][m + (k % 2)*BYTES/2];
}

static void unpack(uint32_t words[PHASES][4], uint8_t data[PHASES][BYTES]) {
	/* Big word ordering: word j holds bytes 4*(3-j).. of the little-endian 16-byte value. */
	for (int p = 0; p < PHASES; p++)
		for (int j = 0; j < 4; j++)
			for (int b = 0; b < 4; b++)
				data[p][4*(3 - j) + b] = words[p][j] >> (8*b);
}

static void pack(uint8_t data[PHASES][BYTES], uint32_t words[PHASES][4]) {
	for (int p = 0; p < PHASES; p++)
		for (int j = 0; j < 4; j++) {
			uint32_t w = 0;
			for (int b = 0; b < 4; b++)
				w |= (uint32_t)data[p][4*(3 - j) + b] << (8*b);
			words[p][j] = w;
		}
}

static void burst_write(void) {
	uint8_t data[PHASES][BYTES];
	unpack(M.wrdata, data);
	for (int m = 0; m < MODULES; m++) {
		int zone = write_zone(m);
		int shift = M.weye[m].set ? (M.mod[m].wbs - M.weye[m].wbs) : 0;
		if (zone == 0 && shift == 0)
			continue;                       /* off the eye: the write lands nowhere */
		for (int k = 0; k < BEATS; k++) {
			int src = k - shift;            /* a wrong bitslip lands the burst shifted */
			uint8_t v = (src >= 0 && src < BEATS) ? *beat(data, m, src) : 0xff;
			if (zone == 2 && transitions(v) >= 6)
				v ^= 0x18;                  /* marginal: bytes with many transitions fail */
			*beat(M.mem, m, k) = v;
		}
	}
}

static const uint8_t mpr_patterns[4] = {0x55, 0x33, 0x0f, 0x00};

/* The burst a READ returns: the scratch row, or MPR location `mpr` (every DQ the same bit). */
static uint8_t read_beat(int m, int k, int mpr) {
	if (mpr < 0)
		return *beat(M.mem, m, k);
	return ((mpr_patterns[mpr] >> (7 - k)) & 1) ? 0xff : 0x00;
}

static void burst_read(int mpr) {
	uint8_t data[PHASES][BYTES];
	for (int m = 0; m < MODULES; m++) {
		int zone = read_zone(m);
		int shift = M.reye[m].set ? (M.mod[m].rbs - M.reye[m].rbs) : 1;
		for (int k = 0; k < BEATS; k++) {
			int src = k - shift;
			uint8_t v;
			if (zone != 0)
				v = read_beat(m, k, mpr);
			else if (shift != 0)            /* a wrong bitslip: beats shifted, the preamble first */
				v = (src >= 0 && src < BEATS) ? read_beat(m, src, mpr) : 0xff;
			else                            /* a wrong delay: garbled */
				v = read_beat(m, k, mpr) ^ (uint8_t)(0xa5 + 3*k + M.mod[m].rd);
			if (zone == 2 && transitions(v) >= 6)
				v ^= 0x42;                  /* not the write side's mask: no cancelling */
			*beat(data, m, k) = v;
		}
	}
	pack(data, M.rddata);
	M.checks++;
}

static void command_issue(int phase) {
	uint32_t cmd = M.command[phase];
	if (M.control & DFII_CONTROL_SEL)
		return;                             /* hardware control: the injector is not ours */
	if ((cmd & (CMD_CAS|CMD_WE|CMD_CS|CMD_WRDATA)) == (CMD_CAS|CMD_WE|CMD_CS|CMD_WRDATA))
		burst_write();
	else if ((cmd & (CMD_CAS|CMD_CS|CMD_RDDATA)) == (CMD_CAS|CMD_CS|CMD_RDDATA) && !(cmd & CMD_WE))
		burst_read(M.mpr ? (int)(M.baddress[phase] & 3) : -1);
	else if ((cmd & (CMD_RAS|CMD_CAS|CMD_WE|CMD_CS)) == (CMD_RAS|CMD_CAS|CMD_WE|CMD_CS) && M.baddress[phase] == 3)
		M.mpr = (M.address[phase] >> 2) & 1;    /* MR3: A2 = MPR operation */
	/* Activate, precharge and the other mode registers: nothing to model. */
}

/* Delay pulses --------------------------------------------------------------------------------------*/

static void pulse(uint32_t reg) {
	for (int m = 0; m < MODULES; m++) {
		struct module_state *s = &M.mod[m];
		if (!((M.dly_sel >> m) & 1))
			continue;
		switch (reg) {
		case PHY_RDLY_DQ_RST:         s->rd = 0; break;
		case PHY_RDLY_DQ_INC:         if (!M.en_vtc) s->rd = (s->rd + 1) % DELAYS; break;
		case PHY_RDLY_DQ_BITSLIP_RST: s->rbs = 0; break;
		case PHY_RDLY_DQ_BITSLIP:     s->rbs = (s->rbs + 1) % BITSLIPS; break;
		case PHY_WDLY_DQ_RST:         s->dq = 0; break;
		case PHY_WDLY_DQ_INC:         if (!M.en_vtc) s->dq = (s->dq + 1) % DELAYS; break;
		case PHY_WDLY_DQS_RST:        if (M.dqs_rst_wired) { s->dqs = 0; s->dqs_count = 0; } break;
		case PHY_WDLY_DQS_INC:
			if (!M.en_vtc) s->dqs = (s->dqs + 1) % DELAYS;
			s->dqs_count = (s->dqs_count + 1) & 0x1ff;
			break;
		case PHY_WDLY_DQ_BITSLIP_RST: s->wbs = 0; break;
		case PHY_WDLY_DQ_BITSLIP:     s->wbs = (s->wbs + 1) % BITSLIPS; break;
		default: break;
		}
	}
}

/* CSR bus -------------------------------------------------------------------------------------------*/

unsigned long csr_read_simple(unsigned long a) {
	unsigned long off = a - CSR_BASE_ADDR;
	M.reads++;
	if (off >= SDRAM_OFF && off < SDRAM_OFF + 0x100) {
		off -= SDRAM_OFF;
		if (off == SDRAM_CONTROL)
			return M.control;
		if (off >= SDRAM_PI_BASE) {
			int phase = (off - SDRAM_PI_BASE) / PI_STRIDE;
			int reg   = (off - SDRAM_PI_BASE) % PI_STRIDE;
			if (phase < PHASES) {
				if (reg == PI_COMMAND)  return M.command[phase];
				if (reg == PI_ADDRESS)  return M.address[phase];
				if (reg == PI_BADDRESS) return M.baddress[phase];
				if (reg == PI_COMMAND_ISSUE) return 0;
				if (reg >= PI_WRDATA && reg < PI_WRDATA + 16) return M.wrdata[phase][(reg - PI_WRDATA)/4];
				if (reg >= PI_RDDATA && reg < PI_RDDATA + 16) return M.rddata[phase][(reg - PI_RDDATA)/4];
			}
		}
	} else if (off >= DDRPHY_OFF && off < DDRPHY_OFF + 0x100) {
		off -= DDRPHY_OFF;
		switch (off) {
		case PHY_RST:             return M.phy_rst;
		case PHY_EN_VTC:          return M.en_vtc;
		case PHY_HALF_SYS8X_TAPS: return 128;
		case PHY_WLEVEL_EN:       return M.wlevel_en;
		case PHY_CDLY_VALUE:      return M.cdly;
		case PHY_DLY_SEL:         return M.dly_sel;
		case PHY_RDPHASE:         return M.rdphase;
		case PHY_WRPHASE:         return M.wrphase;
		case PHY_WDLY_DQS_INC_COUNT: {
			unsigned long v = 0;
			for (int m = 0; m < MODULES; m++)
				if ((M.dly_sel >> m) & 1)
					v = M.mod[m].dqs_count;
			return v;
		}
		case PHY_WLEVEL_STROBE: case PHY_CDLY_RST: case PHY_CDLY_INC:
		case PHY_RDLY_DQ_RST: case PHY_RDLY_DQ_INC: case PHY_RDLY_DQ_BITSLIP_RST: case PHY_RDLY_DQ_BITSLIP:
		case PHY_WDLY_DQ_RST: case PHY_WDLY_DQ_INC: case PHY_WDLY_DQS_RST: case PHY_WDLY_DQS_INC:
		case PHY_WDLY_DQ_BITSLIP_RST: case PHY_WDLY_DQ_BITSLIP:
			return 0;
		default: break;
		}
	}
	M.unknown++;
	return 0;
}

void csr_write_simple(unsigned long v, unsigned long a) {
	unsigned long off = a - CSR_BASE_ADDR;
	M.writes++;
	v &= 0xffffffff;
	if (off >= SDRAM_OFF && off < SDRAM_OFF + 0x100) {
		off -= SDRAM_OFF;
		if (off == SDRAM_CONTROL) { M.control = v; return; }
		if (off >= SDRAM_PI_BASE) {
			int phase = (off - SDRAM_PI_BASE) / PI_STRIDE;
			int reg   = (off - SDRAM_PI_BASE) % PI_STRIDE;
			if (phase < PHASES) {
				if (reg == PI_COMMAND)       { M.command[phase] = v; return; }
				if (reg == PI_COMMAND_ISSUE) { if (v & 1) command_issue(phase); return; }
				if (reg == PI_ADDRESS)       { M.address[phase] = v; return; }
				if (reg == PI_BADDRESS)      { M.baddress[phase] = v; return; }
				if (reg >= PI_WRDATA && reg < PI_WRDATA + 16) { M.wrdata[phase][(reg - PI_WRDATA)/4] = v; return; }
			}
		}
	} else if (off >= DDRPHY_OFF && off < DDRPHY_OFF + 0x100) {
		off -= DDRPHY_OFF;
		switch (off) {
		case PHY_RST:
			M.phy_rst = v;
			if (v & 1)
				for (int m = 0; m < MODULES; m++) {
					M.mod[m].rd = M.mod[m].dq = M.mod[m].rbs = M.mod[m].wbs = 0;
					if (M.dqs_rst_wired)
						M.mod[m].dqs = M.mod[m].dqs_count = 0;
				}
			return;
		case PHY_EN_VTC:    M.en_vtc = v & 1; return;
		case PHY_WLEVEL_EN: M.wlevel_en = v & 1; return;
		case PHY_WLEVEL_STROBE:                  /* write leveling: the DQS samples read 0 */
			memset(M.rddata[0], 0, sizeof(M.rddata[0]));
			return;
		case PHY_CDLY_RST: M.cdly = 0; return;
		case PHY_CDLY_INC: M.cdly = (M.cdly + 1) % DELAYS; return;
		case PHY_DLY_SEL:  M.dly_sel = v; return;
		case PHY_RDPHASE:  M.rdphase = v & 3; return;
		case PHY_WRPHASE:  M.wrphase = v & 3; return;
		case PHY_RDLY_DQ_RST: case PHY_RDLY_DQ_INC: case PHY_RDLY_DQ_BITSLIP_RST: case PHY_RDLY_DQ_BITSLIP:
		case PHY_WDLY_DQ_RST: case PHY_WDLY_DQ_INC: case PHY_WDLY_DQS_RST: case PHY_WDLY_DQS_INC:
		case PHY_WDLY_DQ_BITSLIP_RST: case PHY_WDLY_DQ_BITSLIP:
			if (v & 1)
				pulse(off);
			return;
		default: break;
		}
	}
	M.unknown++;
}

/* Control (ctypes) ----------------------------------------------------------------------------------*/

void model_reset(void) {
	memset(&M, 0, sizeof(M));
	M.en_vtc  = 1;                           /* the CSR's reset value */
	M.rdphase = 1;
	M.wrphase = 3;
}

void model_set_dqs_rst_wired(int wired) {
	M.dqs_rst_wired = wired;
}

void model_set_write_eye(int m, int wbs, int dqs_lo, int dqs_hi, int off_c, int off_hw, int margin) {
	struct write_eye *e = &M.weye[m];
	e->set = 1;
	e->wbs = wbs; e->dqs_lo = dqs_lo; e->dqs_hi = dqs_hi;
	e->off_c = off_c; e->off_hw = off_hw; e->margin = margin;
}

void model_set_read_eye(int m, int rbs, int c, int hw, int margin) {
	struct read_eye *e = &M.reye[m];
	e->set = 1;
	e->rbs = rbs; e->c = c; e->hw = hw; e->margin = margin;
}

void model_set_state(int m, int wbs, int dqs, int dq, int rbs, int rd) {
	struct module_state *s = &M.mod[m];
	s->wbs = wbs; s->dqs = dqs; s->dq = dq; s->rbs = rbs; s->rd = rd;
	s->dqs_count = dqs & 0x1ff;
}

void model_get_state(int m, int *out) {
	const struct module_state *s = &M.mod[m];
	out[0] = s->wbs; out[1] = s->dqs; out[2] = s->dq; out[3] = s->rbs; out[4] = s->rd;
}

/* The core eyes (no margin), at a hypothetical setting. */
int model_pass_write(int m, int wbs, int dqs, int dq) {
	struct module_state saved = M.mod[m];
	int r;
	M.mod[m].wbs = wbs; M.mod[m].dqs = dqs; M.mod[m].dq = dq;
	r = write_zone(m) != 0;
	M.mod[m] = saved;
	return r;
}

int model_pass_read(int m, int rbs, int rd) {
	struct module_state saved = M.mod[m];
	int r;
	M.mod[m].rbs = rbs; M.mod[m].rd = rd;
	r = read_zone(m) != 0;
	M.mod[m] = saved;
	return r;
}

unsigned long model_reads(void)   { return M.reads; }
unsigned long model_writes(void)  { return M.writes; }
unsigned long model_unknown(void) { return M.unknown; }
unsigned long model_checks(void)  { return M.checks; }
void model_reset_counters(void)   { M.reads = M.writes = M.unknown = M.checks = 0; }
void model_flush(void)            { fflush(stdout); }

/* libbase stubs sdram_init() links against: the memtest fails the first `fail` calls. */
void model_set_memtest_fail(int fail) { M.memtest_fail = fail; M.memtest_calls = 0; }
int  model_memtest_calls(void)        { return M.memtest_calls; }

int memtest(unsigned int *addr, unsigned long maxsize) {
	(void)addr; (void)maxsize;
	M.memtest_calls++;
	return M.memtest_calls > M.memtest_fail;
}

void memspeed(unsigned int *addr, unsigned long size, bool read_only, bool random) {
	(void)addr; (void)size; (void)read_only; (void)random;
}
