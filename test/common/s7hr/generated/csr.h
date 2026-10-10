/* SPDX-License-Identifier: BSD-2-Clause
 *
 * Host test of firmware/ddr_s7hr.c: the CSR accessors it uses, served by the fake board
 * (s7hr_fake.c) instead of a LiteX csr.h.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef S7HR_FAKE_CSR_H
#define S7HR_FAKE_CSR_H

#include <stdint.h>

#define CSR_DDRPHY_BASE                  0xf0021800UL
#define CSR_DDRPHY_CKPHASE_STATUS_VALUE_OFFSET  0
#define CSR_DDRPHY_CKPHASE_STATUS_BUSY_OFFSET   16
#define CSR_DDRPHY_CKPHASE_STATUS_LOCKED_OFFSET 17

#define FAKE_WRDATA_BASE 0x1000UL
#define FAKE_RDDATA_BASE 0x2000UL

void     csr_write_simple(uint32_t value, unsigned long addr);
uint32_t csr_read_simple(unsigned long addr);

void     ddrphy_rst_write(uint32_t v);
void     ddrphy_dly_sel_write(uint32_t v);
void     ddrphy_dq_dly_sel_write(uint32_t v);
void     ddrphy_rdly_dq_rst_write(uint32_t v);
void     ddrphy_rdly_dq_inc_write(uint32_t v);
void     ddrphy_rdly_dq_bitslip_rst_write(uint32_t v);
void     ddrphy_rdly_dq_bitslip_write(uint32_t v);
void     ddrphy_wdly_dq_bitslip_rst_write(uint32_t v);
void     ddrphy_wdly_dq_bitslip_write(uint32_t v);
uint32_t ddrphy_rdphase_read(void);
uint32_t ddrphy_wrphase_read(void);
void     ddrphy_ckphase_inc_write(uint32_t v);
void     ddrphy_ckphase_dec_write(uint32_t v);
uint32_t ddrphy_ckphase_status_read(void);

void     sdram_dfii_control_write(uint32_t v);
void     sdram_dfii_pi_address_write(int phase, uint32_t v);
void     sdram_dfii_pi_baddress_write(int phase, uint32_t v);
void     sdram_dfii_pi_command_issue(int phase, uint32_t cmd);
#define sdram_dfii_pi0_address_write(v)  sdram_dfii_pi_address_write(0, v)
#define sdram_dfii_pi1_address_write(v)  sdram_dfii_pi_address_write(1, v)
#define sdram_dfii_pi2_address_write(v)  sdram_dfii_pi_address_write(2, v)
#define sdram_dfii_pi3_address_write(v)  sdram_dfii_pi_address_write(3, v)
#define sdram_dfii_pi0_baddress_write(v) sdram_dfii_pi_baddress_write(0, v)
#define sdram_dfii_pi1_baddress_write(v) sdram_dfii_pi_baddress_write(1, v)
#define sdram_dfii_pi2_baddress_write(v) sdram_dfii_pi_baddress_write(2, v)
#define sdram_dfii_pi3_baddress_write(v) sdram_dfii_pi_baddress_write(3, v)

void     sdram_generator_reset_write(uint32_t v);
void     sdram_generator_random_write(uint32_t v);
void     sdram_generator_base_write(uint32_t v);
void     sdram_generator_end_write(uint32_t v);
void     sdram_generator_length_write(uint32_t v);
void     sdram_generator_start_write(uint32_t v);
uint32_t sdram_generator_done_read(void);
void     sdram_checker_reset_write(uint32_t v);
void     sdram_checker_random_write(uint32_t v);
void     sdram_checker_base_write(uint32_t v);
void     sdram_checker_end_write(uint32_t v);
void     sdram_checker_length_write(uint32_t v);
void     sdram_checker_start_write(uint32_t v);
uint32_t sdram_checker_done_read(void);
uint32_t sdram_checker_errors_read(void);

#endif
