//--------------------------------------------------------------------------------
// test/ddr shim: the sdram and ddrphy CSR groups of the XCU1525 build (controller 0), in
// LiteX's generated csr.h format. The accessors resolve to the eye model's bus (model.c).
//--------------------------------------------------------------------------------

#include <generated/soc.h>
#ifndef __GENERATED_CSR_H
#define __GENERATED_CSR_H
#include <stdint.h>
#include <system.h>
#include <hw/common.h>

#ifndef CSR_BASE
#define CSR_BASE 0xf0000000L
#endif /* ! CSR_BASE */

/* SDRAM Registers */
#define CSR_SDRAM_BASE (CSR_BASE + 0x20000L)
#define CSR_SDRAM_DFII_CONTROL_ADDR (CSR_BASE + 0x20000L)
#define CSR_SDRAM_DFII_CONTROL_SIZE 1
#define CSR_SDRAM_DFII_PI0_COMMAND_ADDR (CSR_BASE + 0x20004L)
#define CSR_SDRAM_DFII_PI0_COMMAND_SIZE 1
#define CSR_SDRAM_DFII_PI0_COMMAND_ISSUE_ADDR (CSR_BASE + 0x20008L)
#define CSR_SDRAM_DFII_PI0_COMMAND_ISSUE_SIZE 1
#define CSR_SDRAM_DFII_PI0_ADDRESS_ADDR (CSR_BASE + 0x2000cL)
#define CSR_SDRAM_DFII_PI0_ADDRESS_SIZE 1
#define CSR_SDRAM_DFII_PI0_BADDRESS_ADDR (CSR_BASE + 0x20010L)
#define CSR_SDRAM_DFII_PI0_BADDRESS_SIZE 1
#define CSR_SDRAM_DFII_PI0_WRDATA_ADDR (CSR_BASE + 0x20014L)
#define CSR_SDRAM_DFII_PI0_WRDATA_SIZE 4
#define CSR_SDRAM_DFII_PI0_RDDATA_ADDR (CSR_BASE + 0x20024L)
#define CSR_SDRAM_DFII_PI0_RDDATA_SIZE 4
#define CSR_SDRAM_DFII_PI1_COMMAND_ADDR (CSR_BASE + 0x20034L)
#define CSR_SDRAM_DFII_PI1_COMMAND_SIZE 1
#define CSR_SDRAM_DFII_PI1_COMMAND_ISSUE_ADDR (CSR_BASE + 0x20038L)
#define CSR_SDRAM_DFII_PI1_COMMAND_ISSUE_SIZE 1
#define CSR_SDRAM_DFII_PI1_ADDRESS_ADDR (CSR_BASE + 0x2003cL)
#define CSR_SDRAM_DFII_PI1_ADDRESS_SIZE 1
#define CSR_SDRAM_DFII_PI1_BADDRESS_ADDR (CSR_BASE + 0x20040L)
#define CSR_SDRAM_DFII_PI1_BADDRESS_SIZE 1
#define CSR_SDRAM_DFII_PI1_WRDATA_ADDR (CSR_BASE + 0x20044L)
#define CSR_SDRAM_DFII_PI1_WRDATA_SIZE 4
#define CSR_SDRAM_DFII_PI1_RDDATA_ADDR (CSR_BASE + 0x20054L)
#define CSR_SDRAM_DFII_PI1_RDDATA_SIZE 4
#define CSR_SDRAM_DFII_PI2_COMMAND_ADDR (CSR_BASE + 0x20064L)
#define CSR_SDRAM_DFII_PI2_COMMAND_SIZE 1
#define CSR_SDRAM_DFII_PI2_COMMAND_ISSUE_ADDR (CSR_BASE + 0x20068L)
#define CSR_SDRAM_DFII_PI2_COMMAND_ISSUE_SIZE 1
#define CSR_SDRAM_DFII_PI2_ADDRESS_ADDR (CSR_BASE + 0x2006cL)
#define CSR_SDRAM_DFII_PI2_ADDRESS_SIZE 1
#define CSR_SDRAM_DFII_PI2_BADDRESS_ADDR (CSR_BASE + 0x20070L)
#define CSR_SDRAM_DFII_PI2_BADDRESS_SIZE 1
#define CSR_SDRAM_DFII_PI2_WRDATA_ADDR (CSR_BASE + 0x20074L)
#define CSR_SDRAM_DFII_PI2_WRDATA_SIZE 4
#define CSR_SDRAM_DFII_PI2_RDDATA_ADDR (CSR_BASE + 0x20084L)
#define CSR_SDRAM_DFII_PI2_RDDATA_SIZE 4
#define CSR_SDRAM_DFII_PI3_COMMAND_ADDR (CSR_BASE + 0x20094L)
#define CSR_SDRAM_DFII_PI3_COMMAND_SIZE 1
#define CSR_SDRAM_DFII_PI3_COMMAND_ISSUE_ADDR (CSR_BASE + 0x20098L)
#define CSR_SDRAM_DFII_PI3_COMMAND_ISSUE_SIZE 1
#define CSR_SDRAM_DFII_PI3_ADDRESS_ADDR (CSR_BASE + 0x2009cL)
#define CSR_SDRAM_DFII_PI3_ADDRESS_SIZE 1
#define CSR_SDRAM_DFII_PI3_BADDRESS_ADDR (CSR_BASE + 0x200a0L)
#define CSR_SDRAM_DFII_PI3_BADDRESS_SIZE 1
#define CSR_SDRAM_DFII_PI3_WRDATA_ADDR (CSR_BASE + 0x200a4L)
#define CSR_SDRAM_DFII_PI3_WRDATA_SIZE 4
#define CSR_SDRAM_DFII_PI3_RDDATA_ADDR (CSR_BASE + 0x200b4L)
#define CSR_SDRAM_DFII_PI3_RDDATA_SIZE 4

/* DDRPHY Registers */
#define CSR_DDRPHY_BASE (CSR_BASE + 0x20800L)
#define CSR_DDRPHY_RST_ADDR (CSR_BASE + 0x20800L)
#define CSR_DDRPHY_RST_SIZE 1
#define CSR_DDRPHY_EN_VTC_ADDR (CSR_BASE + 0x20804L)
#define CSR_DDRPHY_EN_VTC_SIZE 1
#define CSR_DDRPHY_HALF_SYS8X_TAPS_ADDR (CSR_BASE + 0x20808L)
#define CSR_DDRPHY_HALF_SYS8X_TAPS_SIZE 1
#define CSR_DDRPHY_WLEVEL_EN_ADDR (CSR_BASE + 0x2080cL)
#define CSR_DDRPHY_WLEVEL_EN_SIZE 1
#define CSR_DDRPHY_WLEVEL_STROBE_ADDR (CSR_BASE + 0x20810L)
#define CSR_DDRPHY_WLEVEL_STROBE_SIZE 1
#define CSR_DDRPHY_CDLY_RST_ADDR (CSR_BASE + 0x20814L)
#define CSR_DDRPHY_CDLY_RST_SIZE 1
#define CSR_DDRPHY_CDLY_INC_ADDR (CSR_BASE + 0x20818L)
#define CSR_DDRPHY_CDLY_INC_SIZE 1
#define CSR_DDRPHY_CDLY_VALUE_ADDR (CSR_BASE + 0x2081cL)
#define CSR_DDRPHY_CDLY_VALUE_SIZE 1
#define CSR_DDRPHY_DLY_SEL_ADDR (CSR_BASE + 0x20820L)
#define CSR_DDRPHY_DLY_SEL_SIZE 1
#define CSR_DDRPHY_RDLY_DQ_RST_ADDR (CSR_BASE + 0x20824L)
#define CSR_DDRPHY_RDLY_DQ_RST_SIZE 1
#define CSR_DDRPHY_RDLY_DQ_INC_ADDR (CSR_BASE + 0x20828L)
#define CSR_DDRPHY_RDLY_DQ_INC_SIZE 1
#define CSR_DDRPHY_RDLY_DQ_BITSLIP_RST_ADDR (CSR_BASE + 0x2082cL)
#define CSR_DDRPHY_RDLY_DQ_BITSLIP_RST_SIZE 1
#define CSR_DDRPHY_RDLY_DQ_BITSLIP_ADDR (CSR_BASE + 0x20830L)
#define CSR_DDRPHY_RDLY_DQ_BITSLIP_SIZE 1
#define CSR_DDRPHY_WDLY_DQ_RST_ADDR (CSR_BASE + 0x20834L)
#define CSR_DDRPHY_WDLY_DQ_RST_SIZE 1
#define CSR_DDRPHY_WDLY_DQ_INC_ADDR (CSR_BASE + 0x20838L)
#define CSR_DDRPHY_WDLY_DQ_INC_SIZE 1
#define CSR_DDRPHY_WDLY_DQS_RST_ADDR (CSR_BASE + 0x2083cL)
#define CSR_DDRPHY_WDLY_DQS_RST_SIZE 1
#define CSR_DDRPHY_WDLY_DQS_INC_ADDR (CSR_BASE + 0x20840L)
#define CSR_DDRPHY_WDLY_DQS_INC_SIZE 1
#define CSR_DDRPHY_WDLY_DQS_INC_COUNT_ADDR (CSR_BASE + 0x20844L)
#define CSR_DDRPHY_WDLY_DQS_INC_COUNT_SIZE 1
#define CSR_DDRPHY_WDLY_DQ_BITSLIP_RST_ADDR (CSR_BASE + 0x20848L)
#define CSR_DDRPHY_WDLY_DQ_BITSLIP_RST_SIZE 1
#define CSR_DDRPHY_WDLY_DQ_BITSLIP_ADDR (CSR_BASE + 0x2084cL)
#define CSR_DDRPHY_WDLY_DQ_BITSLIP_SIZE 1
#define CSR_DDRPHY_RDPHASE_ADDR (CSR_BASE + 0x20850L)
#define CSR_DDRPHY_RDPHASE_SIZE 1
#define CSR_DDRPHY_WRPHASE_ADDR (CSR_BASE + 0x20854L)
#define CSR_DDRPHY_WRPHASE_SIZE 1

static inline uint32_t sdram_dfii_control_read(void) {
	return csr_read_simple((CSR_BASE + 0x20000L));
}
static inline void sdram_dfii_control_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20000L));
}
static inline uint32_t sdram_dfii_pi0_command_read(void) {
	return csr_read_simple((CSR_BASE + 0x20004L));
}
static inline void sdram_dfii_pi0_command_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20004L));
}
static inline uint32_t sdram_dfii_pi0_command_issue_read(void) {
	return csr_read_simple((CSR_BASE + 0x20008L));
}
static inline void sdram_dfii_pi0_command_issue_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20008L));
}
static inline uint32_t sdram_dfii_pi0_address_read(void) {
	return csr_read_simple((CSR_BASE + 0x2000cL));
}
static inline void sdram_dfii_pi0_address_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x2000cL));
}
static inline uint32_t sdram_dfii_pi0_baddress_read(void) {
	return csr_read_simple((CSR_BASE + 0x20010L));
}
static inline void sdram_dfii_pi0_baddress_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20010L));
}
static inline uint32_t sdram_dfii_pi1_command_read(void) {
	return csr_read_simple((CSR_BASE + 0x20034L));
}
static inline void sdram_dfii_pi1_command_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20034L));
}
static inline uint32_t sdram_dfii_pi1_command_issue_read(void) {
	return csr_read_simple((CSR_BASE + 0x20038L));
}
static inline void sdram_dfii_pi1_command_issue_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20038L));
}
static inline uint32_t sdram_dfii_pi1_address_read(void) {
	return csr_read_simple((CSR_BASE + 0x2003cL));
}
static inline void sdram_dfii_pi1_address_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x2003cL));
}
static inline uint32_t sdram_dfii_pi1_baddress_read(void) {
	return csr_read_simple((CSR_BASE + 0x20040L));
}
static inline void sdram_dfii_pi1_baddress_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20040L));
}
static inline uint32_t sdram_dfii_pi2_command_read(void) {
	return csr_read_simple((CSR_BASE + 0x20064L));
}
static inline void sdram_dfii_pi2_command_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20064L));
}
static inline uint32_t sdram_dfii_pi2_command_issue_read(void) {
	return csr_read_simple((CSR_BASE + 0x20068L));
}
static inline void sdram_dfii_pi2_command_issue_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20068L));
}
static inline uint32_t sdram_dfii_pi2_address_read(void) {
	return csr_read_simple((CSR_BASE + 0x2006cL));
}
static inline void sdram_dfii_pi2_address_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x2006cL));
}
static inline uint32_t sdram_dfii_pi2_baddress_read(void) {
	return csr_read_simple((CSR_BASE + 0x20070L));
}
static inline void sdram_dfii_pi2_baddress_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20070L));
}
static inline uint32_t sdram_dfii_pi3_command_read(void) {
	return csr_read_simple((CSR_BASE + 0x20094L));
}
static inline void sdram_dfii_pi3_command_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20094L));
}
static inline uint32_t sdram_dfii_pi3_command_issue_read(void) {
	return csr_read_simple((CSR_BASE + 0x20098L));
}
static inline void sdram_dfii_pi3_command_issue_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20098L));
}
static inline uint32_t sdram_dfii_pi3_address_read(void) {
	return csr_read_simple((CSR_BASE + 0x2009cL));
}
static inline void sdram_dfii_pi3_address_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x2009cL));
}
static inline uint32_t sdram_dfii_pi3_baddress_read(void) {
	return csr_read_simple((CSR_BASE + 0x200a0L));
}
static inline void sdram_dfii_pi3_baddress_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x200a0L));
}
static inline uint32_t ddrphy_rst_read(void) {
	return csr_read_simple((CSR_BASE + 0x20800L));
}
static inline void ddrphy_rst_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20800L));
}
static inline uint32_t ddrphy_en_vtc_read(void) {
	return csr_read_simple((CSR_BASE + 0x20804L));
}
static inline void ddrphy_en_vtc_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20804L));
}
static inline uint32_t ddrphy_half_sys8x_taps_read(void) {
	return csr_read_simple((CSR_BASE + 0x20808L));
}
static inline void ddrphy_half_sys8x_taps_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20808L));
}
static inline uint32_t ddrphy_wlevel_en_read(void) {
	return csr_read_simple((CSR_BASE + 0x2080cL));
}
static inline void ddrphy_wlevel_en_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x2080cL));
}
static inline uint32_t ddrphy_wlevel_strobe_read(void) {
	return csr_read_simple((CSR_BASE + 0x20810L));
}
static inline void ddrphy_wlevel_strobe_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20810L));
}
static inline uint32_t ddrphy_cdly_rst_read(void) {
	return csr_read_simple((CSR_BASE + 0x20814L));
}
static inline void ddrphy_cdly_rst_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20814L));
}
static inline uint32_t ddrphy_cdly_inc_read(void) {
	return csr_read_simple((CSR_BASE + 0x20818L));
}
static inline void ddrphy_cdly_inc_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20818L));
}
static inline uint32_t ddrphy_cdly_value_read(void) {
	return csr_read_simple((CSR_BASE + 0x2081cL));
}
static inline void ddrphy_cdly_value_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x2081cL));
}
static inline uint32_t ddrphy_dly_sel_read(void) {
	return csr_read_simple((CSR_BASE + 0x20820L));
}
static inline void ddrphy_dly_sel_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20820L));
}
static inline uint32_t ddrphy_rdly_dq_rst_read(void) {
	return csr_read_simple((CSR_BASE + 0x20824L));
}
static inline void ddrphy_rdly_dq_rst_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20824L));
}
static inline uint32_t ddrphy_rdly_dq_inc_read(void) {
	return csr_read_simple((CSR_BASE + 0x20828L));
}
static inline void ddrphy_rdly_dq_inc_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20828L));
}
static inline uint32_t ddrphy_rdly_dq_bitslip_rst_read(void) {
	return csr_read_simple((CSR_BASE + 0x2082cL));
}
static inline void ddrphy_rdly_dq_bitslip_rst_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x2082cL));
}
static inline uint32_t ddrphy_rdly_dq_bitslip_read(void) {
	return csr_read_simple((CSR_BASE + 0x20830L));
}
static inline void ddrphy_rdly_dq_bitslip_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20830L));
}
static inline uint32_t ddrphy_wdly_dq_rst_read(void) {
	return csr_read_simple((CSR_BASE + 0x20834L));
}
static inline void ddrphy_wdly_dq_rst_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20834L));
}
static inline uint32_t ddrphy_wdly_dq_inc_read(void) {
	return csr_read_simple((CSR_BASE + 0x20838L));
}
static inline void ddrphy_wdly_dq_inc_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20838L));
}
static inline uint32_t ddrphy_wdly_dqs_rst_read(void) {
	return csr_read_simple((CSR_BASE + 0x2083cL));
}
static inline void ddrphy_wdly_dqs_rst_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x2083cL));
}
static inline uint32_t ddrphy_wdly_dqs_inc_read(void) {
	return csr_read_simple((CSR_BASE + 0x20840L));
}
static inline void ddrphy_wdly_dqs_inc_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20840L));
}
static inline uint32_t ddrphy_wdly_dqs_inc_count_read(void) {
	return csr_read_simple((CSR_BASE + 0x20844L));
}
static inline void ddrphy_wdly_dqs_inc_count_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20844L));
}
static inline uint32_t ddrphy_wdly_dq_bitslip_rst_read(void) {
	return csr_read_simple((CSR_BASE + 0x20848L));
}
static inline void ddrphy_wdly_dq_bitslip_rst_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20848L));
}
static inline uint32_t ddrphy_wdly_dq_bitslip_read(void) {
	return csr_read_simple((CSR_BASE + 0x2084cL));
}
static inline void ddrphy_wdly_dq_bitslip_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x2084cL));
}
static inline uint32_t ddrphy_rdphase_read(void) {
	return csr_read_simple((CSR_BASE + 0x20850L));
}
static inline void ddrphy_rdphase_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20850L));
}
static inline uint32_t ddrphy_wrphase_read(void) {
	return csr_read_simple((CSR_BASE + 0x20854L));
}
static inline void ddrphy_wrphase_write(uint32_t v) {
	csr_write_simple(v, (CSR_BASE + 0x20854L));
}

#endif /* ! __GENERATED_CSR_H */
