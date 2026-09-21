# SPDX-License-Identifier: BSD-2-Clause
"""Target-side physical policy for the shared cluster on Xilinx UltraScale+.

These are the existing constraints, isolated from portable compute integration.
They are NOT a timing waiver: new integrations need routed timing and selector
coverage checks. In particular a signal-name refactor is not physically neutral.
"""


def add_cluster_constraints(platform, with_vecunit=True):
    # engreset holds reset for 64 sys cycles. Preserve setup=4 / hold=3 for the
    # reset launch net; firmware cannot submit during the reset/drain handshake.
    platform.add_platform_command(
        "set_multicycle_path -setup 4 -from "
        "[get_cells -hierarchical -filter {{NAME =~ *eng_rst_reg*}}]")
    platform.add_platform_command(
        "set_multicycle_path -hold 3 -from "
        "[get_cells -hierarchical -filter {{NAME =~ *eng_rst_reg*}}]")
    if with_vecunit:
        # Preserve distributed vector imem: a BRAM output register changes the
        # decode launch's clock-to-out and placement. No vector multicycle paths.
        platform.add_platform_command(
            "set_property RAM_STYLE distributed "
            "[get_cells -hierarchical -filter {{NAME =~ *imem_reg*}}]")
