# SPDX-License-Identifier: BSD-2-Clause
"""Target-side physical policy for the shared cluster on Xilinx UltraScale+.

These are the existing constraints, isolated from portable compute integration.
They are NOT a timing waiver: new integrations need routed timing and selector
coverage checks. In particular a signal-name refactor is not physically neutral.
"""


def add_cluster_constraints(platform):
    # engreset holds reset for 64 sys cycles. Preserve setup=4 / hold=3 for the
    # reset launch net; firmware cannot submit during the reset/drain handshake.
    platform.add_platform_command(
        "set_multicycle_path -setup 4 -from "
        "[get_cells -hierarchical -filter {{NAME =~ *eng_rst_reg*}}]")
    platform.add_platform_command(
        "set_multicycle_path -hold 3 -from "
        "[get_cells -hierarchical -filter {{NAME =~ *eng_rst_reg*}}]")
    # The 2 Mbit shared scratchpad and the norm unit's weight buffer in BLOCK RAM. Left to inference, part of the scratchpad went to LUT RAM: every
    # client address then fans out to thousands of LUT-RAM cells across the die, and v15
    # (more clients, more area) placed those paths at -4 ns (route 9.2 ns of 11).
    for name in ("bonsai_scratchpad", "norm_wbuf"):
        platform.add_platform_command(
            "set_property RAM_STYLE block [get_cells -quiet -hierarchical -filter "
            "{{NAME =~ *" + name + "*_reg*}}]")
    # Preserve distributed vector imem: a BRAM output register changes the
    # decode launch's clock-to-out and placement. No vector multicycle paths.
    platform.add_platform_command(
        "set_property RAM_STYLE distributed "
        "[get_cells -hierarchical -filter {{NAME =~ *imem_reg*}}]")
