#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

from setuptools import setup, find_packages

# The gateware needs Migen/LiteX from the LiteX installer (litex_setup.py), not from PyPI. Install
# editable (pip install -e .): the tools use the tree (firmware/, software/user/, models/, build/).

setup(
    name             = "trixium",
    version          = "2026.10",
    description      = "Ternary LLMs on FPGA: PrismML's Bonsai models decoded entirely on the card, open source end to end with LiteX.",
    long_description = open("README.md", encoding="utf-8").read(),
    long_description_content_type = "text/markdown",
    author           = "Enjoy-Digital",
    url              = "https://github.com/enjoy-digital/trixium",
    license          = "BSD-2-Clause",
    python_requires  = ">=3.10",
    install_requires = ["numpy", "tokenizers", "pyserial", "huggingface_hub"],
    extras_require   = {"dev": ["pytest", "pytest-xdist", "pexpect"]},
    py_modules       = ["trixium_xcu1525", "trixium_fk33", "trixium_ypcb", "trixium_sim"],
    packages         = find_packages(exclude=["test*"]),
    package_data     = {"gateware": ["memory/*.xci"]},
    entry_points     = {"console_scripts": ["trixium=software.cli:main"]},
)
