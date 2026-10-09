#!/usr/bin/env python3

#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

from setuptools import setup, find_packages

# The gateware needs Migen/LiteX from the LiteX installer (litex_setup.py), not from PyPI.

setup(
    name             = "trixium",
    version          = "2026.10",
    description      = "PrismML Bonsai ternary LLMs running entirely on FPGA cards (XCU1525, FK33): open-source, bit-exact, built with LiteX.",
    long_description = open("README.md", encoding="utf-8").read(),
    long_description_content_type = "text/markdown",
    author           = "Enjoy-Digital",
    license          = "BSD-2-Clause",
    python_requires  = ">=3.10",
    install_requires = ["numpy", "tokenizers", "pyserial"],
    extras_require   = {"dev": ["pytest", "pytest-xdist", "huggingface_hub", "pexpect"]},
    py_modules       = ["trixium_xcu1525", "trixium_fk33", "trixium_sim"],
    packages         = find_packages(exclude=["test*"]),
    package_data     = {"software": ["lib/*.c", "lib/*.h"], "gateware": ["hbm/*.xci"]},
    entry_points     = {"console_scripts": ["trixium=software.cli:main"]},
)
