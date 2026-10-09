#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""The test tree's conftest: --runslow and --waves (pytest.ini puts the repository and test/ on
sys.path)."""

import os

import pytest

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--waves",   action="store_true", default=False,
                     help="dump waveforms to build/traces/<test id> (as TRIXIUM_TRACE=1)")

def pytest_configure(config):
    # Waveforms only on request (--waves: pytest owns --trace). common.verilator then builds
    # with --trace-fst (cached apart) and its Script/Migen runs dump one FST/VCD per run.
    if config.getoption("--waves"):
        os.environ["TRIXIUM_TRACE"] = "1"

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
