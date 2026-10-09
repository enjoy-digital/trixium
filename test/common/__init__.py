#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""What the test files share: the tree's paths and a way to reuse another test file's oracle.

test/ mirrors the source tree (test/gateware/units/test_silu.py tests gateware/units/silu.py);
the testbench pieces several files use live here: rtl_client.py (compiled scratchpad
clients), cluster.py (the shared integration harness), engine.py (engine/vector-unit CSR
helpers), the C oracles, rtl/ (Verilator harnesses), data/ (archived traces and logs).
pytest.ini puts the repository and test/ on sys.path, so `from common import ...` works from
any directory.
"""

import sys
import importlib.util
from pathlib import Path

# Paths --------------------------------------------------------------------------------------------

COMMON  = Path(__file__).resolve().parent   # test/common: the C oracles.
TEST    = COMMON.parent                     # test/
ROOT    = TEST.parent                       # The repository.
DATA    = COMMON / "data"                   # Archived traces and benchmark logs.
RTL     = COMMON / "rtl"                    # Verilator C++ harnesses.
VECTORS = COMMON / "vectors"

# Helpers ------------------------------------------------------------------------------------------

def import_test(path):
    """The test module at `path` (relative to test/), for its oracle.

    pytest imports test files by basename (no __init__.py in the tree: the basenames are
    unique), so the module already collected in this session is reused, never imported twice.
    """
    path = (TEST / path).resolve()
    name = path.stem
    module = sys.modules.get(name)
    if module is not None:
        if Path(module.__file__).resolve() != path:
            raise ImportError(f"{name} is already imported from {module.__file__}")
        return module
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module
