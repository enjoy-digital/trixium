//
// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

// Default driver of common.verilator: <binary> <config> <script> <outdir>.
//
// The config sets up the run before the script starts, one item per line:
//   cycles N                         cycle cap
//   axi PREFIX key=value...          an AxiSlave on the PREFIX_* ports (see AxiConfig)
//   image MEM FILE [ADDR]            load FILE into memory MEM (DUT memories start zeroed)

#include <sstream>

#include "sim_top.h"
#include "axi_slave.h"
#include "script.h"

int main(int argc, char **argv) {
    sim_require(argc == 4, "usage: config script outdir");
    VerilatedContext ctx;
    SIM_TOP d(&ctx);
    Sim sim;
    sim.ctx = &ctx;
    sim_bind(sim, d);
    for (auto &m : sim.mems) {   // DUT memories: start from zeros, not X.
        std::vector<uint8_t> z(m.second.size, 0);
        m.second.write(0, z.data(), z.size());
    }

    std::ifstream cfg(argv[1]);
    if (!cfg) sim_fail("cannot open config %s", argv[1]);
    std::string line;
    std::vector<Model *> owned;
    while (std::getline(cfg, line)) {
        std::istringstream in(line);
        std::string op;
        if (!(in >> op) || op[0] == '#') continue;
        if (op == "cycles") {
            in >> sim.max_cycles;
        } else if (op == "axi") {
            std::string prefix, kv;
            in >> prefix;
            AxiConfig c;
            while (in >> kv) c.option(prefix, kv);
            auto *a = new AxiSlave(sim, prefix, c);
            sim.models.push_back(a);
            owned.push_back(a);
        } else if (op == "image") {
            std::string mem, file; size_t addr = 0;
            in >> mem >> file >> addr;
            std::vector<uint8_t> data = sim_load(file);
            Mem &m = sim.mem(mem);
            if (addr + data.size() > m.size)
                sim_fail("image %s: %zu bytes at %zu exceed %zu", mem.c_str(), data.size(), addr, m.size);
            m.write(addr, data.data(), data.size());
        } else {
            sim_fail("unknown config item '%s'", op.c_str());
        }
    }

    run_script(sim, argv[2], argv[3]);
    for (auto *m : sim.models) m->report(sim);
    sim.finish();
    d.final();
    for (auto *m : owned) delete m;
    return 0;
}
