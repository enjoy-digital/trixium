//
// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

// Co-simulation library: the compiled DUT as a shared object that common.verilator's Compiled
// drives from Python (ctypes), so a Migen testbench generator runs unchanged on the Verilated
// model. Signals and memories by handle; AXI slaves (axi_slave.h) attached from a config text
// ("axi PREFIX key=value...", one per line). Errors (a model's protocol check, a cycle cap)
// are returned, never exit(): the caller raises them.

#include <sstream>

#include "sim_top.h"
#include "axi_slave.h"

// Cosim --------------------------------------------------------------------------------------------

struct Cosim {
    VerilatedContext     ctx;
    SIM_TOP             *top = nullptr;
    Sim                  sim;
    std::vector<Model *> owned;
    std::vector<Sig *>   sigs;
    std::vector<Mem *>   mems;
    std::string          error;
};

template <typename F> static int guarded(Cosim *c, F f) {
    try {
        f();
        return 0;
    } catch (const std::exception &e) {
        c->error = e.what();
        return -1;
    }
}

// API ----------------------------------------------------------------------------------------------

extern "C" {

const char *cosim_error(Cosim *c) { return c->error.c_str(); }

Cosim *cosim_new(const char *config) {
    sim_throws = true;
    Cosim *c = new Cosim;
    c->top = new SIM_TOP(&c->ctx);
    c->sim.ctx = &c->ctx;
    c->sim.max_cycles = ~0ull;
    sim_bind(c->sim, *c->top);
    c->top->eval();  // Initial blocks ($readmemh of the .init files): the caller's cwd is the build.
    guarded(c, [&]() {
        std::istringstream cfg(config);
        std::string line;
        while (std::getline(cfg, line)) {
            std::istringstream in(line);
            std::string op;
            if (!(in >> op) || op[0] == '#') continue;
            if (op == "cycles") {
                in >> c->sim.max_cycles;
            } else if (op == "axi") {
                std::string prefix, kv;
                in >> prefix;
                AxiConfig a;
                while (in >> kv) a.option(prefix, kv);
                auto *m = new AxiSlave(c->sim, prefix, a);
                c->sim.models.push_back(m);
                c->owned.push_back(m);
            } else {
                sim_fail("unknown config item '%s'", op.c_str());
            }
        }
    });
    return c;
}

void cosim_free(Cosim *c) {
    c->sim.finish();
    c->top->final();
    for (auto *m : c->owned) delete m;
    delete c->top;
    delete c;
}

// Signals: a handle (>= 0) by name, -1 when the DUT has no such signal.
int cosim_signal(Cosim *c, const char *name) {
    Sig *s = c->sim.find(name);
    if (!s) return -1;
    c->sigs.push_back(s);
    return (int)c->sigs.size() - 1;
}

int cosim_width(Cosim *c, int h)    { return c->sigs[h]->width; }
int cosim_is_input(Cosim *c, int h) { return c->sigs[h]->input; }

int cosim_get(Cosim *c, int h, uint32_t *words) {
    return guarded(c, [&]() {
        if (!c->sim.settled) c->sim.settle();
        c->sigs[h]->get_words(words);
    });
}

void cosim_set(Cosim *c, int h, const uint32_t *words) {
    c->sigs[h]->set_words(words);
    c->sim.settled = false;
}

// Memories: byte-addressed views (DUT memories, AXI slave memories).
int cosim_memory(Cosim *c, const char *name) {
    auto it = c->sim.mems.find(name);
    if (it == c->sim.mems.end()) return -1;
    c->mems.push_back(&it->second);
    return (int)c->mems.size() - 1;
}

unsigned long long cosim_memory_size(Cosim *c, int h) { return c->mems[h]->size; }

int cosim_memory_access(Cosim *c, int h, unsigned long long addr, uint8_t *data, unsigned long long len,
                        int write) {
    return guarded(c, [&]() {
        Mem *m = c->mems[h];
        if (addr + len > m->size)
            sim_fail("memory %s: [%llu, %llu) beyond %zu bytes", m->name.c_str(), addr, addr + len, m->size);
        if (write) {
            m->write(addr, data, len);
            c->sim.settled = false;
        } else {
            if (!c->sim.settled) c->sim.settle();
            m->read(addr, data, len);
        }
    });
}

// Clock: n rising edges (models sample and answer each one).
int cosim_tick(Cosim *c, unsigned long long n) {
    return guarded(c, [&]() { for (unsigned long long i = 0; i < n; i++) c->sim.tick(); });
}

unsigned long long cosim_cycle(Cosim *c) { return c->sim.cycle; }

}
