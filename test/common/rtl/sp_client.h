//
// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

// Scratchpad model for a resident scratchpad client (the sp_rd_* / sp_wr_* port every unit
// behind the ScratchpadArbiter exposes): reads answered one clock later (0xA5A5A5A5 on the
// lanes when no read was issued, so a client must not use stale data), writes granted at
// random (grant_pct percent of the cycles), and a write payload must hold while ungranted.
// The memory is a byte image of 8-lane (256-bit) beats, registered as `name`.

#pragma once

#include "sim.h"

struct SpClient : Model {
    std::vector<uint8_t> mem;
    uint32_t  state;         // rtl_client's LCG: the grant sequence of the archived runs.
    uint32_t  grant_pct;
    size_t    beats;
    Sig *rd_req, *rd_adr, *rd_dat, *wr_req, *wr_adr, *wr_dat, *wr_we, *wr_gnt;
    uint32_t  pending[8] = {0};
    bool      rd_valid = false, held = false, grant = true;
    uint32_t  ha = 0, hw = 0, hd[8] = {0};
    uint64_t  rolled = ~0ull;
    bool      force_grant = false; // Reset/cancel cycles: always grant.

    SpClient(Sim &sim, const std::string &name, size_t size, uint32_t grant_pct, uint32_t seed = 777)
        : state(seed), grant_pct(grant_pct) {
        mem.assign(size, 0);
        beats  = size/32;
        rd_req = &sim.sig("sp_rd_req"); rd_adr = &sim.sig("sp_rd_adr"); rd_dat = &sim.sig("sp_rd_dat");
        wr_req = &sim.sig("sp_wr_req"); wr_adr = &sim.sig("sp_wr_adr"); wr_dat = &sim.sig("sp_wr_dat");
        wr_we  = &sim.sig("sp_wr_we");  wr_gnt = &sim.sig("sp_wr_gnt");
        sim.add_memory(sim_vector_memory(name, &mem));
    }

    uint32_t *beat(size_t a) { return reinterpret_cast<uint32_t *>(mem.data()) + 8*a; }

    void drive(Sim &sim) override {
        if (rolled != sim.cycle) {
            rolled = sim.cycle;
            if (force_grant) grant = true;
            else { state = state*1664525u + 1013904223u; grant = ((state >> 8) % 100) < grant_pct; }
        }
        wr_gnt->set(grant);
        uint32_t lanes[8];
        for (int l = 0; l < 8; l++) lanes[l] = rd_valid ? pending[l] : 0xA5A5A5A5u;
        rd_dat->set_words(lanes);
    }

    void sample(Sim &) override {
        uint32_t dat[8];
        wr_dat->get_words(dat);
        bool req = wr_req->get();
        uint32_t adr = (uint32_t)wr_adr->get(), we = (uint32_t)wr_we->get();
        if (held) sim_require(req && adr == ha && we == hw && !std::memcmp(hd, dat, 32),
                              "write payload changed while ungranted");
        if (req && grant) {
            sim_require(adr < beats, "write out of range");
            for (int l = 0; l < 8; l++) if (we >> l & 1) beat(adr)[l] = dat[l];
        }
        held = req && !grant; ha = adr; hw = we; std::memcpy(hd, dat, 32);
        rd_valid = rd_req->get();
        if (rd_valid) {
            uint32_t a = (uint32_t)rd_adr->get();
            sim_require(a < beats, "read out of range");
            std::memcpy(pending, beat(a), 32);
        }
    }
};
