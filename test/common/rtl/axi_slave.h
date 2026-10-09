//
// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

// AXI4 slave memory model for compiled testbenches (C++ counterpart of
// gateware/memory/hbm_model.py's HBMChannelModel, plus random stalls on every channel).
//
// Ports are found by name: <prefix>_{ar,r,aw,w,b}_<field> (verilator.py's axi_ios()). INCR
// bursts, addresses modulo the memory size (a window, as the Migen model's Memory port
// truncates them). Read service:
//   - in order (reorder = 0): one burst at a time, `latency` cycles from an idle queue,
//     queued bursts chained back to back;
//   - reorder = N: the HBM controller's behaviour - same-ID bursts strictly serialized
//     (`latency` cooldown after each), different IDs overlapped, responses in rotating
//     service order.
// Write service: AW then W ("aw_first"), W ahead of AW ("w_first", a 64-beat W FIFO) or
// AWREADY only once WVALID shows ("aw_after_w"); one B per burst. Every stall is a percent
// per cycle (AR/AW/W ready dropped, R/B valid withheld). Checked: master payloads hold
// while ungranted, WLAST on the burst's last beat, AxLEN within max_len (16 = the HBM IP's
// AXI3 4-bit AxLEN). Handshakes can be logged ("@axi <prefix> ar|aw <cycle> <addr> <len> <id>").
// <prefix>_freeze, a testbench-driven control (HBMChannelModel.freeze): while it is 1 the read
// server delivers nothing, a port that never answers.

#pragma once

#include <deque>

#include "sim.h"

struct AxiConfig {
    size_t   size        = 64*1024;
    uint32_t latency     = 16;
    uint32_t r_stall     = 0;   // Percent.
    uint32_t ar_stall    = 0;
    uint32_t aw_stall    = 0;
    uint32_t w_stall     = 0;
    uint32_t b_stall     = 0;
    uint32_t reorder     = 0;   // 0, or a power of two (ID queues).
    uint32_t max_len     = 256; // Beats per burst.
    uint32_t seed        = 1;
    bool     log         = false;
    std::string write_order = "aw_first";

    // One "key=value" option (the config line of script_main.cpp, cosim.cpp).
    void option(const std::string &prefix, const std::string &kv) {
        size_t k = kv.find('=');
        if (k == std::string::npos) sim_fail("axi %s: bad option '%s'", prefix.c_str(), kv.c_str());
        std::string key = kv.substr(0, k), val = kv.substr(k + 1);
        auto num = [&]() { return (uint32_t)std::strtoul(val.c_str(), nullptr, 0); };
        if      (key == "size")        size        = std::strtoull(val.c_str(), nullptr, 0);
        else if (key == "latency")     latency     = num();
        else if (key == "r_stall")     r_stall     = num();
        else if (key == "ar_stall")    ar_stall    = num();
        else if (key == "aw_stall")    aw_stall    = num();
        else if (key == "w_stall")     w_stall     = num();
        else if (key == "b_stall")     b_stall     = num();
        else if (key == "reorder")     reorder     = num();
        else if (key == "max_len")     max_len     = num();
        else if (key == "seed")        seed        = num();
        else if (key == "log")         log         = num() != 0;
        else if (key == "write_order") write_order = val;
        else sim_fail("axi %s: unknown option '%s'", prefix.c_str(), key.c_str());
    }
};

struct AxiSlave : Model {
    struct Burst { uint64_t addr; uint32_t len; uint32_t id; };

    std::string          prefix;
    AxiConfig            cfg;
    std::vector<uint8_t> mem;
    SimRandom            rnd;
    uint32_t             beat_bytes;
    uint8_t              freeze = 0;  // <prefix>_freeze: withhold every read beat.

    // Ports.
    Sig *ar_valid, *ar_ready, *ar_addr, *ar_len, *ar_id;
    Sig *r_valid, *r_ready, *r_data, *r_id, *r_last, *r_resp;
    Sig *aw_valid, *aw_ready, *aw_addr, *aw_len, *aw_id;
    Sig *w_valid, *w_ready, *w_data, *w_strb, *w_last;
    Sig *b_valid, *b_ready, *b_id, *b_resp;

    // Per-cycle stall decisions (rolled once per cycle).
    uint64_t rolled = ~0ull;
    bool st_ar, st_r, st_aw, st_w, st_b;

    // Read: in-order server.
    std::deque<Burst> arq;
    enum { R_IDLE, R_LAT, R_STREAM } rstate = R_IDLE;
    Burst    rcur{};
    uint32_t rbeat = 0, rlat = 0;

    // Read: reorder server.
    std::vector<std::deque<Burst>> qs;
    std::vector<bool>     st_valid;
    std::vector<Burst>    st_burst;
    std::vector<uint32_t> st_lat, cooldown;
    bool     cur_valid = false;
    uint32_t qp = 0;

    // Write.
    std::deque<Burst> awq;
    std::deque<std::pair<std::vector<uint32_t>, std::pair<uint64_t, bool>>> wq; // w_first FIFO: data, strb, last.
    enum { W_IDLE, W_STREAM, W_RESP } wstate = W_IDLE;
    Burst    wcur{};
    uint32_t wbeat = 0;

    // Hold checks (payload captured while valid & ~ready).
    bool ar_held = false, aw_held = false, w_held = false;
    uint64_t ar_h[3], aw_h[3];
    std::vector<uint32_t> w_h;

    // Statistics.
    uint64_t n_ar = 0, n_aw = 0, n_r = 0, n_w = 0;

    AxiSlave(Sim &sim, const std::string &prefix, const AxiConfig &cfg) : prefix(prefix), cfg(cfg), rnd(cfg.seed) {
        auto p = [&](const char *f, bool required = true) -> Sig * {
            Sig *s = sim.find(prefix + "_" + f);
            if (!s && required) sim_fail("AXI %s: no port %s_%s", prefix.c_str(), prefix.c_str(), f);
            return s;
        };
        ar_valid = p("ar_valid"); ar_ready = p("ar_ready"); ar_addr = p("ar_addr"); ar_len = p("ar_len"); ar_id = p("ar_id", false);
        r_valid  = p("r_valid");  r_ready  = p("r_ready");  r_data  = p("r_data");  r_id   = p("r_id", false);
        r_last   = p("r_last");   r_resp   = p("r_resp", false);
        aw_valid = p("aw_valid"); aw_ready = p("aw_ready"); aw_addr = p("aw_addr"); aw_len = p("aw_len"); aw_id = p("aw_id", false);
        w_valid  = p("w_valid");  w_ready  = p("w_ready");  w_data  = p("w_data");  w_strb = p("w_strb"); w_last = p("w_last");
        b_valid  = p("b_valid");  b_ready  = p("b_ready");  b_id    = p("b_id", false); b_resp = p("b_resp", false);
        beat_bytes = r_data->width/8;
        sim_require(cfg.size % beat_bytes == 0, "AXI memory size must be a multiple of the beat");
        mem.assign(cfg.size, 0);
        w_h.assign(w_data->words() + 2, 0);
        if (cfg.reorder) {
            sim_require((cfg.reorder & (cfg.reorder - 1)) == 0, "reorder must be a power of two");
            qs.resize(cfg.reorder);
            st_valid.assign(cfg.reorder, false);
            st_burst.resize(cfg.reorder);
            st_lat.assign(cfg.reorder, 0);
            cooldown.assign(cfg.reorder, 0);
        }
        sim.add_memory(sim_vector_memory(prefix, &mem));
        sim.add(prefix + "_freeze", 1, &freeze, true);
    }

    uint64_t beat_offset(const Burst &b, uint32_t beat) const {
        uint64_t depth = cfg.size/beat_bytes;
        return ((b.addr/beat_bytes + beat) % depth)*beat_bytes;
    }

    void roll(Sim &sim) {
        if (rolled == sim.cycle) return;
        rolled = sim.cycle;
        st_ar = rnd.percent(cfg.ar_stall);
        st_r  = rnd.percent(cfg.r_stall);
        st_aw = rnd.percent(cfg.aw_stall);
        st_w  = rnd.percent(cfg.w_stall);
        st_b  = rnd.percent(cfg.b_stall);
    }

    // Read data/valid of the burst being served.
    bool serving(Burst *b, uint32_t *beat) {
        if (cfg.reorder) { if (!cur_valid) return false; *b = rcur; *beat = rbeat; return true; }
        if (rstate != R_STREAM) return false;
        *b = rcur; *beat = rbeat;
        return true;
    }

    bool w_accepting() const {
        if (cfg.write_order == "w_first") return wq.size() < 64;
        return wstate == W_STREAM;
    }

    void drive(Sim &sim) override {
        roll(sim);
        // AR.
        bool ar_ok;
        if (cfg.reorder) ar_ok = qs[(ar_id ? ar_id->get() : 0) & (cfg.reorder - 1)].size() < 4;
        else             ar_ok = arq.size() < 8;
        ar_ready->set(ar_ok && !st_ar);
        // R.
        Burst b; uint32_t beat;
        if (serving(&b, &beat) && !st_r && !freeze) {
            r_valid->set(1);
            std::vector<uint32_t> w(r_data->words());
            std::memcpy(w.data(), mem.data() + beat_offset(b, beat), beat_bytes);
            r_data->set_words(w.data());
            if (r_id) r_id->set(b.id);
            r_last->set(beat == b.len);
            if (r_resp) r_resp->set(0);
        } else {
            r_valid->set(0);
            r_last->set(0);
        }
        // AW.
        bool aw_ok = awq.size() < 8 && !st_aw;
        if (cfg.write_order == "aw_after_w") aw_ok = aw_ok && w_valid->get();
        aw_ready->set(aw_ok);
        // W.
        w_ready->set(w_accepting() && !st_w);
        // B.
        b_valid->set(wstate == W_RESP && !st_b);
        if (b_id) b_id->set(wcur.id);
        if (b_resp) b_resp->set(0);
    }

    void hold_check(bool valid, bool ready, bool &held, uint64_t *h, Sig *a, Sig *l, Sig *i, const char *ch) {
        uint64_t cur[3] = {a->get(), l->get(), i ? i->get() : 0};
        if (held && (!valid || std::memcmp(cur, h, sizeof(cur))))
            sim_fail("AXI %s: %s payload changed or valid dropped while ungranted", prefix.c_str(), ch);
        held = valid && !ready;
        std::memcpy(h, cur, sizeof(cur));
    }

    void sample(Sim &sim) override {
        bool arv = ar_valid->get(), arr = ar_ready->get();
        bool rv  = r_valid->get(),  rr  = r_ready->get();
        bool awv = aw_valid->get(), awr = aw_ready->get();
        bool wv  = w_valid->get(),  wr  = w_ready->get();
        bool bv  = b_valid->get(),  br  = b_ready->get();

        hold_check(arv, arr, ar_held, ar_h, ar_addr, ar_len, ar_id, "AR");
        hold_check(awv, awr, aw_held, aw_h, aw_addr, aw_len, aw_id, "AW");
        {
            std::vector<uint32_t> cur(w_data->words() + 2);
            w_data->get_words(cur.data());
            cur[w_data->words()]     = (uint32_t)w_strb->get();
            cur[w_data->words() + 1] = (uint32_t)w_last->get();
            if (w_held && (!wv || cur != w_h))
                sim_fail("AXI %s: W payload changed or valid dropped while ungranted", prefix.c_str());
            w_held = wv && !wr;
            w_h = cur;
        }

        // Read server.
        bool r_fire = rv && rr;
        if (r_fire) n_r++;
        if (cfg.reorder) read_reorder(r_fire);
        else             read_inorder(r_fire);
        if (arv && arr) {
            Burst nb{ar_addr->get(), (uint32_t)ar_len->get(), ar_id ? (uint32_t)ar_id->get() : 0};
            if (nb.len + 1 > cfg.max_len)
                sim_fail("AXI %s: AR burst of %u beats exceeds %u (AxLEN overflow)", prefix.c_str(), nb.len + 1, cfg.max_len);
            if (cfg.log) std::printf("@axi %s ar %llu %llx %u %u\n", prefix.c_str(), (unsigned long long)sim.cycle,
                                     (unsigned long long)nb.addr, nb.len, nb.id);
            if (cfg.reorder) qs[nb.id & (cfg.reorder - 1)].push_back(nb);
            else             arq.push_back(nb);
            n_ar++;
        }

        // Write server.
        if (cfg.write_order == "w_first" && wv && wr) {
            std::vector<uint32_t> d(w_data->words());
            w_data->get_words(d.data());
            wq.push_back({d, {w_strb->get(), (bool)w_last->get()}});
        }
        switch (wstate) {
        case W_IDLE:
            if (!awq.empty()) { wcur = awq.front(); awq.pop_front(); wbeat = 0; wstate = W_STREAM; }
            break;
        case W_STREAM: {
            std::vector<uint32_t> d(w_data->words());
            uint64_t strb = 0;
            bool last = false, fire = false;
            if (cfg.write_order == "w_first") {
                if (!wq.empty()) { d = wq.front().first; strb = wq.front().second.first; last = wq.front().second.second; wq.pop_front(); fire = true; }
            } else if (wv && wr) {
                w_data->get_words(d.data()); strb = w_strb->get(); last = w_last->get(); fire = true;
            }
            if (fire) {
                n_w++;
                if (last != (wbeat == wcur.len))
                    sim_fail("AXI %s: WLAST %d on beat %u of a %u-beat burst", prefix.c_str(), last, wbeat, wcur.len + 1);
                uint8_t *dst = mem.data() + beat_offset(wcur, wbeat);
                const uint8_t *src = reinterpret_cast<const uint8_t *>(d.data());
                for (uint32_t i = 0; i < beat_bytes; i++)
                    if (strb >> i & 1) dst[i] = src[i];
                wbeat++;
                if (last) wstate = W_RESP;
            }
            break;
        }
        case W_RESP:
            if (bv && br) wstate = W_IDLE;
            break;
        }
        if (awv && awr) {
            Burst nb{aw_addr->get(), (uint32_t)aw_len->get(), aw_id ? (uint32_t)aw_id->get() : 0};
            if (nb.len + 1 > cfg.max_len)
                sim_fail("AXI %s: AW burst of %u beats exceeds %u (AxLEN overflow)", prefix.c_str(), nb.len + 1, cfg.max_len);
            if (cfg.log) std::printf("@axi %s aw %llu %llx %u %u\n", prefix.c_str(), (unsigned long long)sim.cycle,
                                     (unsigned long long)nb.addr, nb.len, nb.id);
            awq.push_back(nb);
            n_aw++;
        }
    }

    void read_inorder(bool r_fire) {
        switch (rstate) {
        case R_IDLE:
            if (!arq.empty()) { rcur = arq.front(); arq.pop_front(); rbeat = 0; rlat = cfg.latency; rstate = R_LAT; }
            break;
        case R_LAT:
            if (rlat == 0) rstate = R_STREAM; else rlat--;
            break;
        case R_STREAM:
            if (r_fire) {
                if (rbeat == rcur.len) {
                    // Chain the next queued burst back to back (latency applies from idle only).
                    if (!arq.empty()) { rcur = arq.front(); arq.pop_front(); rbeat = 0; }
                    else rstate = R_IDLE;
                } else rbeat++;
            }
            break;
        }
    }

    void read_reorder(bool r_fire) {
        uint32_t nq = cfg.reorder;
        bool cur_done = r_fire && rbeat == rcur.len;
        // Pick the first ready staged burst at/after qp.
        int pick = -1;
        for (uint32_t off = 0; off < nq; off++) {
            uint32_t q = (qp + off) & (nq - 1);
            if (st_valid[q] && st_lat[q] == 0) { pick = q; break; }
        }
        bool promote = (!cur_valid || cur_done) && pick >= 0;
        uint32_t done_q = rcur.id & (nq - 1);
        if (cur_valid && cur_done) cooldown[done_q] = cfg.latency + 1;
        if (promote) {
            rcur = st_burst[pick]; rbeat = 0; cur_valid = true; st_valid[pick] = false;
        } else if (cur_done) {
            cur_valid = false;
        } else if (r_fire) {
            rbeat++;
        }
        for (uint32_t q = 0; q < nq; q++) {
            if (cooldown[q]) cooldown[q]--;
            bool busy_here = cur_valid && (rcur.id & (nq - 1)) == q;
            if (!qs[q].empty() && !st_valid[q] && cooldown[q] == 0 && !busy_here) {
                st_burst[q] = qs[q].front(); qs[q].pop_front(); st_valid[q] = true; st_lat[q] = cfg.latency;
            } else if (st_valid[q] && st_lat[q]) {
                st_lat[q]--;
            }
        }
        qp = (qp + 1) & (nq - 1);
    }

    void report(Sim &) override {
        std::printf("@axistat %s %llu %llu %llu %llu\n", prefix.c_str(), (unsigned long long)n_ar,
                    (unsigned long long)n_r, (unsigned long long)n_aw, (unsigned long long)n_w);
    }
};
