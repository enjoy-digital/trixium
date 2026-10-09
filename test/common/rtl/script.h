//
// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

// Script interpreter: runs a common.verilator.Script against a Sim. One operation per line:
//
//   set NAME HEX                      drive an input (holds until set again)
//   step N                            N clock cycles
//   reset N                           sys_rst high for N cycles
//   wait CAP FATAL TAG COND...        cycle until every COND (NAME==HEX / NAME!=HEX) holds,
//                                     at most CAP cycles; FATAL=1 fails the run on timeout
//   handshake VALID READY CAP TAG     VALID=1 until a cycle with READY, then VALID=0
//   peek NAME TAG                     record a value
//   count NAME TAG / count_end TAG    count the cycles NAME != 0 in between
//   read MEM ADDR LEN TAG             save LEN bytes of a memory to <out>/TAG.bin
//   dump NAME...                      signals printed if the run fails
//   random NAME PCT SEED              drive NAME 1 on PCT percent of the cycles (PCT < 0: stop)
//
// Results go to stdout as "@<kind> <tag> ..." lines (verilator.py parses them).

#pragma once

#include <sstream>

#include "sim.h"

// Helpers ------------------------------------------------------------------------------------------

static std::vector<uint32_t> script_hex_words(const std::string &hex, int words) {
    std::vector<uint32_t> w(words, 0);
    int bit = 0;
    for (int i = (int)hex.size() - 1; i >= 0 && bit < 32*words; i--, bit += 4) {
        char c = hex[i];
        uint32_t v = (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 :
                     (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0xFF;
        if (v == 0xFF) sim_fail("bad hex value '%s'", hex.c_str());
        w[bit/32] |= v << (bit % 32);
    }
    return w;
}

struct ScriptCond {
    Sig *sig;
    bool eq;
    std::vector<uint32_t> value;
    bool holds() const {
        std::vector<uint32_t> cur(sig->words());
        sig->get_words(cur.data());
        return (cur == value) == eq;
    }
};

// Random input: a 0/1 drawn every cycle (a withheld grant, a stalling handshake).
struct RandomInput : Model {
    Sig      *sig;
    int       pct;
    SimRandom rnd;
    uint64_t  rolled = ~0ull;
    RandomInput(Sig *sig, int pct, uint32_t seed) : sig(sig), pct(pct), rnd(seed) {}
    void drive(Sim &sim) override {
        if (pct < 0 || rolled == sim.cycle) return;
        rolled = sim.cycle;
        sig->set(rnd.percent((uint32_t)pct));
    }
    void sample(Sim &) override {}
};

// Interpreter --------------------------------------------------------------------------------------

static void run_script(Sim &sim, const std::string &path, const std::string &outdir) {
    std::ifstream f(path);
    if (!f) sim_fail("cannot open script %s", path.c_str());
    std::vector<Sig *> dumps;
    std::map<std::string, std::pair<Sig *, uint64_t>> counters;
    std::map<std::string, RandomInput *> randoms;
    std::string cur_line;

    sim_on_fail = [&]() {
        std::fprintf(stderr, "at cycle %llu, script line: %s\n", (unsigned long long)sim.cycle, cur_line.c_str());
        for (Sig *s : dumps) std::fprintf(stderr, "  %s = 0x%s\n", s->name.c_str(), s->hex().c_str());
    };
    sim.on_cycle.push_back([&]() {
        for (auto &c : counters) if (c.second.first->get()) c.second.second++;
    });

    std::string line;
    while (std::getline(f, line)) {
        cur_line = line;
        std::istringstream in(line);
        std::string op;
        if (!(in >> op) || op[0] == '#') continue;
        if (op == "set") {
            std::string name, hex;
            in >> name >> hex;
            Sig &s = sim.sig(name);
            s.set_words(script_hex_words(hex, s.words()).data());
            sim.settled = false;
        } else if (op == "step") {
            uint64_t n; in >> n;
            for (uint64_t i = 0; i < n; i++) sim.tick();
        } else if (op == "reset") {
            int n; in >> n;
            sim.reset(n);
        } else if (op == "wait") {
            uint64_t cap; int fatal; std::string tag, c;
            in >> cap >> fatal >> tag;
            std::vector<ScriptCond> conds;
            while (in >> c) {
                size_t k = c.find("==");
                bool eq = k != std::string::npos;
                if (!eq) k = c.find("!=");
                if (k == std::string::npos) sim_fail("bad condition '%s'", c.c_str());
                Sig *s = &sim.sig(c.substr(0, k));
                std::vector<uint32_t> v = script_hex_words(c.substr(k + 2), s->words());
                if (s->width % 32) v.back() &= (1u << (s->width % 32)) - 1;
                conds.push_back({s, eq, v});
            }
            uint64_t n = 0;
            bool ok = false;
            for (;;) {
                if (!sim.settled) sim.settle();
                ok = true;
                for (auto &cd : conds) ok = ok && cd.holds();
                if (ok || n >= cap) break;
                sim.tick();
                n++;
            }
            if (!ok && fatal) sim_fail("%s: timeout after %llu cycles", tag.c_str(), (unsigned long long)n);
            std::printf("@wait %s %llu %d\n", tag.c_str(), (unsigned long long)n, ok);
        } else if (op == "handshake") {
            std::string valid, ready, tag; uint64_t cap;
            in >> valid >> ready >> cap >> tag;
            Sig &v = sim.sig(valid), &r = sim.sig(ready);
            v.set(1);
            sim.settled = false;
            uint64_t n = 0;
            for (;;) {
                if (!sim.settled) sim.settle();
                bool fire = r.get();
                sim.tick();
                if (fire) break;
                if (++n >= cap) sim_fail("%s: %s never accepted after %llu cycles", tag.c_str(), valid.c_str(),
                                         (unsigned long long)n);
            }
            v.set(0);
            sim.settled = false;
            std::printf("@handshake %s %llu\n", tag.c_str(), (unsigned long long)n);
        } else if (op == "peek") {
            std::string name, tag;
            in >> name >> tag;
            if (!sim.settled) sim.settle();
            std::printf("@peek %s %s\n", tag.c_str(), sim.sig(name).hex().c_str());
        } else if (op == "count") {
            std::string name, tag;
            in >> name >> tag;
            counters[tag] = {&sim.sig(name), 0};
        } else if (op == "count_end") {
            std::string tag;
            in >> tag;
            auto it = counters.find(tag);
            if (it == counters.end()) sim_fail("count_end %s without count", tag.c_str());
            std::printf("@count %s %llu\n", tag.c_str(), (unsigned long long)it->second.second);
            counters.erase(it);
        } else if (op == "read") {
            std::string mem, tag; size_t addr, len;
            in >> mem >> addr >> len >> tag;
            Mem &m = sim.mem(mem);
            if (addr + len > m.size) sim_fail("read %s [%zu, %zu) beyond %zu bytes", mem.c_str(), addr, addr + len, m.size);
            std::vector<uint8_t> buf(len);
            m.read(addr, buf.data(), len);
            sim_save(outdir + "/" + tag + ".bin", buf.data(), len);
            std::printf("@read %s %zu\n", tag.c_str(), len);
        } else if (op == "random") {
            std::string name; int pct; uint32_t seed;
            in >> name >> pct >> seed;
            auto it = randoms.find(name);
            if (it == randoms.end()) {
                auto *r = new RandomInput(&sim.sig(name), pct, seed);
                randoms[name] = r;
                sim.models.push_back(r);
            } else {
                it->second->pct = pct;
            }
            sim.settled = false;
        } else if (op == "dump") {
            std::string name;
            while (in >> name) dumps.push_back(&sim.sig(name));
        } else {
            sim_fail("unknown script operation '%s'", op.c_str());
        }
    }
    sim_on_fail = nullptr;
    sim.on_cycle.pop_back();
    std::printf("@cycles total %llu\n", (unsigned long long)sim.cycle);
}
