// SPDX-License-Identifier: BSD-2-Clause
#include "Vregion_csr.h"
#include "verilated.h"
#include "bonsai_region_upload.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
struct UploadIO {
    Vregion_csr &dut;
    uint64_t ticks=0;
    void step() { dut.sys_clk=0; dut.eval(); dut.sys_clk=1; dut.eval(); ticks++; }
    uint32_t read(unsigned offset) {
        step();
        switch (offset) {
        case BNS_UP_ABI: return dut.abi;
        case BNS_UP_STATUS: return dut.status;
        case BNS_UP_ERROR: return dut.error;
        case BNS_UP_SPACE: return dut.space;
        case BNS_UP_WORDS: return dut.words;
        case BNS_UP_BEATS: return dut.beats;
        case BNS_UP_ACCEPTED: return dut.accepted;
        case BNS_UP_COMPLETED: return dut.completed;
        default: require(false, "unknown upload read register"); return 0;
        }
    }
    void write(unsigned offset, uint32_t value) {
        CData *re=nullptr;
        switch (offset) {
        case BNS_UP_REGION: dut.region_storage=value; re=&dut.region_re; break;
        case BNS_UP_OFFSET_HI: dut.offset_storage=(dut.offset_storage&0xffffffffu)|(uint64_t(value)<<32); re=&dut.offset_re; break;
        case BNS_UP_OFFSET_LO: dut.offset_storage=(dut.offset_storage&0xffffffff00000000ull)|value; re=&dut.offset_re; break;
        case BNS_UP_LENGTH: dut.length_storage=value; re=&dut.length_re; break;
        case BNS_UP_START: dut.start_storage=value; re=&dut.start_re; break;
        case BNS_UP_POKE: dut.poke_storage=value; re=&dut.poke_re; break;
        default: require(false, "unknown upload write register");
        }
        *re=1; step(); *re=0; step();
    }
};
int main() {
    Vregion_csr dut;
    auto clock=[&]() { dut.sys_clk=0; dut.eval(); dut.sys_clk=1; dut.eval(); };
    auto write=[&](CData &strobe, auto &storage, auto value) {
        storage=value; strobe=1; clock(); strobe=0; clock();
    };
    dut.sys_rst=1; clock(); clock(); dut.sys_rst=0; clock(); clock();
    require(dut.abi==0x10000, "wrong upload ABI");
    auto arm=[&](unsigned offset, unsigned beats) {
        write(dut.region_re, dut.region_storage, 7);
        write(dut.offset_re, dut.offset_storage, uint64_t(offset));
        write(dut.length_re, dut.length_storage, beats);
        write(dut.start_re, dut.start_storage, 1);
    };
    auto reset=[&]() {
        write(dut.reset_re, dut.reset_storage, 1);
        for (unsigned i=0; i<90; i++) clock();
        dut.freeze=0;
        for (unsigned i=0; i<3000 && dut.reset_busy; i++) clock();
        require(!dut.reset_busy && !dut.reset_stuck && !dut.error && !dut.status, "failed reset/drain");
    };
    auto upload=[&](unsigned offset, unsigned beats, unsigned expected_count, bool hold_b) {
        dut.freeze=hold_b; arm(offset, beats);
        // Staging writes must not change the accepted snapshot.
        write(dut.offset_re, dut.offset_storage, uint64_t(1)<<48);
        write(dut.length_re, dut.length_storage, 1);
        unsigned sent=0, waited=0; bool exhausted=false;
        while (sent<beats*8) {
            require(!dut.error, "upload error before payload end");
            unsigned credits=dut.space;
            require(credits<=256 && credits<=beats*8-sent, "invalid CSR credits");
            if (!credits) {
                if (hold_b && sent>32*8) { dut.freeze=0; exhausted=true; }
                clock(); require(++waited<10000, "upload credits timed out"); continue;
            }
            // Exercise a whole credit batch without polling between writes.
            for (unsigned j=0; j<credits; j++, sent++)
                write(dut.poke_re, dut.poke_storage, 0x61000000u+sent);
        }
        for (unsigned i=0; i<5000 && dut.completed!=expected_count; i++) clock();
        require(!dut.error && dut.completed==expected_count && !(dut.status&1), "upload completion failed");
        require(dut.words==beats*8 && dut.beats==beats, "incorrect payload accounting");
        if (hold_b) require(exhausted, "test failed to exhaust upload credits");
        for (unsigned i=0; i<beats; i++) {
            dut.debug_addr=offset/32+i; dut.eval();
            for (unsigned j=0; j<8; j++) require(dut.debug_data[j]==0x61000000u+i*8+j, "memory mismatch");
        }
    };
    upload(0x1000, 96, 1, true);
    // Partial beat plus accepted AXI writes, with B held. Reset must cancel and
    // discard the partial CSR packer, drain the bus, and preserve success count.
    dut.freeze=1; arm(0x4000, 40);
    for (unsigned i=0; i<9; i++) write(dut.poke_re, dut.poke_storage, 0x7000u+i);
    for (unsigned i=0; i<50; i++) clock();
    require(dut.status&1, "not busy before cancellation");
    reset(); require(dut.completed==1, "cancelled upload counted as success");
    upload(0x8000, 39, 2, false);
    require(dut.accepted==3, "wrong accepted command count");
    write(dut.poke_re, dut.poke_storage, 42);
    require(dut.error==512 && (dut.status&4) && !dut.space, "idle word was not rejected");
    reset();
    arm(0, 1);
    write(dut.start_re, dut.start_storage, 1);
    for (unsigned i=0; i<10; i++) clock();
    require(dut.error&256, "duplicate start not rejected");
    reset(); require(dut.completed==2, "failure counted as success");
    // Run the actual portable C upload driver against compiled gateware, not a
    // substitute register-state model. Read/write callbacks advance real clocks.
    UploadIO device{dut};
    bns_upload_io io{&device,
        [](void *p, uint32_t r) { return static_cast<UploadIO*>(p)->read(r); },
        [](void *p, uint32_t r, uint32_t v) { static_cast<UploadIO*>(p)->write(r,v); },
        [](void *p) { return static_cast<UploadIO*>(p)->ticks; },
        [](void *p) { static_cast<UploadIO*>(p)->step(); }};
    uint8_t payload[39*32+1];
    for (unsigned i=0; i<sizeof(payload); i++) payload[i]=uint8_t(i*17);
    bns_upload_progress progress;
    int rc=bns_region_upload(&io, 7, 0xc000, payload+1, sizeof(payload)-1, 10000, &progress);
    require(rc==0 && progress.completed_bytes==39*32 && !progress.hardware_error, "portable driver failed real RTL upload");
    for (unsigned i=0; i<39; i++) {
        dut.debug_addr=0xc000/32+i; dut.eval();
        for (unsigned j=0; j<8; j++) {
            const uint8_t *p=payload+1+i*32+j*4;
            uint32_t word=uint32_t(p[0])|uint32_t(p[1])<<8|uint32_t(p[2])<<16|uint32_t(p[3])<<24;
            require(dut.debug_data[j]==word, "portable driver payload mismatch");
        }
    }
    rc=bns_region_upload(&io, 7, uint64_t(1)<<32, payload+1, 32, 10000, &progress);
    require(rc==BNS_UP_HARDWARE && progress.hardware_error==2 && !progress.completed_bytes, "high offset aliased");
    reset();
    dut.freeze=1;
    rc=bns_region_upload(&io, 7, 0x1000, payload+1, 32, 100, &progress);
    require(rc==BNS_UP_TIMEOUT && !progress.completed_bytes, "missing B response did not time out");
    reset();
    std::puts("exact CSR uploads, credit exhaustion, reset/drain and portable C driver passed");
}
