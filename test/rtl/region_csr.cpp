// SPDX-License-Identifier: BSD-2-Clause
#include "Vregion_csr.h"
#include "verilated.h"
#include "bonsai_region_upload.h"
#include "bonsai_region_read.h"
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
struct ReadIO {
    UploadIO &device;
    uint32_t read(unsigned bank, unsigned offset) {
        auto &dut=device.dut; device.step();
        if (bank==BNS_RB_SCRATCHPAD_BANK) {
            require(offset==BNS_RB_SP_DATA, "unknown scratch read"); return dut.sp_csr_data;
        }
        require(bank==BNS_RB_VERIFY_BANK, "unknown readback bank");
        switch (offset) {
        case BNS_RB_ABI: return dut.v_abi;
        case BNS_RB_STATUS: return dut.v_status;
        case BNS_RB_ERROR: return dut.v_error;
        case BNS_RB_BEATS: return dut.v_beats;
        case BNS_RB_ACCEPTED: return dut.v_accepted;
        case BNS_RB_COMPLETED: return dut.v_completed;
        case BNS_RB_SP_PENDING: return dut.v_sp_pending;
        case BNS_RB_SP_COMPLETED: return dut.v_sp_completed;
        default: require(false, "unknown readback register"); return 0;
        }
    }
    void write(unsigned bank, unsigned offset, uint32_t value) {
        auto &dut=device.dut;
        CData *re=nullptr;
        if (bank==BNS_RB_SCRATCHPAD_BANK) {
            require(offset==BNS_RB_SP_ADR, "unknown scratch write");
            dut.sp_adr_storage=value; re=&dut.sp_adr_re;
        } else {
            require(bank==BNS_RB_VERIFY_BANK, "unknown write bank");
            switch (offset) {
            case BNS_RB_REGION: dut.v_region_storage=value; re=&dut.v_region_re; break;
            case BNS_RB_OFFSET_HI: dut.v_offset_storage=(dut.v_offset_storage&0xffffffffu)|(uint64_t(value)<<32); re=&dut.v_offset_re; break;
            case BNS_RB_OFFSET_LO: dut.v_offset_storage=(dut.v_offset_storage&0xffffffff00000000ull)|value; re=&dut.v_offset_re; break;
            case BNS_RB_LENGTH: dut.v_length_storage=value; re=&dut.v_length_re; break;
            case BNS_RB_DST: dut.v_dst_storage=value; re=&dut.v_dst_re; break;
            case BNS_RB_START: dut.v_start_storage=value; re=&dut.v_start_re; break;
            default: require(false, "unknown readback write");
            }
        }
        *re=1; device.step(); *re=0; device.step();
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
        require(!dut.reset_busy && !dut.reset_stuck && !dut.error && !dut.status
                && !dut.v_error && !dut.v_status, "failed reset/drain");
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
    auto read_arm=[&](uint64_t offset, unsigned beats, unsigned dst=0x400) {
        write(dut.v_region_re, dut.v_region_storage, 7);
        write(dut.v_offset_re, dut.v_offset_storage, offset);
        write(dut.v_length_re, dut.v_length_storage, beats);
        write(dut.v_dst_re, dut.v_dst_storage, dst);
        write(dut.v_start_re, dut.v_start_storage, 1);
    };
    auto read_verify=[&](unsigned completed) {
        read_arm(0xc000,39);
        write(dut.v_offset_re, dut.v_offset_storage, uint64_t(1)<<48);
        write(dut.v_dst_re, dut.v_dst_storage, 0x8000);
        for (unsigned i=0; i<5000 && dut.v_completed!=completed; i++) clock();
        require(!dut.v_error && dut.v_completed==completed && !dut.v_status && dut.v_beats==39, "readback completion");
        for (unsigned i=0; i<39; i++) {
            dut.debug_addr=0xc000/32+i; dut.sp_addr=0x400/32+i; dut.eval();
            for (unsigned j=0; j<8; j++) require(dut.sp_data[j]==dut.debug_data[j], "upload readback mismatch");
        }
    };
    read_verify(1);
    // Reset with real accepted reads outstanding. Old responses must drain to
    // the shim, not a new read's destination or its success accounting.
    dut.freeze=1; read_arm(0xc000,39);
    for (unsigned i=0; i<60; i++) clock();
    require(dut.v_status&1, "readback not busy before reset");
    reset(); require(dut.v_completed==1, "cancelled read counted as success");
    read_verify(2); require(dut.v_accepted==3, "read accepted accounting");
    dut.freeze=1; read_arm(0xc000,39);
    write(dut.v_start_re, dut.v_start_storage, 1);
    require((dut.v_status&4) && (dut.v_error&256), "duplicate read start not rejected");
    reset(); require(dut.v_completed==2, "failed read counted as success");
    read_arm(uint64_t(1)<<32,1);
    for (unsigned i=0; i<40; i++) clock();
    require((dut.v_status&4) && dut.v_error==2 && dut.v_completed==2, "read offset aliased");
    reset(); read_verify(3);
    ReadIO read_device{device};
    bns_read_io read_io{&read_device,
        [](void *p,uint32_t bank,uint32_t offset) { return static_cast<ReadIO*>(p)->read(bank,offset); },
        [](void *p,uint32_t bank,uint32_t offset,uint32_t value) { static_cast<ReadIO*>(p)->write(bank,offset,value); },
        [](void *p) { return static_cast<ReadIO*>(p)->device.ticks; },
        [](void *p) { static_cast<ReadIO*>(p)->device.step(); }};
    bns_read_progress read_progress;
    uint8_t copied[sizeof(payload)+1];
    rc=bns_region_readback(&read_io,7,0xc000,copied+1,39*32,0x400,64,10000,&read_progress);
    require(rc==0 && read_progress.completed_bytes==39*32, "portable readback failed real RTL");
    for (unsigned i=0; i<39*32; i++) require(copied[i+1]==payload[i+1], "portable readback stale/wrong bytes");
    rc=bns_region_verify(&read_io,7,0xc000,payload+1,39*32,0x400,96,10000,&read_progress);
    require(rc==0 && read_progress.completed_bytes==39*32, "portable verification failed real RTL");
    // Corrupt a physical-memory byte after a successful upload. The real C
    // verifier must find its exact offset after a completely verified prefix.
    dut.debug_addr=0xc000/32+3; dut.eval();
    for (unsigned j=0; j<8; j++) dut.debug_write[j]=dut.debug_data[j];
    dut.debug_write[1]^=0x100; dut.debug_we=1; clock(); dut.debug_we=0;
    rc=bns_region_verify(&read_io,7,0xc000,payload+1,39*32,0x400,64,10000,&read_progress);
    require(rc==BNS_RB_MISMATCH && read_progress.mismatch_offset==101
            && read_progress.completed_bytes==64, "physical corruption was not located");
    dut.debug_addr=0xc000/32+3; dut.debug_write[1]^=0x100;
    dut.debug_we=1; clock(); dut.debug_we=0;
    // CSR data remains stale while the CPU owns the scratchpad read port.
    // Count polling must time out, never compare or return that stale word.
    dut.hold_peek=1;
    rc=bns_region_verify(&read_io,7,0xc000,payload+1,32,0x400,64,200,&read_progress);
    require(rc==BNS_RB_TIMEOUT && !read_progress.completed_bytes && dut.v_sp_pending,
            "withheld scratch read did not time out");
    const unsigned peek_before=dut.v_sp_completed;
    reset(); require(!dut.v_sp_pending && dut.v_sp_completed==peek_before, "reset failed to cancel pending peek");
    dut.hold_peek=0;
    rc=bns_region_verify(&read_io,7,0xc000,payload+1,39*32,0x400,64,10000,&read_progress);
    require(rc==0 && read_progress.completed_bytes==39*32, "readback failed after peek reset/drain");
    rc=bns_region_upload(&io, 7, uint64_t(1)<<32, payload+1, 32, 10000, &progress);
    require(rc==BNS_UP_HARDWARE && progress.hardware_error==2 && !progress.completed_bytes, "high offset aliased");
    reset();
    dut.freeze=1;
    rc=bns_region_upload(&io, 7, 0x1000, payload+1, 32, 100, &progress);
    require(rc==BNS_UP_TIMEOUT && !progress.completed_bytes, "missing B response did not time out");
    reset();
    std::puts("portable C upload/readback/verify against real RTL, corruption, blocked peeks and reset/drain passed");
}
