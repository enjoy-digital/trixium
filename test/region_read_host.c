/* SPDX-License-Identifier: BSD-2-Clause */
#include "bonsai_region_read.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

struct fake {
    uint32_t regs[14], data, peek_address;
    uint64_t now, next_offset, chunk_offset, initial_offset, corrupt;
    unsigned writes, starts, requests, premature, read_left, peek_left;
    unsigned no_read, no_peek, bad_beats, bad_accepted, bad_completed, bad_peek, pending_after;
    unsigned error_after, peeks_after_error;
};
static uint8_t byte_at(uint64_t address) { return (uint8_t)(address*17+(address>>32)*37+9); }
static void tick(struct fake *f)
{
    f->now++;
    if (f->read_left && !f->no_read && !--f->read_left) {
        f->regs[BNS_RB_STATUS/4]=0;
        f->regs[BNS_RB_COMPLETED/4]+=f->bad_completed ? 2 : 1;
        f->regs[BNS_RB_BEATS/4]=f->regs[BNS_RB_LENGTH/4]+f->bad_beats;
        if (f->bad_accepted) f->regs[BNS_RB_ACCEPTED/4]++;
        f->next_offset+=(uint64_t)f->regs[BNS_RB_LENGTH/4]*32;
        if (f->error_after) f->regs[BNS_RB_ERROR/4]=16;
    }
    if (f->peek_left && !f->no_peek && !--f->peek_left) {
        f->data=0;
        for (unsigned j=0; j<4; j++) {
            uint64_t address=f->chunk_offset+f->peek_address-f->regs[BNS_RB_DST/4]+j;
            uint8_t value=byte_at(address);
            if (address-f->initial_offset==f->corrupt) value^=1;
            f->data|=(uint32_t)value<<(8*j);
        }
        f->regs[BNS_RB_SP_PENDING/4]=f->pending_after;
        f->regs[BNS_RB_SP_COMPLETED/4]+=f->bad_peek ? 2 : 1;
    }
}
static uint32_t read32(void *p, uint32_t bank, uint32_t offset)
{
    struct fake *f=p; tick(f);
    if (bank==BNS_RB_SCRATCHPAD_BANK) {
        assert(offset==BNS_RB_SP_DATA);
        if (f->peek_left || f->regs[BNS_RB_SP_PENDING/4]) f->premature++;
        return f->data;
    }
    assert(bank==BNS_RB_VERIFY_BANK && !(offset&3) && offset<=BNS_RB_SP_COMPLETED);
    return f->regs[offset/4];
}
static void write32(void *p, uint32_t bank, uint32_t offset, uint32_t value)
{
    struct fake *f=p; tick(f); f->writes++;
    if (bank==BNS_RB_SCRATCHPAD_BANK) {
        assert(offset==BNS_RB_SP_ADR && !f->regs[BNS_RB_SP_PENDING/4]);
        assert(!f->regs[BNS_RB_STATUS/4]);
        assert(value>=f->regs[BNS_RB_DST/4] && value<f->regs[BNS_RB_DST/4]+32*f->regs[BNS_RB_LENGTH/4]);
        f->regs[BNS_RB_SP_PENDING/4]=1; f->peek_left=11; f->peek_address=value; f->requests++;
        if (f->regs[BNS_RB_ERROR/4]) f->peeks_after_error++;
        return;
    }
    assert(bank==BNS_RB_VERIFY_BANK && !(offset&3) && offset<=BNS_RB_START);
    f->regs[offset/4]=value;
    if (offset==BNS_RB_START) {
        assert(value==1 && !f->regs[BNS_RB_STATUS/4]);
        f->chunk_offset=(uint64_t)f->regs[BNS_RB_OFFSET_HI/4]<<32 | f->regs[BNS_RB_OFFSET_LO/4];
        assert(f->chunk_offset==f->next_offset && f->regs[BNS_RB_REGION/4]==42);
        assert(f->regs[BNS_RB_LENGTH/4] && f->regs[BNS_RB_LENGTH/4]<=2);
        f->regs[BNS_RB_STATUS/4]=1; f->regs[BNS_RB_ACCEPTED/4]++; f->read_left=7; f->starts++;
    }
}
static uint64_t now_us(void *p) { return ((struct fake *)p)->now; }
static void wait_step(void *p) { tick(p); }
static void init(struct fake *f, uint64_t offset)
{
    memset(f, 0, sizeof(*f)); f->next_offset=f->initial_offset=offset; f->corrupt=UINT64_MAX;
    f->regs[BNS_RB_ABI/4]=0x10001; f->data=0xdeadbeef;
}
int main(void)
{
    struct fake f;
    struct bns_read_io io={&f,read32,write32,now_us,wait_step};
    struct bns_read_progress progress;
    uint8_t source[161], output[162];
    const uint64_t base=(UINT64_C(1)<<32)-32;
    for (unsigned i=0; i<160; i++) source[i+1]=byte_at(base+i);
    init(&f,base); memset(output,0xff,sizeof(output));
    assert(!bns_region_readback(&io,42,base,output+1,160,64,64,10000,&progress));
    assert(!memcmp(source+1,output+1,160) && output[0]==255 && output[161]==255);
    assert(f.starts==3 && f.requests==40 && !f.premature && progress.completed_bytes==160);
    init(&f,base); f.now=UINT64_MAX-15;
    f.regs[BNS_RB_ACCEPTED/4]=f.regs[BNS_RB_COMPLETED/4]=f.regs[BNS_RB_SP_COMPLETED/4]=UINT32_MAX;
    assert(!bns_region_verify(&io,42,base,source+1,160,64,64,10000,&progress));
    assert(f.starts==3 && f.requests==40 && !f.premature && progress.completed_bytes==160);
    assert(progress.mismatch_offset==UINT64_MAX && f.regs[BNS_RB_COMPLETED/4]==2);
    for (unsigned bad=0; bad<160; bad++) {
        init(&f,base); f.corrupt=bad;
        assert(bns_region_verify(&io,42,base,source+1,160,64,64,10000,&progress)==BNS_RB_MISMATCH);
        assert(progress.mismatch_offset==bad && progress.completed_bytes==(bad/64)*64 && !f.premature);
    }
    init(&f,base);
    assert(bns_region_verify(NULL,42,base,source,32,64,64,100,&progress)==BNS_RB_INVALID);
    assert(bns_region_readback(&io,42,base,NULL,32,64,64,100,&progress)==BNS_RB_INVALID);
    assert(bns_region_verify(&io,42,base,NULL,32,64,64,100,&progress)==BNS_RB_INVALID);
    assert(bns_region_verify(&io,42,base+1,source,32,64,64,100,&progress)==BNS_RB_INVALID);
    assert(bns_region_verify(&io,42,base,source,31,64,64,100,&progress)==BNS_RB_INVALID);
    assert(bns_region_verify(&io,42,base,source,32,65,64,100,&progress)==BNS_RB_INVALID);
    assert(bns_region_verify(&io,42,base,source,32,64,63,100,&progress)==BNS_RB_INVALID);
    assert(bns_region_verify(&io,42,base,source,32,64,0,100,&progress)==BNS_RB_INVALID);
    assert(bns_region_verify(&io,42,base,source,32,1<<24,64,100,&progress)==BNS_RB_INVALID);
    assert(bns_region_verify(&io,42,UINT64_MAX-31,source,32,64,64,100,&progress)==BNS_RB_INVALID);
    assert(bns_region_verify(&io,42,base,source,32,64,64,0,&progress)==BNS_RB_INVALID && !f.writes);
    for (unsigned abi=0; abi<3; abi++) {
        f.regs[BNS_RB_ABI/4]=(uint32_t[]){0,0x10000,0x20000}[abi];
        assert(bns_region_verify(&io,42,base,source,32,64,64,100,&progress)==BNS_RB_UNSUPPORTED && !f.writes);
    }
    init(&f,base); f.regs[BNS_RB_STATUS/4]=1;
    assert(bns_region_verify(&io,42,base,source,32,64,64,100,&progress)==BNS_RB_BUSY && !f.writes);
    init(&f,base); f.regs[BNS_RB_SP_PENDING/4]=1;
    assert(bns_region_verify(&io,42,base,source,32,64,64,100,&progress)==BNS_RB_BUSY && !f.writes);
    init(&f,base); f.regs[BNS_RB_STATUS/4]=8;
    assert(bns_region_verify(&io,42,base,source,32,64,64,100,&progress)==BNS_RB_PROTOCOL && !f.writes);
    for (unsigned fault=0; fault<8; fault++) {
        init(&f,base);
        if (fault==0) f.no_read=1;
        if (fault==1) f.no_peek=1;
        if (fault==2) f.bad_beats=1;
        if (fault==3) f.bad_accepted=1;
        if (fault==4) f.bad_completed=1;
        if (fault==5) f.bad_peek=1;
        if (fault==6) f.pending_after=1;
        if (fault==7) f.error_after=1;
        int rc=bns_region_verify(&io,42,base,source+1,32,64,64,1000,&progress);
        assert(rc==(fault<2 ? BNS_RB_TIMEOUT : fault==7 ? BNS_RB_HARDWARE : BNS_RB_PROTOCOL));
        assert(!progress.completed_bytes && !f.premature && !f.peeks_after_error);
        if (fault==7) assert(progress.hardware_error==16);
    }
    init(&f,base); io.wait=NULL;
    assert(!bns_region_verify(&io,42,base,source+1,160,64,64,10000,NULL));
    puts("portable readback: full-width chunks, delayed peeks, exact comparison, corruption, deadlines and wrap passed");
}
