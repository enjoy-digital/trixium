/* SPDX-License-Identifier: BSD-2-Clause */
#include "bonsai_region_upload.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct fake {
    uint32_t regs[14];
    uint64_t now, next_offset, words_written;
    unsigned armed, writes, no_space, no_done, bad_space, bad_count, fault_after;
};
static uint32_t read32(void *p, uint32_t r)
{
    struct fake *f=p;
    assert(r<=BNS_UP_COMPLETED && !(r&3));
    f->now++;
    if ((f->regs[BNS_UP_STATUS/4]&1) && !f->no_done &&
        f->regs[BNS_UP_WORDS/4]==f->regs[BNS_UP_LENGTH/4]*8) {
        f->regs[BNS_UP_STATUS/4]=0;
        f->regs[BNS_UP_COMPLETED/4]++;
        f->regs[BNS_UP_BEATS/4]=f->regs[BNS_UP_LENGTH/4];
        f->next_offset+=(uint64_t)f->regs[BNS_UP_LENGTH/4]*32;
    }
    if (r==BNS_UP_SPACE) {
        unsigned remaining=f->regs[BNS_UP_LENGTH/4]*8-f->regs[BNS_UP_WORDS/4];
        if (f->no_space) return 0;
        if (f->bad_space) return 65535;
        return remaining<256 ? remaining : 256;
    }
    if (r==BNS_UP_WORDS && f->bad_count) return f->regs[r/4]+1;
    return f->regs[r/4];
}
static void write32(void *p, uint32_t r, uint32_t v)
{
    struct fake *f=p;
    f->writes++;
    assert(r<=BNS_UP_COMPLETED && !(r&3));
    if (r==BNS_UP_START) {
        assert(v==1 && !f->regs[BNS_UP_STATUS/4]);
        assert(f->regs[BNS_UP_REGION/4]==42);
        assert(((uint64_t)f->regs[BNS_UP_OFFSET_HI/4]<<32 | f->regs[BNS_UP_OFFSET_LO/4])==f->next_offset);
        assert(f->regs[BNS_UP_LENGTH/4] && f->regs[BNS_UP_LENGTH/4]<=4*1024*1024);
        f->regs[BNS_UP_STATUS/4]=1;
        f->regs[BNS_UP_WORDS/4]=0;
        f->regs[BNS_UP_ACCEPTED/4]++;
        f->armed++;
    } else if (r==BNS_UP_POKE) {
        assert(v==0x5a5a5a5a && f->regs[BNS_UP_STATUS/4]&1);
        assert(f->regs[BNS_UP_WORDS/4]<f->regs[BNS_UP_LENGTH/4]*8);
        f->regs[BNS_UP_WORDS/4]++;
        f->words_written++;
        if (f->fault_after && f->words_written>=f->fault_after) f->regs[BNS_UP_ERROR/4]=8;
    } else f->regs[r/4]=v;
}
static uint64_t now_us(void *p) { return ((struct fake *)p)->now; }
static void wait_step(void *p) { ((struct fake *)p)->now+=10; }
static void init(struct fake *f, uint64_t offset)
{
    memset(f, 0, sizeof(*f));
    f->regs[BNS_UP_ABI/4]=0x10000;
    f->next_offset=offset;
}
int main(void)
{
    struct fake f;
    struct bns_upload_io io={&f, read32, write32, now_us, wait_step};
    struct bns_upload_progress progress;
    uint8_t small[97]; memset(small, 0x5a, sizeof(small));
    const uint64_t offset=(UINT64_C(1)<<32)-32;
    init(&f, offset);
    assert(bns_region_upload(&io,42,offset+1,small+1,32,1000,&progress)==BNS_UP_INVALID);
    assert(!f.writes);
    assert(bns_region_upload(&io,42,offset,small+1,31,1000,&progress)==BNS_UP_INVALID);
    assert(bns_region_upload(&io,42,UINT64_MAX-31,small,64,1000,&progress)==BNS_UP_INVALID);
    assert(bns_region_upload(&io,42,offset,small+1,32,0,&progress)==BNS_UP_INVALID);
    f.regs[BNS_UP_ABI/4]=2;
    assert(bns_region_upload(&io,42,offset,small+1,32,1000,&progress)==BNS_UP_UNSUPPORTED);
    assert(!f.writes);
    init(&f,offset); f.regs[BNS_UP_STATUS/4]=3; f.no_done=1;
    assert(bns_region_upload(&io,42,offset,small+1,32,1000,&progress)==BNS_UP_BUSY && !f.writes);
    init(&f,offset); f.no_space=1;
    assert(bns_region_upload(&io,42,offset,small+1,32,100,&progress)==BNS_UP_TIMEOUT && !progress.completed_bytes);
    init(&f,offset); f.no_done=1;
    assert(bns_region_upload(&io,42,offset,small+1,32,100,&progress)==BNS_UP_TIMEOUT && !progress.completed_bytes);
    init(&f,offset); f.bad_space=1;
    assert(bns_region_upload(&io,42,offset,small+1,32,1000,&progress)==BNS_UP_PROTOCOL && !f.words_written);
    init(&f,offset); f.bad_count=1;
    assert(bns_region_upload(&io,42,offset,small+1,32,1000,&progress)==BNS_UP_PROTOCOL);
    init(&f,offset); f.fault_after=1;
    assert(bns_region_upload(&io,42,offset,small+1,32,1000,&progress)==BNS_UP_HARDWARE);
    assert(progress.hardware_error==8 && !progress.completed_bytes);
    init(&f,offset); f.regs[BNS_UP_COMPLETED/4]=UINT32_MAX; f.regs[BNS_UP_ACCEPTED/4]=UINT32_MAX;
    f.now=UINT64_MAX-10;
    assert(bns_region_upload(&io,42,offset,small+1,96,1000,&progress)==0);
    assert(f.regs[BNS_UP_COMPLETED/4]==0 && progress.completed_bytes==96 && f.words_written==24);

    const size_t big=(128u<<20)+64;
    uint8_t *payload=malloc(big+1); assert(payload);
    memset(payload, 0x5a, big+1);
    init(&f,offset);
    assert(bns_region_upload(&io,42,offset,payload+1,big,UINT64_C(1000000000),&progress)==0);
    assert(f.armed==2 && f.words_written==big/4 && f.next_offset==offset+big && progress.completed_bytes==big);
    free(payload);
    puts("portable region driver: 64-bit chunking, unaligned input, errors, deadlines and counter wrap passed");
}
