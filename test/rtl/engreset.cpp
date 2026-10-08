// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#include "Vengreset.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>

static void require(bool ok,const char *why) {
    if (!ok) { std::fprintf(stderr,"%s\n",why); std::exit(1); }
}
static void tick(Vengreset &d) { d.sys_clk=0; d.eval(); d.sys_clk=1; d.eval(); }
int main() {
    Vengreset d; d.sys_rst=1; tick(d); tick(d); d.sys_rst=0; tick(d);
    require(d.counter==0x1234 && !d.engine_reset,"initial system reset");
    for (unsigned duration:{1u,2u,9u}) {
        d.load=1; d.value=0xabcdef01; tick(d); d.load=0;
        require(d.counter==0xabcdef01,"counter setup");
        d.sys_rst=1; tick(d);
        require(d.engine_reset,"sys reset was overridden by reset insertion");
        for (unsigned i=1;i<duration;i++) tick(d);
        // The domain reset is registered: even a one-clock system pulse must
        // reset the engine on the next edge, including the release edge.
        d.sys_rst=0; tick(d);
        require(d.counter==0x1234 && !d.engine_reset && !d.holding,"system reset did not reset engine state");
        d.enable=1; tick(d); d.enable=0;
        require(d.counter==0x1235,"engine did not resume after system reset");
    }
    d.trigger=1; tick(d); d.trigger=0;
    for (unsigned i=0;i<4;i++) tick(d);
    require(d.holding && d.busy && d.engine_reset && d.counter==0x1234,"soft reset lost hold window");
    for (unsigned i=0;i<12;i++) tick(d);
    require(!d.holding && !d.busy && !d.engine_reset,"soft reset did not release");
    d.load=1; d.value=0xbadfeed; tick(d); d.load=0;
    d.trigger=1; tick(d); d.trigger=0; d.sys_rst=1; tick(d); tick(d);
    require(d.engine_reset && !d.holding && d.counter==0x1234,"global reset during soft hold");
    d.sys_rst=0; tick(d);
    require(!d.engine_reset && !d.holding && d.counter==0x1234,"global release replayed stale hold");
    std::puts("ENGRESET_SYSTEM_PASS pulse_widths=1,2,9 soft_hold=8 overlap=1");
}
