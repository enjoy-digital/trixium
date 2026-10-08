/* This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * trixium_load: upload a BONS weight image to the card's weight memory.
 *
 *   trixium_load [-c devnum] image.img
 *
 * The header+table go to the channel-0 staging area (the firmware's `hdrload` copies them
 * into CPU memory), each channel's payload to its own window. Words are poked through BAR0
 * (trixium_loader.c): ~107 MB/s, and nothing can be dropped, reordered or invented. The
 * memory keeps the weights across firmware reboots (CPU reset); only an FPGA
 * reconfiguration clears them.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "trixium_loader.h"

int main(int argc, char **argv)
{
    int devnum = 0;
    int opt;
    while ((opt = getopt(argc, argv, "c:h")) != -1) {
        if (opt == 'c') devnum = atoi(optarg);
        else { printf("usage: trixium_load [-c devnum] image.img\n"); exit(0); }
    }
    if (optind >= argc) {
        fprintf(stderr, "usage: trixium_load [-c devnum] image.img\n");
        exit(1);
    }
    loader_bar_map(devnum);
    return loader_upload_image(argv[optind], 1) ? 0 : 1;
}
