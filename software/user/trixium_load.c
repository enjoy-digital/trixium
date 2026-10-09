/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium load: upload a BONS weight image to the card's weight memory
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 * The header+table go to the channel-0 staging area (the firmware's `hdrload` copies them
 * into CPU memory), each channel's payload to its own window. Words are written through BAR0
 * (trixium_loader.c): ~107 MB/s, and nothing can be dropped, reordered or invented. The
 * memory keeps the weights across firmware reboots (CPU reset); only an FPGA reconfiguration
 * clears them.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "trixium_loader.h"

/* Help */
/*------*/

static void help(int status)
{
    fprintf(status ? stderr : stdout,
            "TriXium weight loader\n"
            "usage: trixium_load [options] image.img\n"
            "\n"
            "options:\n"
            "-h                                Help.\n"
            "-c device_num                     Select the device (default = 0).\n");
    exit(status);
}

/* Main */
/*------*/

int main(int argc, char **argv)
{
    int devnum = 0;
    int opt;

    /* Parameters. */
    while ((opt = getopt(argc, argv, "c:h")) != -1) {
        switch (opt) {
        case 'c':
            devnum = atoi(optarg);
            break;
        default:
            help(0);
        }
    }
    if (optind >= argc)
        help(1);

    /* Upload. */
    loader_bar_map(devnum);
    return loader_upload_image(argv[optind], 1) ? 0 : 1;
}
