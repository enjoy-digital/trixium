/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium host records for trixium_smi (no SoC headers: shared by every host tool, including
 * the console-only trixium_chat of a board without PCIe).
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 */

#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include "trixium_loader.h"

/* Card state for trixium_smi */
/*----------------------------*/

void loader_state_write(int devnum, const char *kind, const char *fmt, ...)
{
    char path[256], tmp[300];
    va_list ap;

    /* Shared by root (sudo) and user runs: the directory and the records are world-writable. */
    if (mkdir(TRIXIUM_STATE_DIR, 0777) == 0)
        chmod(TRIXIUM_STATE_DIR, 0777);
    snprintf(path, sizeof(path), TRIXIUM_STATE_DIR "/trixium%d.%s.json", devnum, kind);
    snprintf(tmp,  sizeof(tmp),  "%s.%d", path, (int)getpid());
    FILE *f = fopen(tmp, "w");
    if (!f)
        return; /* Best effort: the monitor is optional. */
    fprintf(f, "{\"pid\": %d, ", (int)getpid());
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fprintf(f, "}\n");
    fclose(f);
    chmod(tmp, 0666);
    rename(tmp, path); /* Atomic: a reader never sees half a record. */
}
