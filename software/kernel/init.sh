#!/bin/bash
#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#
# Load trixium.ko + liteuart.ko and open up the device nodes (`make install` sets up udev and
# modules-load.d for the persistent version of this).

cd "$(dirname "$0")"

# Check if trixium module is already installed.
if lsmod | grep -q '^trixium '; then
    echo "trixium module already installed."
    exit 0
fi

# liteuart binds on top of trixium: reload it after the PCIe driver.
if lsmod | grep -q '^liteuart '; then
    rmmod liteuart
fi

# Install trixium module.
INS=$(insmod trixium.ko 2>&1)
if [ "$?" != "0" ] ; then
    ERR=$(echo $INS | sed -s "s/.*trixium.ko: //")
    case $ERR in
    'Invalid module format')
        set -e
        echo "Kernel may have changed, try to rebuild module"
        make -s clean
        make -s
        insmod trixium.ko
        set +e
        ;;
    'No such file or directory')
        set -e
        echo "Module not compiled"
        make -s
        insmod trixium.ko
        set +e
        ;;
    'Required key not available'|*'unsigned module'*)
        echo "Can't insert kernel module, secure boot is probably enabled"
        echo "Please disable it from BIOS"
        exit 1
        ;;
    *)
        >&2 echo $INS
        exit 1
    esac
fi

# Install liteuart module.
insmod liteuart.ko

# Change permissions on trixium/liteuart created devices.
for i in $(seq 0 16) ; do
    chmod 666 /dev/trixium$i /dev/ttyLXU$i > /dev/null 2>&1
done
