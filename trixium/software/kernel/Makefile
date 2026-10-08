# Makefile for the TriXium kernel modules (derived from LitePCIe's).
#   trixium.ko:  PCIe driver (/dev/trixiumN: CSR access, DMA).
#   liteuart.ko: firmware console over PCIe (/dev/ttyLXUN, the crossover UART).
KERNEL_VERSION:=$(shell uname -r)
KERNEL_PATH?=/lib/modules/$(KERNEL_VERSION)/build
ARCH?=$(shell uname -m)

obj-m = trixium.o liteuart.o
trixium-objs = main.o

all: trixium.ko liteuart.ko

trixium.ko: main.c litepcie.h config.h flags.h csr.h soc.h
	make -C $(KERNEL_PATH) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(shell pwd) modules

liteuart.ko: liteuart.c
	make -C $(KERNEL_PATH) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(shell pwd) modules

install: all
	sudo mkdir -p /lib/modules/$(KERNEL_VERSION)/extra
	sudo install -m 644 trixium.ko liteuart.ko /lib/modules/$(KERNEL_VERSION)/extra/
	sudo depmod -a
	echo trixium  | sudo tee    /etc/modules-load.d/trixium.conf > /dev/null
	echo liteuart | sudo tee -a /etc/modules-load.d/trixium.conf > /dev/null
	echo 'KERNEL=="trixium*", MODE="0666"' | sudo tee /etc/udev/rules.d/99-trixium.rules > /dev/null

uninstall:
	sudo rm -f /lib/modules/$(KERNEL_VERSION)/extra/trixium.ko \
	           /lib/modules/$(KERNEL_VERSION)/extra/liteuart.ko \
	           /etc/modules-load.d/trixium.conf /etc/udev/rules.d/99-trixium.rules
	sudo depmod -a

clean:
	make -C $(KERNEL_PATH) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(shell pwd) clean
	rm -f *~

.PHONY: all install uninstall clean
