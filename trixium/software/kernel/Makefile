# Makefile for the TriXium kernel modules (LitePCIe-derived).
# bonsai.ko: PCIe driver (/dev/bonsaiN, DMA, CSR access).
# liteuart.ko: crossover UART console over PCIe (/dev/ttyLXU0).
KERNEL_VERSION:=$(shell uname -r)
KERNEL_PATH?=/lib/modules/$(KERNEL_VERSION)/build
ARCH?=$(shell uname -m)

obj-m = bonsai.o liteuart.o
bonsai-objs = main.o

all: bonsai.ko liteuart.ko

bonsai.ko: main.c litepcie.h config.h flags.h csr.h soc.h
	make -C $(KERNEL_PATH) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(shell pwd) modules

liteuart.ko: liteuart.c
	make -C $(KERNEL_PATH) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(shell pwd) modules

install: all
	sudo mkdir -p /lib/modules/$(KERNEL_VERSION)/extra
	sudo install -m 644 bonsai.ko liteuart.ko /lib/modules/$(KERNEL_VERSION)/extra/
	sudo depmod -a
	echo bonsai   | sudo tee    /etc/modules-load.d/bonsai.conf > /dev/null
	echo liteuart | sudo tee -a /etc/modules-load.d/bonsai.conf > /dev/null
	echo 'KERNEL=="bonsai*", MODE="0666"' | sudo tee /etc/udev/rules.d/99-bonsai.rules > /dev/null

uninstall:
	sudo rm -f /lib/modules/$(KERNEL_VERSION)/extra/bonsai.ko \
	           /lib/modules/$(KERNEL_VERSION)/extra/liteuart.ko \
	           /etc/modules-load.d/bonsai.conf /etc/udev/rules.d/99-bonsai.rules
	sudo depmod -a

clean:
	make -C $(KERNEL_PATH) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(shell pwd) clean
	rm -f *~

.PHONY: all install uninstall clean
