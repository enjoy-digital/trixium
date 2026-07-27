CFLAGS=-O2 -Wall -g -I../kernel -Iliblitepcie -MMD -fPIC
LDFLAGS=-g
CC=$(CROSS_COMPILE)gcc
AR=ar

PROGS=bonsai_util bonsai_load

all: $(PROGS)

liblitepcie/liblitepcie.a: liblitepcie/litepcie_dma.o liblitepcie/litepcie_flash.o liblitepcie/litepcie_helpers.o
	ar rcs $@ $+
	ranlib $@

bonsai_util: liblitepcie/liblitepcie.a bonsai_util.o
	$(CC) $(LDFLAGS) -o $@ $^ -Lliblitepcie -llitepcie

bonsai_load: liblitepcie/liblitepcie.a bonsai_load.o
	$(CC) $(LDFLAGS) -o $@ $^ -Lliblitepcie -llitepcie

clean:
	rm -f $(PROGS) *.o *.a *.d *~ liblitepcie/*.a liblitepcie/*.o liblitepcie/*.d

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

liblitepcie/%.o: liblitepcie/%.c
	$(CC) -c $(CFLAGS) -o $@ $<

-include $(wildcard *.d)
