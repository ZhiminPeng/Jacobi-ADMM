GSLROOT=/usr/local
# use this if on 64-bit machine with 64-bit GSL libraries
# ARCH=x86_64
# use this if on 32-bit machine with 32-bit GSL libraries
# ARCH=i386

MPICC=mpicc
CC=gcc
CFLAGS=-Wall -std=c99 -I$(GSLROOT)/include
LDFLAGS=-L$(GSLROOT)/lib -lgsl -lgslcblas -lm

all: jacobiADMM

jacobiADMM: jacobiADMM.o mmio.o
	$(MPICC) $(CFLAGS) jacobiADMM.o mmio.o -o jacobiADMM $(LDFLAGS)

#gam: gam.o mmio.o
#	$(MPICC) $(CFLAGS) $(LDFLAGS) gam.o mmio.o -o gam

jacobiADMM.o: jacobiADMM.c mmio.o
	$(MPICC) $(CFLAGS) -c jacobiADMM.c

#gam.o: gam.c mmio.o
#	$(MPICC) $(CFLAGS) -c gam.c

mmio.o: mmio.c
	$(CC) $(CFLAGS) -c mmio.c

clean:
	rm -vf *.o jacobiADMM
