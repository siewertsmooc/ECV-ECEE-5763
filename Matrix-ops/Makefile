INCLUDE_DIRS = 
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= genmatrixops.cpp geeksrotate.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	genmatrixops geeksrotate

clean:
	-rm -f *.o *.d
	-rm -f genmatrixops geeksrotate

distclean:
	-rm -f *.o *.d
	-rm -f genmatrixops geeksrotate

geeksrotate: geeksrotate.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

genmatrixops: genmatrixops.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

depend:

.cpp.o:
	$(CC) $(CFLAGS) -c $<
