INCLUDE_DIRS = -I /home/ssiewert/Halide/include
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= -lrt
CPPLIBS= -L /home/ssiewert/Halide/bin -lHalide -lpthread -ldl -lpng

HFILES= 
CFILES= 
CPPFILES= brighten.cpp

SRCS= ${HFILES} ${CFILES}
CPPOBJS= ${CPPFILES:.cpp=.o}

all:	brighten

clean:
	-rm -f *.o *.d
	-rm -f brighten

distclean:
	-rm -f *.o *.d

brighten: brighten.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o $(CPPLIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

.cpp.o:
	$(CC) $(CFLAGS) -c $<
