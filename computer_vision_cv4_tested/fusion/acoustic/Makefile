INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt

HFILES= 
CFILES= record.c play.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all:	record play

clean:
	-rm -f *.o *.d
	-rm -f record play

play: play.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lasound

record: record.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lasound

depend:

.cpp.o: $(SRCS)
	$(CC) $(CFLAGS) -c $<
