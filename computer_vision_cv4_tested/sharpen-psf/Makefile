INCLUDE_DIRS = 
LIB_DIRS = 
CC = gcc

CDEFS=
#CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O0 -msse3 -malign-double $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O2 -msse3 -malign-double $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 $(INCLUDE_DIRS) $(CDEFS)
CFLAGS= -O3 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 -msse3 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 -mssse3 $(INCLUDE_DIRS) $(CDEFS)
LIBS=-lpthread

PRODUCT=sharpen_grid sharpen

HFILES= 
CFILES= sharpen_grid.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

sharpen: sharpen.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ sharpen.o $(LIBS)

sharpen_grid: sharpen_grid.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ sharpen_grid.o $(LIBS)


depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
