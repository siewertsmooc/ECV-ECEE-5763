INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) 
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt

HFILES= 
CPPFILES= BackgroundRemover.cpp FingerCount.cpp FaceDetector.cpp SkinDetector.cpp 
CFILES= main.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all:	main

clean:
	-rm -f *.o *.d
	-rm -f main detector

main  : 
	$(CC) $(CPPFILES) $(CFILES) $(CFLAGS) -o detector `pkg-config --libs opencv4` $(LIBS)

depend:

.cpp.o: $(SRCS)
	$(CC) $(CFLAGS) -c $<



