#
# Basic vector matrix library tested on Ubuntu
#

CPPINCLUDE_DIRS =  -I.
CPPLIBS =
LIBS =

CPP = g++
CFLAGS = -Wall -c -I.
CPPFLAGS = -Wall $(INCLUDE_DIRS)
LFLAGS = -Wall 

CPPSOURCE = vectmat.cpp
CPPOUTFILE = vectmat
CPPOBJS = $(CPPSOURCE:.cpp=.o)

all: $(CPPOUTFILE)

$(CPPOUTFILE): $(CPPOBJS)
	$(CPP) $(CPPFLAGS) $(CPPOBJS) -o $(CPPOUTFILE) `pkg-config --libs opencv` $(CPPLIBS)

clean:
	rm -f *.o $(CPPOUTFILE)
