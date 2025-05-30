# Specify the path to your CUDA installation
CUDA_PATH ?= /usr/local/cuda

# Compiler
CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc

# Include directories for CUDA and OpenCV
INCLUDES := -I$(CUDA_PATH)/include $(shell pkg-config --cflags opencv4)

# Library paths for CUDA and OpenCV
LIBRARIES := -L$(CUDA_PATH)/lib64 $(shell pkg-config --libs opencv4)

# Compiler flags
CXXFLAGS := -std=c++11 -Wall
NVCCFLAGS := -std=c++11 -Xcompiler -Wall

# Architecture-specific flags
SMS ?= 50 52 61 75
GENCODE_FLAGS := $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))

# Targets
TARGET1 := faceblur
TARGET2 := playground
TARGET3 := playground_driver
TARGET4 := fov

all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)

$(TARGET1): faceblur.o utilities.o
	$(CXX) $^ $(LIBRARIES) -o $@

$(TARGET2): playground.o utilities.o
	$(CXX) $^ $(LIBRARIES) -o $@

$(TARGET3): playground_driver.o utilities.o
	$(CXX) $^ $(LIBRARIES) -o $@

$(TARGET4): fov.o
	$(CXX) $^ $(LIBRARIES) -o $@

%.o: %.cpp
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) *.o

run1: $(TARGET1)
	./$(TARGET1)

run2: $(TARGET2)
	./$(TARGET2)

.PHONY: all clean run1 run2

