CC = nvcc
COMPILER_FLAGS = -std=c++11 --compiler-options='`pkg-config libpng --cflags` `sdl2-config --cflags` -Wall -pthread -c -O3'
LINKER_FLAGS = `pkg-config libpng --libs` `sdl2-config --libs` -lSDL2 -lpthread -lcuda
OBJS = ray.o world.o png_wrapper.o
TARGET = ray.app
CFLAGS += -D LINUX

all: $(TARGET)
clean:
	-rm *.o *.app

$(TARGET): $(OBJS)
	nvcc -arch=sm_52 $(LINKER_FLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp
	nvcc -x cu $(COMPILER_FLAGS) -arch=sm_52 -D_FORCE_INLINES -D_CRT_SECURE_NO_DEPRECATE -I $(CUDA_HOME)/samples/common/inc -dc $< -o $@

