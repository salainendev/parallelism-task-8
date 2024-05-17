INFO = -Minfo=all
LIBS = -cudalib=cublas -lboost_program_options
GPU = -acc=gpu
CXX = pgc++
all:main

main:clean
	nvcc -o $@ main.cu -I/cub/cub.cuh -lboost_program_options

.PHONY:force
force: main

clean:all
	rm main