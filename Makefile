.PHONY: all

all: build_lightning

build_lightning:
	mkdir -p out
	nvcc -x cu -arch=sm_70 -dc lightning.cu -o out/lightning.o -g -G -O2
	nvcc -x cu -arch=sm_70 -dc apsf.cu -o out/apsf.o -g -G -O2
	g++ -c -std=c++17 -Wall camera.cpp -o out/camera.o -g -O2
	nvcc -arch=sm_70 out/lightning.o out/camera.o out/apsf.o -o out/lightning -lGL -lGLEW -lglfw -lfreeimage -G -O2 -g

release:
	nvcc -x cu -arch=sm_70 -dc lightning.cu -o out/lightning_opt.o -O2 -g
	nvcc -x cu -arch=sm_70 -dc apsf.cu -o out/apsf_opt.o -g -O2
	g++ -c -std=c++17 -Wall camera.cpp -o out/camera_opt.o -O2
	nvcc -arch=sm_70 out/lightning_opt.o out/camera_opt.o out/apsf_opt.o -o out/lightning_opt -lGL -lGLEW -lglfw -lfreeimage -O3 -g
