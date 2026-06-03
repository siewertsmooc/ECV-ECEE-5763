all: run

run: faceDetect
	./faceDetect --cascade="haarcascade_frontalface_alt.xml" --nested-cascade="haarcascade_eye.xml" lena.jpg

faceDetect: faceDetect.cpp
	g++ faceDetect.cpp $(shell pkg-config --cflags --libs opencv4) -o faceDetect

clean:
	rm faceDetect
