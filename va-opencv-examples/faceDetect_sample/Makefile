all: run

run: faceDetect
	./faceDetect --cascade="haarcascade_frontalface_alt.xml" --nested-cascade="haarcascade_eye.xml" lena.jpg

faceDetect: faceDetect.cpp
	g++ faceDetect.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -o faceDetect

clean:
	rm faceDetect
