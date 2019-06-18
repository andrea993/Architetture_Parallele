mandelbrot.jar: ParLoop.class Mandelbrot.class manifest.txt
	jar cfm mandelbrot.jar manifest.txt -C mandelbrotMT .
Mandelbrot.class ParLoop.class: Mandelbrot.java
	javac Mandelbrot.java ParLoop.java -d mandelbrotMT
clean:
	rm -f *.jar \
rm -rf mandelbrotMT \
rm -f out.bmp



