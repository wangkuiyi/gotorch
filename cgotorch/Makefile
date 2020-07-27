all : libcgotorch.so

libcgotorch.so : cgotorch.cc cgotorch.h
	clang++ -v -std=c++14 \
	-I ${TORCHPATH}/include \
	-I ${TORCHPATH}/include/torch/csrc/api/include \
	-fPIC \
	-shared \
	$< \
	-o $@ \
	-L${TORCHPATH}/lib -lc10 -ltorch -ltorch_cpu

clean:
	rm -rf *.so
