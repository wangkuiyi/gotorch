all: mnist

mnist: main.cpp
	clang++ -std=c++14 \
	-I ../cgotorch/libtorch/include \
	-I ../cgotorch/libtorch/include/torch/csrc/api/include \
	-L ../cgotorch/libtorch/lib \
	main.cpp \
	-O -o mnist \
	-Wl,-rpath,../cgotorch/libtorch/lib \
	-pthread \
	-lc10 -ltorch -ltorch_cpu \
	-D_GLIBCXX_USE_CXX11_ABI=1

clean:
	rm mnist
