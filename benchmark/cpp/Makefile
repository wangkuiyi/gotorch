all: main

main:
	clang++ -std=c++14 \
	-I /go/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/include \
	-I /go/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/include/torch/csrc/api/include \
	-L /go/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/lib \
	main.cpp \
	-o mnist \
	-Wl,-rpath,/go/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/lib \
	-pthread \
	-lc10 -ltorch -ltorch_cpu \
	-D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm -rf *.so