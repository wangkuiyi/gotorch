all: main

main:
	clang++ -std=c++14 \
	-I ${GOPATH}/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/include \
	-I ${GOPATH}/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/include/torch/csrc/api/include \
	-L ${GOPATH}/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/lib \
	main.cpp \
	-o mnist \
	-Wl,-rpath,${GOPATH}/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/lib \
	-pthread \
	-lc10 -ltorch -ltorch_cpu \
	-D_GLIBCXX_USE_CXX11_ABI=1

clean:
	rm -rf *.so
