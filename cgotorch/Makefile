all : libcgotorch.so many_threads

srcs := $(wildcard *.cc)
hdrs := $(wildcard *.h)

libcgotorch.so : $(srcs) $(hdrs) ${LIBTORCH_DIR}
	rm -f libtorch
	ln -s ${LIBTORCH_DIR} libtorch
	${CXX} -std=c++14 \
	-I .. \
	-I libtorch/include \
	-I libtorch/include/torch/csrc/api/include \
	-L libtorch/lib \
	-fPIC \
	-shared \
	$(srcs) \
	-O -o $@ ${INSTALL_NAME} \
	-Wl,-rpath,libtorch/lib \
	-Wl,-${LOAD} libtorch/lib/libc10.${LIB_SUFFIX} \
	-lc10 -ltorch -ltorch_cpu \
	-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} \
	${CUDA_FLAGS}

many_threads : many_threads.cc
	${CXX} -fopenmp -std=c++14 $< -o $@ -O

clean:
	rm -rf *.so many_threads
