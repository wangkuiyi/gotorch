all : libcgotorch.so

srcs := $(wildcard *.cc)
hdrs := $(wildcard *.h)
objs := $(srcs:%.cc=%.o)

%.o: %.cc $(hdrs)
	${CXX} -std=c++14 \
	-I .. -I libtorch/include \
	-I libtorch/include/torch/csrc/api/include \
	-fPIC ${CUDA_FLAGS} \
	-O -c $< -o $@

libcgotorch.so: $(objs) ${LIBTORCH_DIR}
	${CXX} 	-L libtorch/lib \
	$(objs) \
	-shared \
	-o $@ ${INSTALL_NAME} \
	-Wl,-rpath,libtorch/lib \
	-Wl,-${LOAD} libtorch/lib/libc10.${LIB_SUFFIX} \
	-lc10 -ltorch -ltorch_cpu \
	-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} \

clean:
	rm -rf *.so
