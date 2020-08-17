all : libcgotorch.so

libcgotorch.so : torch.cc functional.cc init.cc optim.cc cgotorch.h ${LIBTORCH_DIR}
	rm -f libtorch
	ln -s ${LIBTORCH_DIR} libtorch
	${CXX} -std=c++14 \
	-I .. \
	-I libtorch/include \
	-I libtorch/include/torch/csrc/api/include \
	-L libtorch/lib \
	-fPIC \
	-shared \
	torch.cc functional.cc init.cc optim.cc \
	-o $@ ${INSTALL_NAME} \
	-Wl,-rpath,libtorch/lib \
	-Wl,-${LOAD} libtorch/lib/libc10.${LIB_SUFFIX} \
	-lc10 -ltorch -ltorch_cpu \
	-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}

clean:
	rm -rf *.so
