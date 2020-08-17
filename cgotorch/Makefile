all : libcgotorch.so

libcgotorch.so : torch.cc functional.cc init.cc optim.cc cgotorch.h ${LIBTORCH}
	rm -f libtorch
	ln -s ${LIBTORCH} libtorch
	g++ -std=c++14 \
	-I .. \
	-I libtorch/include \
	-I libtorch/include/torch/csrc/api/include \
	-L libtorch/lib \
	-fPIC \
	-shared \
	torch.cc functional.cc init.cc optim.cc \
	-o $@ -install_name @rpath/$@ \
	-Wl,-rpath,libtorch/lib \
	-Wl,-all_load libtorch/lib/libc10.dylib \
	-lc10 -ltorch -ltorch_cpu

clean:
	rm -rf *.so
