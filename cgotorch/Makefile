all : libcgotorch.so

libtorch.zip:
	./download_libtorch.sh

libtorch : libtorch.zip
	unzip -qq -o $<

libcgotorch.so : cgotorch.cc cgotorch.h libtorch
	clang++ -std=c++14 \
	-I libtorch/include \
	-I libtorch/include/torch/csrc/api/include \
	-L libtorch/lib \
	-fPIC \
	-shared \
	$< \
	-o $@ -install_name @rpath/$@ \
	-Wl,-rpath,libtorch/lib \
	-Wl,-all_load libtorch/lib/libc10.* \
	-lc10 -ltorch -ltorch_cpu

clean:
	rm -rf *.so
