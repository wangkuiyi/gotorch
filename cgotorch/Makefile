all : libcgotorch.so

libtorch-macos-1.5.1.zip :
	curl -LsO https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip

macos/libtorch : libtorch-macos-1.5.1.zip
	unzip -qq -o $< -d macos

libcgotorch.so : cgotorch.cc cgotorch.h macos/libtorch
	rm -rf libtorch/
	ln -s macos/libtorch libtorch
	clang++ -std=c++14 \
	-I .. \
	-I libtorch/include \
	-I libtorch/include/torch/csrc/api/include \
	-L libtorch/lib \
	-fPIC \
	-shared \
	cgotorch.cc \
	-o $@ -install_name @rpath/$@ \
	-Wl,-rpath,libtorch/lib \
	-Wl,-all_load libtorch/lib/libc10.dylib \
	-lc10 -ltorch -ltorch_cpu

clean:
	rm -rf *.so
