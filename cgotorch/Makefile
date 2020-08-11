all : libcgotorch.so

libtorch-macos-1.6.0.zip :
	curl -LsO https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip

macos/libtorch : libtorch-macos-1.6.0.zip
	unzip -qq -o $< -d macos

libcgotorch.so : torch.cc functional.cc init.cc optim.cc cgotorch.h macos/libtorch
	rm -f libtorch
	ln -s macos/libtorch libtorch
	clang++ -std=c++14 \
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
