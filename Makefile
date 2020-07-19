all : 01-backward 02-learn 03-adam

backward : 01-backward.cc libtorch
learn : 02-learn.cc libtorch
adam : 03-adam.cc libtorch

% : %.cc libtorch
	clang++ -std=c++14 \
	-I libtorch/include \
	-I libtorch/include/torch/csrc/api/include \
	-L libtorch/lib \
	$< \
	-o $@ \
	-Wl,-rpath,libtorch/lib \
	-Wl,-all_load libtorch/lib/libc10.dylib \
	libtorch/lib/libc10.dylib \
	libtorch/lib/libtorch.dylib \
	libtorch/lib/libtorch_cpu.dylib

libtorch-macos-1.5.1.zip :
	wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip

libtorch : libtorch-macos-1.5.1.zip
	unzip -qq -o $<
