all : libcgotorch.so		#01-backward 02-learn 03-adam

libtorch-macos-1.5.1.zip :
	wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip

libtorch : libtorch-macos-1.5.1.zip
	unzip -qq -o $<

libcgotorch.so : ctorch.cc ctorch.h libtorch
	clang++ -std=c++14 \
	-I libtorch/include \
	-I libtorch/include/torch/csrc/api/include \
	-L libtorch/lib \
	-fPIC \
	-shared \
	$< \
	-o $@ \
	-Wl,-rpath,libtorch/lib \
	-Wl,-all_load libtorch/lib/libc10.dylib \
	libtorch/lib/libc10.dylib \
	libtorch/lib/libtorch.dylib \
	libtorch/lib/libtorch_cpu.dylib
