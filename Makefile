demo : demo.cc libtorch/lib/libc10.dylib libtorch/lib/libtorch.dylib libtorch/lib/libtorch_cpu.dylib
	clang++ -std=c++14 \
	-I libtorch/include \
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

libtorch/lib/libc10.dylib : libtorch
libtorch/lib/libtorch.dylib : libtorch
libtorch/lib/libtorch_cpu.dylib : libtorch
