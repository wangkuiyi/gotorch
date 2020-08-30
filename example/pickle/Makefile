pickle : pickle.cc
	g++ -std=c++14 \
	$< -o $@ \
	-I ../../cgotorch \
	-I ../../cgotorch/libtorch/include \
	-I /usr/local/opt/openssl/include \
	-I ../../cgotorch/libtorch/include/torch/csrc/api/include \
	-L ../../cgotorch -Wl,-rpath ../../cgotorch -lcgotorch \
	-L /usr/local/opt/openssl/lib/ -lssl -lcrypto \
	-L ../../cgotorch/libtorch/lib -Wl,-rpath ../../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
