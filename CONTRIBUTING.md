# How to Contribute

## Build and Test

Retrieve the source code into the directory `$GOPATH/src/github.com/wangkuiyi/gotorch` -- you can set GOPATH pointing to any directory by setting `export GOPATH=some/directory`.

```bash
go get github.com/wangkuiyi/gotorch
```

Build the CGO binding of `libtorch`.

```bash
$GOPATH/src/github.com/wangkuiyi/gotorch/cgotorch/build.sh
```

Run the Go examples and unit tests.

```bash
go test -v github.com/wangkuiyi/gotorch/...
```

The above `go test` command might fail and complain that it cannot find the `.so` or `.dylib` files.  If so, please run the following command.

```bash
export LD_LIBRARY_PATH=$GOPATH/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/lib:$LD_LIBRARY_PATH
```
