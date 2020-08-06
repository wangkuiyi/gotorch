# How to Contriubte

## How to Build on macOS

Retrieve the source code into the directory `$GOPATH/src/github.com/wangkuiyi/gotorch`.

```bash
go get github.com/wangkuiyi/gotorch
```

Build the CGO binding of libtorch.

```bash
cd $GOPATH/src/github.com/wangkuiyi/gotorch/cgotorch
make
```

Run the Go examples and unit tests.

```bash
cd ..
go test -v
```