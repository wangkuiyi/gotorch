package vision

// #cgo CFLAGS: -I ${SRCDIR}/../cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch -Wl,-rpath ${SRCDIR}/../cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include <stdlib.h>
// #include "../cgotorch/cgotorch.h"
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/wangkuiyi/gotorch"
)

// Dataset wraps C.DataSet
type Dataset struct {
	T C.Dataset
}

// Transform interface
type Transform interface{}

// Close the Dataset and release memory.
func (d *Dataset) Close() {
	// FIXME: Currently, Dataset corresponds to MNIST dataset.
	C.MNIST_Close(d.T)
}

// MNIST corresponds to torchvision.datasets.MNIST.
func MNIST(dataRoot string, transforms []Transform) *Dataset {
	var dataset C.Dataset
	cstr := C.CString(dataRoot)
	defer C.free(unsafe.Pointer(cstr))
	gotorch.MustNil(unsafe.Pointer(C.Dataset_MNIST(cstr, &dataset)))

	// cache transforms on dataset
	for _, v := range transforms {
		switch t := v.(type) {
		case *NormalizeTransformer:
			C.Dataset_Normalize(&dataset,
				C.double(v.(*NormalizeTransformer).mean),
				C.double(v.(*NormalizeTransformer).stddev))
		default:
			panic(fmt.Sprintf("unsupposed transform type: %T", t))
		}
	}

	return &Dataset{dataset}
}

// Loader struct
type Loader struct {
	T     C.DataLoader
	batch *Batch
	iter  C.Iterator
}

// Batch struct contains data and target
type Batch struct {
	Data   gotorch.Tensor
	Target gotorch.Tensor
}

// NewLoader returns Loader pointer
func NewLoader(dataset *Dataset, batchSize int64) *Loader {
	loader := C.MakeDataLoader(C.Dataset(dataset.T), C.int64_t(batchSize))
	return &Loader{
		T:     loader,
		batch: nil,
		iter:  nil,
	}
}

// Close Loader
func (loader *Loader) Close() {
	C.Loader_Close(loader.T)
}

// minibatch returns the batch data as Tensor slice
func minibatch(iter C.Iterator) *Batch {
	var data C.Tensor
	var target C.Tensor
	C.Iterator_Batch(iter, &data, &target)
	gotorch.SetTensorFinalizer((*unsafe.Pointer)(&data))
	gotorch.SetTensorFinalizer((*unsafe.Pointer)(&target))
	return &Batch{
		Data:   gotorch.Tensor{(*unsafe.Pointer)(&data)},
		Target: gotorch.Tensor{(*unsafe.Pointer)(&target)},
	}
}

// Scan scans the batch from Loader
func (loader *Loader) Scan() bool {
	// make the previous batch object to be unreachable
	// to release the Tensor memory.
	loader.batch = nil
	gotorch.GC()
	if loader.iter == nil {
		loader.iter = C.Loader_Begin(loader.T)
		loader.batch = minibatch(loader.iter)
		return true
	}
	// returns false if no next iteration
	if C.Loader_Next(loader.T, loader.iter) == false {
		return false
	}
	loader.batch = minibatch(loader.iter)
	return true
}

// Batch returns the batch data on the current iteration.
func (loader *Loader) Batch() *Batch {
	return loader.batch
}
