package data

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include <stdlib.h>
// #include "../cgotorch/cgotorch.h"
import "C"
import (
	"fmt"
	"unsafe"
)

// Dataset struct
type Dataset struct {
	T C.Dataset
}

// Transform interface
type Transform interface{}

// Normalize transform struct
type Normalize struct {
	mean, stddev float64
}

// NewMNIST returns MNIST dataset
func NewMNIST(dataRoot string, transforms []Transform) *Dataset {
	var dataset C.Dataset
	cstr := C.CString(dataRoot)
	defer C.free(unsafe.Pointer(cstr))
	MustNil(unsafe.Pointer(C.Dataset_MNIST(cstr, &dataset)))

	// cache transforms on dataset
	for _, v := range transforms {
		switch t := v.(type) {
		case *Normalize:
			C.Dataset_Normalize(&dataset, C.double(v.(*Normalize).mean), C.double(v.(*Normalize).stddev))
		default:
			panic(fmt.Sprintf("unsupposed transform type: %T", t))
		}
	}

	return &Dataset{dataset}
}

// Close Dataset to release the memory
func (d *Dataset) Close() {
	C.MNIST_Close(d.T)
}

// NewNormalize returns normalize transformer
func NewNormalize(mean float64, stddev float64) *Normalize {
	return &Normalize{mean, stddev}
}

// Loader struct
type Loader struct {
	T     C.DataLoader
	batch *Batch
	iter  C.Iterator
}

// Batch struct contains data and target
type Batch struct {
	Data   Tensor
	Target Tensor
}

// NewLoader returns Loader pointer
func NewLoader(dataset *Dataset, batchSize int64) *Loader {
	loader := C.MakeLoader(C.Dataset(dataset.T), C.int64_t(batchSize))
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

// NewBatch returns the batch data as Tensor slice
func NewBatch(iter C.Iterator) *Batch {
	var data C.Tensor
	var target C.Tensor
	C.Iterator_Batch(iter, &data, &target)
	SetTensorFinalizer((*unsafe.Pointer)(&data))
	SetTensorFinalizer((*unsafe.Pointer)(&target))
	return &Batch{
		Data:   Tensor{(*unsafe.Pointer)(&data)},
		Target: Tensor{(*unsafe.Pointer)(&target)},
	}
}

// Scan scans the batch from Loader
func (loader *Loader) Scan() bool {
	// make the previous batch object to be unreachable
	// to release the Tensor memory.
	loader.batch = nil
	GC()
	if loader.iter == nil {
		loader.iter = C.Loader_Begin(loader.T)
		loader.batch = NewBatch(loader.iter)
		return true
	}
	// returns false if no next iteration
	if C.Loader_Next(loader.T, loader.iter) == false {
		return false
	}
	loader.batch = NewBatch(loader.iter)
	return true
}

// Batch returns the batch data on the current iteration.
func (loader *Loader) Batch() *Batch {
	return loader.batch
}
