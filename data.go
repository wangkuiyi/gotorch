package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include <stdlib.h>
// #include "cgotorch.h"
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
	T unsafe.Pointer
}

// Stack transform struct
type Stack struct {
	T unsafe.Pointer
}

// NewMNIST returns MNIST dataset
func NewMNIST(dataRoot string) *Dataset {
	var dataset C.Dataset
	cstr := C.CString(dataRoot)
	defer C.free(unsafe.Pointer(cstr))
	mustNil(C.MNIST(cstr, &dataset))
	return &Dataset{dataset}
}

// Close Dataset to release the memory
func (d *Dataset) Close() {
	C.MNIST_Close(d.T)
}

// NewNormalize returns normalize transformer
func NewNormalize(mean float64, stddev float64) *Normalize {
	return &Normalize{unsafe.Pointer(C.Normalize(C.double(mean), C.double(stddev)))}
}

// NewStack returns Stack tranformer
func NewStack() *Stack {
	return &Stack{unsafe.Pointer(C.Stack())}
}

// AddTransforms adds a slice of Transform
func (d *Dataset) AddTransforms(transforms []Transform) {
	for _, trans := range transforms {
		switch v := trans.(type) {
		case *Normalize:
			C.Dataset_Normalize(d.T, (C.Transform)(trans.(*Normalize).T))
		case *Stack:
			C.Dataset_Stack(d.T, (C.Transform)(trans.(*Stack).T))
		default:
			panic(fmt.Sprintf("unsupposed transform type: %T", v))
		}
	}
}

// DataLoader struct
type DataLoader struct {
	T     C.DataLoader
	batch *Batch
	iter  C.Iterator
}

// Batch struct contains data and target
type Batch struct {
	Data   Tensor
	Target Tensor
}

// NewDataLoader returns DataLoader pointer
func NewDataLoader(dataset *Dataset, batchSize int) *DataLoader {
	loader := C.MakeDataLoader(C.Dataset(dataset.T), C.int64_t(batchSize))
	return &DataLoader{
		T:     loader,
		batch: nil,
		iter:  nil,
	}
}

// Close DataLoader
func (loader *DataLoader) Close() {
	C.Loader_Close(loader.T)
}

// NewBatch returns the batch data as Tensor slice
func NewBatch(iter C.Iterator) *Batch {
	var data C.Tensor
	var target C.Tensor
	C.Iterator_Batch(iter, &data, &target)
	setTensorFinalizer(&data)
	setTensorFinalizer(&target)
	return &Batch{
		Data:   Tensor{&data},
		Target: Tensor{&target},
	}
}

// Scan scans the batch from DataLoader
func (loader *DataLoader) Scan() bool {
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
func (loader *DataLoader) Batch() *Batch {
	return loader.batch
}
