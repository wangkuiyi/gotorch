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

// Dataset interface
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
	cstr := C.CString(dataRoot)
	defer C.free(unsafe.Pointer(cstr))
	return &Dataset{C.MNIST(cstr)}
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
			C.AddNormalize(d.T, (C.Transform)(trans.(*Normalize).T))
		case *Stack:
			C.AddStack(d.T, (C.Transform)(trans.(*Stack).T))
		default:
			panic(fmt.Sprintf("unsupposed transform type: %T", v))
		}
	}
}

// DataLoader struct
type DataLoader struct {
	T     C.DataLoader
	iter  C.Iterator
	batch *Batch
}

// Batch struct
type Batch struct {
	Data   Tensor
	Target Tensor
}

// NewDataLoader returns DataLoader pointer
func NewDataLoader(dataset Dataset, batchSize int) *DataLoader {
	loader := C.DataLoaderWithSequenceSampler(C.Dataset(dataset.T), C.int(batchSize))
	return &DataLoader{
		T:     loader,
		iter:  nil,
		batch: &Batch{},
	}
}

// NewBatch returns Batch pointer
func NewBatch(batch *C.Tensor) *Batch {
	// TODO
	return &Batch{}
}

// Scan scans the batch from DataLoader
func (loader *DataLoader) Scan() bool {
	if loader.iter == nil {
		loader.iter = C.Begin(loader.T)
		loader.batch = NewBatch(C.Batch(loader.iter))
		return true
	}
	if C.IsEOF(loader.T, loader.iter) {
		return false
	}
	C.Next(loader.iter)
	loader.batch = NewBatch(C.Batch(loader.iter))
	return true
}

// Batch returns the batch on current interator
func (loader *DataLoader) Batch() *Batch {
	return loader.batch
}
