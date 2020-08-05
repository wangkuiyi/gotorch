package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include <stdlib.h>
// #include "cgotorch.h"
import "C"
import (
	"fmt"
	"reflect"
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
	cstr := C.CString(dataRoot)
	defer C.free(unsafe.Pointer(cstr))
	return &Dataset{C.MNIST(cstr)}
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
	T    C.DataLoader
	data []Tensor
	iter C.Iterator
}

// Data struct which contains Data and Target sample
type Data struct {
	Data   Tensor
	Target Tensor
}

// NewDataLoader returns DataLoader pointer
func NewDataLoader(dataset *Dataset, batchSize int) *DataLoader {
	loader := C.MakeDataLoader(C.Dataset(dataset.T), C.int(batchSize))
	return &DataLoader{
		T:    loader,
		data: []Tensor{},
		iter: nil,
	}
}

// Close DataLoader
func (loader *DataLoader) Close() {
	C.Loader_Close(loader.T)
}

// NewData returns the batch data as Tensor slice
func NewData(iter C.Iterator) []Tensor {
	T := make([]C.Tensor, 2)
	p := (*reflect.SliceHeader)(unsafe.Pointer(&T)).Data
	C.Loader_Data(iter, (*C.Tensor)(unsafe.Pointer(p)))
	setTensorArrayFinalizer(T)
	return []Tensor{Tensor{(*C.Tensor)(T[0])}, Tensor{(*C.Tensor)(T[1])}}
}

// Scan scans the batch from DataLoader
func (loader *DataLoader) Scan() bool {
	if loader.iter == nil {
		loader.iter = C.Loader_Begin(loader.T)
		loader.data = NewData(loader.iter)
	}
	// returns false if no next iteration
	if C.Loader_Next(loader.T, loader.iter) == false {
		return false
	}
	loader.data = NewData(loader.iter)
	return true
}

// Data returns the data in DataLoader
func (loader *DataLoader) Data() []Tensor {
	return loader.data
}
