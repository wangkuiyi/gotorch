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
	data *Data
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
		data: nil,
		iter: nil,
	}
}

// NewData returns Data in DataLoader
func NewData(data C.Data) *Data {
	return &Data{
		Data:   Tensor{&data.Data},
		Target: Tensor{&data.Target},
	}
}

// Scan scans the batch from DataLoader
func (loader *DataLoader) Scan() bool {
	if loader.iter == nil {
		loader.iter = C.Loader_Begin(loader.T)
		loader.data = NewData(C.Loader_Data(loader.iter))
	}
	// returns false if no next iteration
	if C.Loader_Next(loader.T, loader.iter) == false {
		return false
	}
	C.Loader_Data(loader.iter)
	return true
}

// Data returns the data in DataLoader
func (loader *DataLoader) Data() *Data {
	return loader.data
}
