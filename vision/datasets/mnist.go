package datasets

// #cgo CFLAGS: -I ${SRCDIR}/../../cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch -Wl,-rpath ${SRCDIR}/../../cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/../../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include <stdlib.h>
// #include "../../cgotorch/cgotorch.h"
import "C"
import (
	"fmt"
	"log"
	"unsafe"

	"github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

// MNISTDataset wraps C.MNISTDataSet
type MNISTDataset struct {
	dataset C.MNISTDataset
}

// Close the Dataset and release memory.
func (d *MNISTDataset) Close() {
	// FIXME: Currently, Dataset corresponds to MNIST dataset.
	C.MNISTDataset_Close(d.dataset)
}

// MNIST corresponds to torchvision.datasets.MNIST.
func MNIST(dataRoot string, trans []transforms.Transform) *MNISTDataset {
	dataRoot = cacheDir(dataRoot)
	if e := downloadMNIST(dataRoot); e != nil {
		log.Fatalf("Failed to download MNIST dataset: %v", e)
	}

	var dataset C.MNISTDataset
	cstr := C.CString(dataRoot)
	defer C.free(unsafe.Pointer(cstr))
	gotorch.MustNil(unsafe.Pointer(C.CreateMNISTDataset(cstr, &dataset)))

	// cache transforms on dataset
	for _, v := range trans {
		switch t := v.(type) {
		case *transforms.NormalizeTransformer:
			trans := v.(*transforms.NormalizeTransformer)
			C.MNISTDataset_Normalize(&dataset,
				(*C.double)(unsafe.Pointer(&trans.Mean[0])),
				C.int64_t(len(trans.Mean)),
				(*C.double)(unsafe.Pointer(&trans.Stddev[0])),
				C.int64_t(len(trans.Stddev)))
		default:
			panic(fmt.Sprintf("unsupposed transform type: %dataset", t))
		}
	}

	return &MNISTDataset{dataset}
}

// MNISTLoader struct
type MNISTLoader struct {
	loader C.MNISTLoader
	batch  *Batch
	iter   C.MNISTIterator
}

// Batch struct contains data and target
type Batch struct {
	Data   gotorch.Tensor
	Target gotorch.Tensor
}

// NewMNISTLoader returns Loader pointer
func NewMNISTLoader(dataset *MNISTDataset, batchSize int64) *MNISTLoader {
	return &MNISTLoader{
		loader: C.CreateMNISTLoader(
			C.MNISTDataset(dataset.dataset), C.int64_t(batchSize)),
		batch: nil,
		iter:  nil,
	}
}

// Close Loader
func (loader *MNISTLoader) Close() {
	C.MNISTLoader_Close(loader.loader)
	C.MNISTIterator_Close(loader.iter)
}

// minibatch returns the batch data as Tensor slice
func minibatch(iter C.MNISTIterator) *Batch {
	var data C.Tensor
	var target C.Tensor
	C.MNISTIterator_Batch(iter, &data, &target)
	gotorch.SetTensorFinalizer((*unsafe.Pointer)(&data))
	gotorch.SetTensorFinalizer((*unsafe.Pointer)(&target))
	return &Batch{
		Data:   gotorch.Tensor{(*unsafe.Pointer)(&data)},
		Target: gotorch.Tensor{(*unsafe.Pointer)(&target)},
	}
}

// Scan scans the batch from Loader
func (loader *MNISTLoader) Scan() bool {
	// make the previous batch object to be unreachable
	// to release the Tensor memory.
	loader.batch = nil
	gotorch.GC()
	if loader.iter == nil {
		loader.iter = C.MNISTLoader_Begin(loader.loader)
		loader.batch = minibatch(loader.iter)
		return true
	}
	// returns false if no next iteration
	if C.MNISTIterator_Next(loader.iter, loader.loader) == false {
		return false
	}
	loader.batch = minibatch(loader.iter)
	return true
}

// Batch returns the batch data on the current iteration.
func (loader *MNISTLoader) Batch() *Batch {
	return loader.batch
}
