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

	gotorch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/data"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

// MNISTDataset wraps C.MNISTDataSet
type MNISTDataset struct {
	dataset C.MNISTDataset
	loader  C.MNISTLoader
	iter    C.MNISTIterator
}

// Close the Dataset and release memory.
func (d *MNISTDataset) Close() {
	// FIXME: Currently, Dataset corresponds to MNIST dataset.
	C.MNISTDataset_Close(d.dataset)
	C.MNISTLoader_Close(d.loader)
	C.MNISTIterator_Close(d.iter)
}

// MNIST corresponds to torchvision.datasets.MNIST.
func MNIST(dataRoot string, trans []transforms.Transform, batchSize int64) *MNISTDataset {
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
	loader := C.CreateMNISTLoader(dataset, C.int64_t(batchSize))
	return &MNISTDataset{
		dataset: dataset,
		loader:  loader,
		iter:    C.MNISTLoader_Begin(loader)}
}

// Get fetch a batch of examples and collate to one example
func (d *MNISTDataset) Get() *data.Example {
	if C.MNISTIterator_IsEnd(d.iter, d.loader) {
		return nil
	}
	var x, y unsafe.Pointer
	C.MNISTIterator_Batch(d.iter, (*C.Tensor)(&x), (*C.Tensor)(&y))
	C.MNISTIterator_Next(d.iter, d.loader)
	return data.NewExample(gotorch.Tensor{&x}, gotorch.Tensor{&y})
}

// Reset resets the status of the Dataset
func (d *MNISTDataset) Reset() {
	C.MNISTIterator_Close(d.iter)
	d.iter = C.MNISTLoader_Begin(d.loader)
}
