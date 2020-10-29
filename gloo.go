package gotorch

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -lgloo -lc10d -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"
import "unsafe"

// FileStore struct
type FileStore struct {
	FS *C.FileStore
}

// NewFileStore creates a FileStore instance
func NewFileStore(path string, size int64) FileStore {
	var t C.FileStore
	MustNil(unsafe.Pointer(C.Gloo_NewFileStore(C.CString(path), C.int64_t(size), &t)))
	return FileStore{&t}
}

// ProcessGroupGloo struct
type ProcessGroupGloo struct {
	PGG *C.ProcessGroupGloo
}

// NewProcessGroupGloo creates a ProcessGroupGloo instance
func NewProcessGroupGloo(fs FileStore, rank, size int64) ProcessGroupGloo {
	var t C.ProcessGroupGloo
	MustNil(unsafe.Pointer(C.Gloo_NewProcessGroupGloo(*fs.FS, C.int64_t(rank), C.int64_t(size), &t)))
	return ProcessGroupGloo{&t}
}

// AllReduce sum tensors
func (ProcessGroupGloo pg) AllReduce(tensors []Tensor) {
	CT := []C.Tensor{}
	for _, t := range tensors {
		CT = append(CT, C.Tensor(*t.T))
	}
	p := (*C.Tensor)(unsafe.Pointer(&CT[0]))
	MustNil(unsafe.Pointer(C.Gloo_allreduce(*pg.PGG, p, C.int64_t(len(CT)))))
}
