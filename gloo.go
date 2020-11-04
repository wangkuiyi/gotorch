package gotorch

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -lgloo -lc10d -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"
import "unsafe"

// Store struct
type Store struct {
	Store *C.Store
}

// NewFileStore creates a FileStore instance
func NewFileStore(path string, size int64) Store {
	var t C.Store
	MustNil(unsafe.Pointer(C.Gloo_NewFileStore(C.CString(path), C.int64_t(size), &t)))
	return Store{&t}
}

// NewTCPStore creates a TCPStore instance
func NewTCPStore(addr string, port, size int64, isServer bool) Store {
	is := 0
	if isServer {
		is = 1
	}
	var t C.Store
	MustNil(unsafe.Pointer(C.Gloo_NewTCPStore(C.CString(addr), C.int64_t(port), C.int64_t(size), C.int64_t(is), &t)))
	return Store{&t}
}

// Close a store
func (s Store) Close() {
	MustNil(unsafe.Pointer(C.Gloo_DeleteStore(*s.Store)))
	s.Store = nil
}

// ProcessGroupGloo struct
type ProcessGroupGloo struct {
	PGG *C.ProcessGroupGloo
}

// NewProcessGroupGloo creates a ProcessGroupGloo instance
func NewProcessGroupGloo(s Store, rank, size int64) ProcessGroupGloo {
	var t C.ProcessGroupGloo
	MustNil(unsafe.Pointer(C.Gloo_NewProcessGroupGloo(*s.Store, C.int64_t(rank), C.int64_t(size), &t)))
	return ProcessGroupGloo{&t}
}

// Close a group
func (pg ProcessGroupGloo) Close() {
	MustNil(unsafe.Pointer(C.Gloo_DeleteProcessGroupGloo(*pg.PGG)))
	pg.PGG = nil
}

// AllReduce method: todo(qijun) only support sum
func (pg ProcessGroupGloo) AllReduce(tensors []Tensor) {
	CT := []C.Tensor{}
	for _, t := range tensors {
		CT = append(CT, C.Tensor(*t.T))
	}
	p := (*C.Tensor)(unsafe.Pointer(&CT[0]))
	MustNil(unsafe.Pointer(C.Gloo_allreduce(*pg.PGG, p, C.int64_t(len(CT)))))
}

// AllReduceCoalesced method: tensors will be flattened and
// concatenated (coalesced). This means that input tensors
// must have the same device, layout and type.
func (pg ProcessGroupGloo) AllReduceCoalesced(tensors []Tensor) {
	CT := []C.Tensor{}
	for _, t := range tensors {
		CT = append(CT, C.Tensor(*t.T))
	}
	p := (*C.Tensor)(unsafe.Pointer(&CT[0]))
	MustNil(unsafe.Pointer(C.Gloo_allreduce_coalesced(*pg.PGG, p, C.int64_t(len(CT)))))
}

// Broadcast method
func (pg ProcessGroupGloo) Broadcast(tensors []Tensor) {
	CT := []C.Tensor{}
	for _, t := range tensors {
		CT = append(CT, C.Tensor(*t.T))
	}
	p := (*C.Tensor)(unsafe.Pointer(&CT[0]))
	MustNil(unsafe.Pointer(C.Gloo_broadcast(*pg.PGG, p, C.int64_t(len(CT)))))
}
