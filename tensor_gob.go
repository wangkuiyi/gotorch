package gotorch

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// GobEncode calls C.Tensor_Encode to encode a tensor into a pickle.  If the
// tensor is on GPU, C.Tenosr_Encode clones it in CPU before encoding, so the
// result always encodes a CPU tensor.
func (t Tensor) GobEncode() ([]byte, error) {
	if t.T == nil {
		return nil, fmt.Errorf("Cannot encode nil tensor")
	}

	var b C.ByteBuffer
	MustNil(unsafe.Pointer(
		C.Tensor_Encode(C.Tensor(*t.T), (*C.ByteBuffer)(unsafe.Pointer(&b)))))

	bs := C.GoBytes(C.ByteBuffer_Data(b), C.int(int(int64(C.ByteBuffer_Size(b)))))
	C.ByteBuffer_Free(b)
	return bs, nil
}

// GobDecode makes Tensor implements the gob.GobDecoder interface.
func (t *Tensor) GobDecode(buf []byte) error {
	var n C.Tensor
	MustNil(unsafe.Pointer(
		C.Tensor_Decode(C.CBytes(buf), C.int64_t(int64(len(buf))), &n)))
	SetTensorFinalizer((*unsafe.Pointer)(&n))
	*t = Tensor{T: (*unsafe.Pointer)(&n)}
	return nil
}
