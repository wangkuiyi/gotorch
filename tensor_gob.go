package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
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

// GobDecodeTensor decodes a tensor from []byte.  We don't define
// Tenosr.GobDecode because we want to create a new tensor instead of overwrite
// an existing one.  It is easier to manage the GC of a new tensor.
func GobDecodeTensor(buf []byte) (Tensor, error) {
	var t C.Tensor
	MustNil(unsafe.Pointer(
		C.Tensor_Decode(C.CBytes(buf), C.int64_t(int64(len(buf))), &t)))

	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}, nil
}
