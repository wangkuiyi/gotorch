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
	var b C.ByteBuffer
	err := unsafe.Pointer(
		C.Tensor_Encode(C.Tensor(*t.T), (*C.ByteBuffer)(unsafe.Pointer(&b))))

	if err != nil {
		msg := C.GoString((*C.char)(err))
		C.FreeString((*C.char)(err))
		return nil, fmt.Errorf(msg)
	}

	bs := C.GoBytes(unsafe.Pointer(C.ByteBuffer_Data(b)), C.int(int(int64(C.ByteBuffer_Size(b)))))
	C.ByteBuffer_Free(b)
	return bs, nil
}
