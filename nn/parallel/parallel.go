package parallel

// #cgo CFLAGS: -I ${SRCDIR}/../../ -I ${SRCDIR}../../cgotorch/libtorch/include
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch -Wl,-rpath ${SRCDIR}/../../cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/../../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
// Tensor goModuleForward(char *m, Tensor input);
import "C"
import (
	"reflect"
	"runtime"
	"unsafe"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
)

//export goModuleForward
func goModuleForward(m *C.char, input C.Tensor) C.Tensor {
	module := (*(*nn.IModule)(unsafe.Pointer(m)))
	forward := reflect.ValueOf(module).MethodByName("Forward")
	args := []reflect.Value{reflect.ValueOf(torch.Tensor{(*unsafe.Pointer)(&input)})}
	return C.Tensor(forward.Call(args)[0].Interface().(torch.Tensor).T)
}

// DataParallel Evaluates module(input) in parallel across the given devices.
// If `devices` is not supplied, the invocation is parallelized across all available CUDA devices.
// If `outputDevice` is supplied, the final, combined tensor will be placed on this device. If not, it defaults to the first device in devices.
// In detail, this method performs the following four distinct steps:
//    1. Scatter the input to the given devices,
//    2. Replicate (deep clone) the model on each device,
//    3. Evaluate each module with its input on its device,
//    4. Gather the outputs of each replica into a single output tensor, located on the `outputDevice`.
func DataParallel(m nn.IModule, input torch.Tensor, devices []torch.Device, outputDevice torch.Device, dim int64) torch.Tensor {
	// Convert `m` to `*C.char` to workaround the "cgo argument has Go pointer to Go
	// pointer" check
	torch.MustNil(unsafe.Pointer(C.DataParallel((*C.char)(unsafe.Pointer(&m)), C.goModuleForward, C.Tensor(input.T), nil, 0, nil, 0)))
	runtime.KeepAlive(&m)
	return torch.Tensor{}
}
