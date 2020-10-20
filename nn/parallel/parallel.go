package parallel

// #cgo CFLAGS: -I ${SRCDIR}/../../ -I ${SRCDIR}../../cgotorch/libtorch/include
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch -Wl,-rpath ${SRCDIR}/../../cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/../../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"
import (
	"reflect"
	"unsafe"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
)

//export goModuleForward
func goModuleForward(m unsafe.Pointer, input C.Tensor) C.Tensor {
	module := (*(*interface{})(m)).(nn.IModule)
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
	return torch.Tensor{}
}
