#ifndef __C_TORCH_H_
#define __C_TORCH_H_

#ifdef __cplusplus
extern "C" {
#endif

  typedef void* Tensor;
  Tensor RandN(int rows, int cols, int require_grad);
  Tensor MM(Tensor a, Tensor b);
  Tensor Sum(Tensor a);

  void Tensor_Backward(Tensor a);
  Tensor Tensor_Grad(Tensor a);
  void Tensor_Print(Tensor a);

#ifdef __cplusplus
}
#endif

#endif //__C_TORCH_H_
