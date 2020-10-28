# AllReduce

## Introduction

Data parallelism enables distributed training by communicating gradients
before the optimizer step to make sure that parameters of all model replicas
are updated using exactly the same set of gradients,
and hence model replicas can stay consistent across iterations.

AllReduce is a common strategy for communicating gradients in data parallelism.
Following is the pseudocode describing the training procedures
under AllReduce strategy.

```python
broadcast(parameters, rank=0)
while True:
    load_minibatch()
    forward()
    backward()
    allreduce(gradients)
    update()
```

First, we broadcast model parameters of rank 0 to other processes.
Each process loads a minibatch of training data,
does forward/backward computation, and gets the gradients.
We launch AllReduce to communicate gradients among the processes.
At last, we update the parameters in each process individually.

## AllReduce in PyTorch

Before discussing how to support AllReduce in GoTorch,
it's necessary to have a thorough suvey on
the current implementation of AllReduce in PyTorch.

PyTorch offers several tools to facilitate distributed training,
including [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel)
for single-process multi-thread data parallel training
using multiple GPUs on the same machine,
[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)
for multi-process data parallel training
across GPUs and machines.

Single-process multi-GPU is not the recommended mode,
becase of its overhead of scatter/gather and GIL contention in every forward pass.
So, let's focus on DistributedDataParallel.

### Collective Communication Library

PyTorch could use different collective communication libraries as the backend,
including [NCCL](https://developer.nvidia.com/nccl) and [Gloo](https://github.com/facebookincubator/gloo).
NCCL supports GPU, while Gloo supports both CPU and GPU.
The performance on GPU of NCCL is better than Gloo.
So we use NCCL in GPU training, and Gloo in CPU training.

Besides, PyTorch provides a library, [c10d](https://github.com/pytorch/pytorch/tree/master/torch/lib/c10d),
which wrappers NCCL/Gloo, to manipulate `torch::Tensor` directly.
It brings much convenience.

Following is an example:

```cpp
#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroupGloo.hpp>

using namespace ::c10d;

int main(int argc, char** argv) {
  int rank = atoi(getenv("RANK"));
  int size = atoi(getenv("SIZE"));
  auto store = std::make_shared<FileStore>("/tmp/c10d_example", size);
  ProcessGroupGloo pg(store, rank, size);

  // Create some tensors
  const auto ntensors = 10;
  std::vector<at::Tensor> tensors;
  for (auto i = 0; i < ntensors; i++) {
    auto x =
        at::ones({1000, 16 * (i + 1)}, at::TensorOptions(at::CPU(at::kFloat)));
    tensors.push_back(x);
  }

  // Kick off work
  std::vector<std::shared_ptr<ProcessGroup::Work>> pending;
  for (auto i = 0; i < ntensors; i++) {
    std::vector<at::Tensor> tmp = {tensors[i]};
    pending.push_back(pg.allreduce(tmp));
  }

  // Wait for work to complete
  for (auto& work : pending) {
    work->wait();
  }
}
```

### DistributedSampler

The training samples are partitioned statically in distributed training of PyTorch.
The [DistributedSampler](https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler)
generates a sequence of indices of training samples for each training process.
Then, each process loads a subset samples by the indices.

**Note:** The dataset is assumed to be of constant size.

### Launch Utility

The `torch.distributed` package provides a launch utility in
[torch.distributed.launch](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py).
This helper utility can be used to
launch multiple processes per node for distributed training.
If the utility is used for GPU training,
each distributed process will be operating on a single GPU.

### Optimization

The naive implementation of training procedures
in [Introduction](#Introduction) section has two performance concerns:

- Collective communication performs poorly on
  small tensors, which will be especially prominent on large models
  with massive numbers of small parameters.
- Separating gradient computation and synchronization forfeits the opportunity
  to overlap computation with communication due to the hard boundary in between.

PyTorch does more optimizations to solve these two problems:

- Bucketing gradients to reduce AllReduce kernels overhead.
- Registering AllReduce kernels as autograd hooks
  to overlap communication and computation.

For more details, please refer to the [paper](https://arxiv.org/abs/2006.15704).

## AllReduce in GoTorch

We plan to implement the functionalities of
DistributedDataParallel gradually in GoTorch.
At stage 1, we provide a naive solution.
An MNIST distributed example is the target in this stage.
At stage 2, we will provide an optimized solution.
Bucketing gradients and registering hooks will be implemented at this stage.

### RecordIODataLoader

The RecordIO format is a simple format for a sequence of binary records.
It provides a way to seek the beginning of any record in a file.
We could partition the RecordIO data and assgin to training processes.
At stage 1, we support static sharding only.
Following are the steps of static sharding in distributed training:

1. Convert samples into RecordIO format.
1. Partition records into several tasks. Each task contains
   one or more `{file, start_idx, end_idx}` tuples.
1. Shuffle tasks and assign a subset of tasks to a training process.
1. Decode records in tasks and feed to the neural network.

### Go Wrapper of c10d Library

[ProcessGroupNCCL](https://github.com/pytorch/pytorch/blob/master/torch/lib/c10d/ProcessGroupNCCL.hpp)
implements NCCL bindings for c10d library.
After adding a Go wrapper of this class,
we could do allreduce on torch tensors in Go.

### Go Launch Utility

Go provides [os/exec](https://golang.org/pkg/os/exec/) library to spawn processes.

### Optimization at Stage 2

TBD

## Reference

- <https://pytorch.org/docs>
- <https://arxiv.org/abs/2006.15704>
