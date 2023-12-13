import torch
import torch.nn as nn
from torch.nn.parallel import parallel_apply, comm
from torch.nn.parallel._functions import Broadcast
from torch.distributed import all_reduce


class f(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, devices):
        x = Broadcast.apply(devices, x)
        return x

    @staticmethod    
    def backward(ctx, gradient):
        gradient = comm.reduce_add(gradient, ctx.devices[0])
        return gradient


class g(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, devices):
        ctx.devices = devices
        x = comm.reduce_add(x, devices[0])
        return x

    @staticmethod    
    def backward(ctx, gradient):
        gradient = Broadcast.apply(ctx.devices, gradient)
        return gradient


class ShardedMLP(nn.Module):
    def __init__(self, n_embd, num_shards):
        super().__init__()
        self.num_shards = num_shards
        self.shards1 = nn.ModuleList()
        self.shards2 = nn.ModuleList()
        self.devices = [torch.device(f'cuda:{i}') for i in range(num_shards)]
        assert (4*n_embd) % num_shards == 0

        for i in range(num_shards):
            self.shards1.append(nn.Linear(n_embd, 4*n_embd//num_shards, bias=False, device=self.devices[i]))
            self.shards2.append(nn.Linear(4*n_embd//num_shards, n_embd, bias=False, device=self.devices[i]))

    def forward(self, x):
        x = f.apply(x, self.devices)
        x = parallel_apply(self.shards1, x, None, self.devices) 
        x = parallel_apply(self.shards2, x, None, self.devices)
        x = g.apply(x, self.devices)
        return x


if __name__ == '__main__':
    n_embd = 1024
    num_shards = 4
    model = ShardedMLP(n_embd, num_shards)
    x = torch.rand(4, n_embd).to('cuda:0')
    sharded_out = model(x)

    model_seq = nn.Sequential(
                    nn.Linear(n_embd, 4*n_embd, bias=False),
                    nn.Linear(4*n_embd, n_embd, bias=False)
                )

    step = 4*n_embd//num_shards
    for i in range(num_shards):
        model_seq[0].weight.data[i*step:(i+1)*step, :] = model.shards1[i].weight.data
        model_seq[1].weight.data[:, i*step:(i+1)*step] = model.shards2[i].weight.data
    model_seq.to('cuda:0')
    seq_out = model_seq(x)

    assert torch.allclose(sharded_out, seq_out, atol=1e-5)

