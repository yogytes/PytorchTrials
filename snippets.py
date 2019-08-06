from __future__ import print_function
import torch


if __name__ == "__main__":
#Construct a 5x3 matrix, uninitialized:
    x = torch.empty(5, 3)
   # print(x)

    x = torch.rand(5,3)
    #print(x)

    x = torch.zeros(5,3, dtype=torch.long)
    #print(x)

    t = torch.tensor([2,3])
    #print(t)

    x=torch.randn(4,4)
    y= x.view(16)
    z= x.view(-1, 8) # size -1 is infered from other dimensions

    #print(x, y, z)

    # torch numpy conversion
    n_a = z.numpy()
    #print(n_a)
    t_t = torch.from_numpy(n_a)
    #print(t_t)

    ''' Autograd : automatic derivative '''
    x = torch.ones(2, 2, requires_grad=True)
    print(x)

    y = x+2
    print(y.grad_fn)

    z = y * y * 3
    out = z.mean()

    print(z, out)

    out.backward()
    print(x.grad)

    x = torch.randn(3, requires_grad=True)

    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2

    print(y)
    #vector Jacobian product
    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)

    print(x.grad)

  