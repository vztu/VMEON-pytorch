import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable


class Gdn3DFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x, gamma, beta):
        # save variables for backprop
        ctx.save_for_backward(x, gamma, beta)
        n, c, d, h, w = list(x.size())
        # input is formatted as NCDHW, here we need it to be NDHWC
        tx = x.permute(0, 2, 3, 4, 1).contiguous()
        tx = tx.view(-1, c)
        tx2 = tx * tx
        # rbeta = beta.repeat(n * h * w, 1)
        denominator = tx2.mm(gamma) + beta
        ty = tx / torch.sqrt(denominator)
        y = ty.view(n, d, h, w, c)
        y = y.permute(0, 4, 1, 2, 3).contiguous()
        return y

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        x, gamma, beta = ctx.saved_variables
        # input is formatted as NCDHW, here we need it to be NDHWC
        n, c, d, h, w = list(grad_output.size())
        tx = x.permute(0, 2, 3, 4, 1).contiguous()
        tx = tx.view(-1, c)
        tx2 = tx * tx
        # rbeta = beta.repeat(n * h * w, 1)
        denominator = tx2.mm(gamma) + beta

        tdzdy = grad_output.permute(0, 2, 3, 4, 1).contiguous()
        tdzdy = tdzdy.view(-1, c)
        gy = (tdzdy * torch.pow(denominator, -0.5) - (tdzdy * tx *
                                                      torch.pow(denominator, -1.5)).mm(gamma.t()) * tx)
        gy = gy.view(n, d, h, w, c)
        grad_input = gy.permute(0, 4, 1, 2, 3).contiguous()
        tmp = -0.5 * torch.pow(denominator, -1.5) * tx * tdzdy
        grad_beta = torch.sum(tmp, 0)
        grad_gamma = tx2.t().mm(tmp)

        return grad_input, grad_gamma, grad_beta


class Gdn3D(nn.Module):
    def __init__(self, input_channel):
        super(Gdn3D, self).__init__()
        self.input_channel = input_channel
        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.
        self.gamma = nn.Parameter(torch.Tensor(input_channel, input_channel))
        self.beta = nn.Parameter(torch.Tensor(input_channel))

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return Gdn3DFunction.apply(input, self.gamma, self.beta)

    def __str__(self):
        return self.__class__.__name__ + '(gamma_size=(%d, %d), beta_size=(%d))' %\
               (self.gamma.size()[0], self.gamma.size()[1], self.beta.size()[0])

    __repr__ = __str__


if __name__ == '__main__':
    # GDN unit test
    test_input = (Variable(torch.randn(10, 2, 8, 5, 5).double(), requires_grad=True),
                  Variable(torch.randn(2, 2).double()+10, requires_grad=True),
                  Variable(torch.randn(2).double()+100, requires_grad=True))
    res = torch.autograd.gradcheck(Gdn3DFunction.apply, test_input, raise_exception=True)
    print(res) # res should be True if the gradients are correct.
