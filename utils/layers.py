from utils.utils import *


def make_divisible(v, divisor):
    # Function ensures all layers have a channel number that is divisible by 8
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    return math.ceil(v / divisor) * divisor


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def execute(self, x):
        return x.view(x.size(0), -1)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def execute(self, x):
        return jt.contrib.concat(x, self.d)


class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def execute(self, x, outputs):
        return jt.contrib.concat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = jt.zeros((self.n,)) # layer weights

    def execute(self, x, outputs):
        # Weights
        if self.weight:
            w = jt.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class MixConv2d(nn.Module):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        super(MixConv2d, self).__init__()

        groups = len(k)
        if method == 'equal_ch':  # equal channels per group
            i = jt.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv(in_channels=in_ch,
                                          out_channels=ch[g],
                                          kernel_size=k[g],
                                          stride=stride,
                                          padding=k[g] // 2,  # 'same' pad
                                          dilation=dilation,
                                          bias=bias) for g in range(groups)])

    def execute(self, x):
        return jt.contrib.concat([m(x) for m in self.m], 1)


# Activation functions below -------------------------------------------------------------------------------------------
class SwishImplementation(jt.Function):
    def execute(x):
        self.x = x
        return x * jt.sigmoid(x)

    def grad(grad_output):
        x = self.x
        sx = jt.sigmoid(x)  # sigmoid(ctx)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MishImplementation(jt.Function):

    def execute(x):
        self.x  = x
        return x.mul(jt.tanh(nn.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    def grad(ctx, grad_output):
        x = self.x
        sx = jt.sigmoid(x)
        fx = nn.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientSwish(nn.Module):
    def execute(self, x):
        return SwishImplementation()(x)


class MemoryEfficientMish(nn.Module):
    def execute(self, x):
        return MishImplementation()(x)


class Swish(nn.Module):
    def execute(self, x):
        return x * jt.sigmoid(x)


class HardSwish(nn.Module):  # https://arxiv.org/pdf/1905.02244.pdf
    def execute(self, x):
        return x * nn.hardtanh(x + 3, 0., 6., True) / 6.


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def execute(self, x):
        return x * nn.softplus(x).tanh()
