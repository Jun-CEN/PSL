import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.modules.utils import _triple

from models.base.base_blocks import BaseBranch
from models.base.base_blocks import BRANCH_REGISTRY, STEM_REGISTRY

class route_func_multiple_dense(nn.Module):

    def __init__(self, c_in, num_experts, num_heads):
        super(route_func_multiple_dense, self).__init__()
        self.c_in = c_in
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts*c_in*num_heads)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x.reshape(-1, self.num_heads, self.num_experts, self.c_in, 1, 1, 1, 1))
        return x

class route_func_dense_mlp_asym(nn.Module):

    def __init__(self, c_in, c_out, ratio, num_experts, bn_eps=1e-5):
        super(route_func_dense_mlp_asym, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_experts = num_experts
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc = nn.Conv3d(
            in_channels     = c_in,
            out_channels    = c_in//ratio*num_experts,
            kernel_size     = 1,
            stride          = 1, 
            padding         = 0,
            bias            = False
        )
        self.gn = nn.GroupNorm(
            num_groups=num_experts,
            num_channels=c_in//ratio*num_experts,
            eps=bn_eps,
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(
            in_channels     = c_in//ratio*num_experts,
            out_channels    = num_experts*c_out,
            kernel_size     = 1,
            stride          = 1, 
            padding         = 0,
            bias            = False,
            groups          = num_experts,
        )
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.gn(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x.reshape(-1, self.num_experts, self.c_out, 1, 1, 1, 1))
        return x

class route_func_dense_conv_asym(nn.Module):

    def __init__(self, c_in, c_out, num_experts):
        super(route_func_dense_conv_asym, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_experts = num_experts
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc = nn.Conv3d(
            in_channels     = c_in,
            out_channels    = num_experts*c_out,
            kernel_size     = 1,
            stride          = 1, 
            padding         = 0,
            bias            = False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.sigmoid(x.reshape(-1, self.num_experts, self.c_out, 1, 1, 1, 1))
        return x

class route_func_dense_asym(nn.Module):

    def __init__(self, c_in, c_out, num_experts):
        super(route_func_dense_asym, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_experts = num_experts
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts*c_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x.reshape(-1, self.num_experts, self.c_out, 1, 1, 1, 1))
        return x

class route_func_dense(nn.Module):

    def __init__(self, c_in, num_experts):
        super(route_func_dense, self).__init__()
        self.c_in = c_in
        self.num_experts = num_experts
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts*c_in)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x.reshape(-1, self.num_experts, self.c_in, 1, 1, 1, 1))
        return x

class route_func_softmax(nn.Module):
    def __init__(self, c_in, num_experts):
        super(route_func_softmax, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x/30)
        return x

class route_func(nn.Module):
    r"""Extended to 3D from CondConv.
    CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        c_in (int): Number of channels in the input image
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, c_in, num_experts):
        super(route_func, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class CondConv3d(nn.Module):
    r"""Extended from CondConv2d.
    CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv3d, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        k, c_out, c_in, kt, kh, kw = self.weight.size()
        x = x.view(1, -1, t, h, w)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kt, kh, kw)
        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv3d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv3d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-3), output.size(-2), output.size(-1))
        return output

class CondConv3dDense(nn.Module):
    r"""Extended from CondConv2d.
    CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv3dDense, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        k, c_out, c_in, kt, kh, kw = self.weight.size()
        x = x.view(1, -1, t, h, w)
        weight = self.weight.unsqueeze(0)
        combined_weight = (weight*routing_weight).sum(1).view(-1, c_in, kt, kh, kw)
        if self.bias is not None:
            combined_bias = (self.bias.unsqueeze(0) * routing_weight.squeeze()).sum(1).view(-1)
            output = F.conv3d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv3d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-3), output.size(-2), output.size(-1))
        return output

class CondConv3dInterDecoupled(nn.Module):
    r"""Extended from CondConv2d.
    CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv3dInterDecoupled, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, 1, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, 1))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        k, c_out, c_in, kt, kh, kw = self.weight.size()
        x = x.view(1, -1, t, h, w)
        weight = self.weight.unsqueeze(0)
        combined_weight = (weight*routing_weight).sum(1).view(-1, c_in, kt, kh, kw)
        if self.bias is not None:
            combined_bias = (self.bias.unsqueeze(0) * routing_weight.squeeze()).sum(1).view(-1)
            output = F.conv3d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv3d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, self.out_channels, output.size(-3), output.size(-2), output.size(-1))
        return output

class CondConv3dInterDecoupledAndGeneration(nn.Module):
    r"""Extended from CondConv2d.
    CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv3dInterDecoupledAndGeneration, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, 1, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, 1))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        k, c_out, c_in, kt, kh, kw = self.weight.size()
        x = x.view(1, -1, t, h, w)
        weight = self.weight.unsqueeze(0).unsqueeze(1)
        combined_weight = (weight*routing_weight).sum((1,2)).view(-1, c_in, kt, kh, kw)
        if self.bias is not None:
            combined_bias = (self.bias.unsqueeze(0).unsqueeze(1) * routing_weight.squeeze()).sum((1,2)).view(-1)
            output = F.conv3d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv3d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, self.out_channels, output.size(-3), output.size(-2), output.size(-1))
        return output

class CondConv3dMSRAFill(nn.Module):
    r"""Extended from CondConv2d.
    CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv3dMSRAFill, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        k, c_out, c_in, kt, kh, kw = self.weight.size()
        x = x.view(1, -1, t, h, w)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kt, kh, kw)
        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv3d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv3d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-3), output.size(-2), output.size(-1))
        return output

@BRANCH_REGISTRY.register()
class CondConvBaseline(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(CondConvBaseline, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.num_experts = cfg.VIDEO.BACKBONE.BRANCH.NUM_EXPERTS

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = CondConv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b_rf = route_func(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = CondConv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b2_rf = route_func(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x, self.b2_rf(x))
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class CondConvAll(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(CondConvAll, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.num_experts = cfg.VIDEO.BACKBONE.BRANCH.NUM_EXPERTS

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = CondConv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            num_experts     = self.num_experts,
            bias            = False
        )
        self.a_rf = route_func(
            c_in = self.dim_in,
            num_experts=self.num_experts
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = CondConv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b_rf = route_func(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = CondConv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b2_rf = route_func(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = CondConv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            num_experts     = self.num_experts,
            bias            = False
        )
        self.c_rf = route_func(
            c_in = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x, self.a_rf(x))
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x, self.b2_rf(x))
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x, self.c_rf(x))
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class CondConvDense(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(CondConvDense, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.num_experts = cfg.VIDEO.BACKBONE.BRANCH.NUM_EXPERTS

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = CondConv3dDense(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b_rf = route_func_dense(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = CondConv3dDense(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b2_rf = route_func_dense(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x, self.b2_rf(x))
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class CondConvInterDecoupled(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(CondConvInterDecoupled, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.num_experts = cfg.VIDEO.BACKBONE.BRANCH.NUM_EXPERTS

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = CondConv3dInterDecoupled(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b_rf = route_func_dense(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = CondConv3dInterDecoupled(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b2_rf = route_func_dense(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x, self.b2_rf(x))
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class CondConvInterDecoupledV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(CondConvInterDecoupledV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.num_experts = cfg.VIDEO.BACKBONE.BRANCH.NUM_EXPERTS

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = CondConv3dInterDecoupled(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b_rf = route_func_dense_conv_asym(
            c_in=self.num_filters//self.expansion_ratio,
            c_out=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = CondConv3dInterDecoupled(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b2_rf = route_func_dense_conv_asym(
            c_in=self.num_filters//self.expansion_ratio,
            c_out=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x, self.b2_rf(x))
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class CondConvInterDecoupledAll(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(CondConvInterDecoupledAll, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.num_experts = cfg.VIDEO.BACKBONE.BRANCH.NUM_EXPERTS

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = CondConv3dInterDecoupled(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            num_experts     = self.num_experts,
            bias            = False
        )
        self.a_rf = route_func_dense_asym(
            c_in=self.dim_in,
            c_out=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = CondConv3dInterDecoupled(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b_rf = route_func_dense(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = CondConv3dInterDecoupled(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b2_rf = route_func_dense(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = CondConv3dInterDecoupled(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            num_experts     = self.num_experts,
            bias            = False
        )
        self.c_rf = route_func_dense_asym(
            c_in=self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            c_out=self.num_filters,
            num_experts=self.num_experts,
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x, self.a_rf(x))
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x, self.b2_rf(x))
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x, self.c_rf(x))
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class CondConvInterDecoupledAndGeneration(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(CondConvInterDecoupledAndGeneration, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.num_experts = cfg.VIDEO.BACKBONE.BRANCH.NUM_EXPERTS

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = CondConv3dInterDecoupledAndGeneration(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b_rf = route_func_multiple_dense(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
            num_heads=2,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = CondConv3dInterDecoupledAndGeneration(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b2_rf = route_func_multiple_dense(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
            num_heads=2,
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x, self.b2_rf(x))
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x


@BRANCH_REGISTRY.register()
class CondConvBaselineMSRAFill(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(CondConvBaselineMSRAFill, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.num_experts = cfg.VIDEO.BACKBONE.BRANCH.NUM_EXPERTS

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = CondConv3dMSRAFill(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b_rf = route_func(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = CondConv3dMSRAFill(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b2_rf = route_func(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x, self.b2_rf(x))
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class CondConvBaselineMSRAFillSpatial(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(CondConvBaselineMSRAFillSpatial, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.num_experts = cfg.VIDEO.BACKBONE.BRANCH.NUM_EXPERTS

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = CondConv3dMSRAFill(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b_rf = route_func(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class CondConvSoftmax(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(CondConvSoftmax, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.num_experts = cfg.VIDEO.BACKBONE.BRANCH.NUM_EXPERTS

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = CondConv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b_rf = route_func_softmax(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = CondConv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            num_experts     = self.num_experts,
            bias            = False
        )
        self.b2_rf = route_func_softmax(
            c_in=self.num_filters//self.expansion_ratio,
            num_experts=self.num_experts,
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x, self.b2_rf(x))
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x



if __name__ == "__main__":
    B, C, T, H, W = (2, 3, 8, 56, 56)
    x = torch.rand((B,C,T,H,W))
    conv3d = CondConv3d(
        in_channels=3, 
        out_channels=4, 
        kernel_size=[1,3,3], 
        stride=1, 
        padding=[0,1,1], 
        num_experts=2, bias=False
    )
    rf = route_func(
        c_in=3, 
        num_experts=2
    )
    
    xout = conv3d(x, rf(x))
    print("finish")
