import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Lambda

torch.set_default_dtype(torch.float32)

def averagePooling(x):
    return torch.mean(x, -1, False)

def maxPooling(x):
    return torch.max(x, -1, False)


class DilatedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(DilatedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation, groups, bias)

    def forward(self, inputs):
        outputs = super(DilatedConv2d, self).forward(inputs)
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, kernel_size, padding, dropout_rate, dilation=1, repeat_times=2):
        super(ResidualBlock, self).__init__()
        self.repeat_times = repeat_times
        self.filter_conv1 = DilatedConv2d(in_channels=res_channels, kernel_size=[1, kernel_size], padding=0, out_channels=res_channels, dilation=dilation)
        self.filter_conv2 = DilatedConv2d(in_channels=res_channels, kernel_size=[1, kernel_size], padding=0, out_channels=res_channels, dilation=dilation)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=res_channels)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=res_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.skip_conv = nn.Conv2d(in_channels=res_channels, kernel_size=1, padding=0, out_channels=skip_channels)

    def forward(self,inputs):
        skip_out = self.skip_conv(inputs)
        output = inputs

        # do padding here since 'same' padding is not supported in pth2onnx conversion
        output = F.pad(output, (1, 2, 0, 0))  # left, right, top, bottom
        output = self.filter_conv1(output)
        output = self.batch_norm1(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = F.pad(output, (1, 2, 0, 0))  # left, right, top, bottom
        output = self.filter_conv2(output)
        output = self.batch_norm2(output)
        output = self.relu(output)
        output = self.dropout(output)
        res_out = output + skip_out
        return res_out, output


class TCN(nn.Module):
    """Creates a TCN layer.
        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).
        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        Returns:
            A TCN layer.
        """

    def __init__(self,
                 feature_size,
                 num_outputs=1,
                 nb_filters=128,
                 kernel_size=4,
                 nb_stacks=1,
                 dilation_depth=4,
                 padding=1,
                 use_skip_connections=True,
                 dropout_rate=0.1,
                 return_sequences=False):
        super(TCN, self).__init__()
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = [2**i for i in range(dilation_depth)] * nb_stacks
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.pre = nn.Conv2d(feature_size, self.nb_filters, 1, padding=0)
        # dilation conv is currently disabled because it's not supported in tensorrt
        self.main = nn.ModuleList(
            [ResidualBlock(res_channels=self.nb_filters,
                          skip_channels=self.nb_filters,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                        #   dilation=dilation
                          dropout_rate=self.dropout_rate)
            for dilation in self.dilations])
        self.post = nn.Sequential(nn.Linear(nb_filters, num_outputs))

        if not isinstance(nb_filters, int):
            raise Exception()

    def forward(self, inputs):
        # permute input to be in order of NCHW for conv computation with torch
        # the expected order here: batch_size * feature_size * 1(height of each feature) * timesteps
        # inputs = inputs.permute(0, 3, 1, 2)
        outputs = self.pre(inputs)
        skip_connections = []

        for layer in self.main:
            outputs, skip = layer(outputs)
            skip_connections.append(skip)

        if self.use_skip_connections:
            outputs = sum([s for s in skip_connections])
        if not self.return_sequences:
            outputs = Lambda(averagePooling)(outputs)
            outputs = outputs.view([-1, outputs.shape[-2] * outputs.shape[-1]])
        outputs = self.post(outputs)

        return outputs
    
    