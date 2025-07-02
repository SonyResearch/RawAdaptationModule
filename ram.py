from mmengine import MessageHub
from torch import nn
import torch.nn.functional as F
import torch


def conv_block(in_channels, out_channels, norm='BN'):
    if norm == 'IN':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        )
    elif norm == 'BN':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    elif norm is None:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.LeakyReLU()
        )


class RPEncoder(nn.Module):
    """
    Raw Parameter Encoder, receives an input image and generate a feature vector.

    Args:
        img_size (int): down-sampled input size.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
    """
    def __init__(self, img_size=256, in_channels=3, out_channels=128, ) -> None:
        super().__init__()
        self.img_size = img_size
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, 16, 7, padding="same"),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, out_channels, 3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear")
        return self.seq(x)


class RPDecoder(nn.Module):
    """
    Raw Parameter Decoder, receives a feature vector and generate ISP parameters.

    Args:
        out_channels (int): number of output channels.
        in_channels (int): number of input channels.
    """
    def __init__(self, out_channels, in_channels=128) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


def define_feature_fusion(in_channels, out_channels=3, ffm_params=None):
    # Selects feature fusion module architecture, default: BN_HG.

    if ffm_params is None:
        ffm_type = 'BN_HG'
        mid_channels = 64
    else:
        ffm_type = ffm_params.pop('ffm_type', 'BN_HG')
        mid_channels = ffm_params.pop('mid_channels', 64)

    if ffm_type == 'IN':
        ffm = nn.Sequential(
            conv_block(in_channels, 16, 'IN'),
            conv_block(16, mid_channels, 'IN'),
            nn.Conv2d(mid_channels, out_channels, 1, padding="same"),
        )
    elif ffm_type == 'HG':
        ffm = nn.Sequential(
            conv_block(in_channels, 16, None),
            conv_block(16, mid_channels, None),
            conv_block(mid_channels, 16, None),
            nn.Conv2d(16, out_channels, 1, padding="same"),
        )
    elif ffm_type == 'IN_HG':
        ffm = nn.Sequential(
            conv_block(in_channels, 16, 'IN'),
            conv_block(16, mid_channels, 'IN'),
            conv_block(mid_channels, 16, 'IN'),
            nn.Conv2d(16, out_channels, 1, padding="same"),
        )
    elif ffm_type == 'BN_HG':
        ffm = nn.Sequential(
            conv_block(in_channels, 16, 'BN'),
            conv_block(16, mid_channels, 'BN'),
            conv_block(mid_channels, 16, 'BN'),
            nn.Conv2d(16, out_channels, 1, padding="same"),
        )
    elif ffm_type == 'BN':
        ffm = nn.Sequential(
            conv_block(in_channels, 16, 'BN'),
            conv_block(16, mid_channels, 'BN'),
            nn.Conv2d(mid_channels, out_channels, 1, padding="same"),
        )
    return ffm


class RawAdaptationModule(nn.Module):
    """
    Raw Adaptation Module (RAM).
    Learn and apply ISP parameters in parallel adaptively for every input image.

    Args:
        in_channels (int): number of input channels.
        img_size (int): down-sampled input size.
        out_channels (int): output channels of RPE
        functions (list): a list of ISP functions to apply
        ffm_params (dict): parameters for the feature fusion module
        clamp_values (bool): needed when the input contains negative values
    """
    def __init__(self,
                 in_channels: int = 3,
                 img_size: int = 256,
                 out_channels: int = 128,
                 functions: list = [],
                 ffm_params: dict = None,
                 clamp_values: bool = False
                 ):
        super().__init__()
        self.message_hub = MessageHub.get_current_instance()
        self.in_channels = in_channels

        self.functions = functions
        self.encoder = RPEncoder(img_size=img_size, in_channels=in_channels, out_channels=out_channels)

        for function in self.functions:
            self.define_function(function, rpe_out_channels=out_channels, input_channels=in_channels)

        self.norm_layer = nn.BatchNorm2d(in_channels, affine=True)

        ffm_in_channels = in_channels * max(len(functions), 1)
        self.ffm = define_feature_fusion(in_channels=ffm_in_channels,
                                         ffm_params=ffm_params)

        self.clamp_values = clamp_values

    def forward(self, x, training=True):
        assert len(self.functions) > 0, "functions list is empty"

        outputs = []

        if self.clamp_values:
            x = torch.clamp(x, min=0)
        input = x
        x = self.encoder(x)
        for function in self.functions:
            out = self.apply_function(input, function, x, training)
            outputs.append(out)

        x = torch.cat(outputs, dim=1)

        x = self.ffm(x)
        x = self.norm_layer(x)

        return x

    def learn_gamma(self, inputs, training):
        gamma = self.gamma(inputs).view(-1, 1, 1, 1)
        if gamma is not None and training:
            self.message_hub.update_scalar(f'train/gamma_val',
                                           float(gamma.mean()))
        return gamma

    def define_function(self, function, rpe_out_channels, input_channels):
        # Define decoder architecture for each function

        if function == 'gamma':
            self.gamma = nn.Sequential(
                RPDecoder(in_channels=rpe_out_channels, out_channels=1),
                nn.Sigmoid()
            )
        elif function == 'ccm':
            self.conv_cc = RPDecoder(in_channels=rpe_out_channels, out_channels=(input_channels ** 2))
        elif function == 'wb':
            self.conv_wb = RPDecoder(in_channels=rpe_out_channels, out_channels=input_channels)
        elif function == 'brightness':
            self.conv_bright = nn.Sequential(
                RPDecoder(in_channels=rpe_out_channels, out_channels=1),
                nn.Sigmoid()
            )

    def apply_function(self, input, function, x, training):
        # Apply ISP function with the predicted parameters on the original input image.

        bs = x.size(0)

        if function == 'gamma':
            gamma = self.learn_gamma(x, training)
            out = input ** gamma
        elif function == 'ccm':
            params_cc = self.conv_cc(x)
            params_cc = params_cc.reshape(bs, self.in_channels, self.in_channels)
            out = torch.einsum('bcij,bnc->bnij', input, params_cc)
        elif function == 'wb':
            params_wb = self.conv_wb(x)
            params_wb = params_wb.reshape(bs, self.in_channels, 1, 1)
            out = input * params_wb
        elif function == 'brightness':
            params_bright = self.conv_bright(x).view(-1, 1, 1, 1)
            out = input + params_bright
        else:
            out = x
        return out


if __name__ == "__main__":
    from torchinfo import summary
    ram = RawAdaptationModule(functions=['wb', 'ccm', 'gamma', 'brightness'],
                              ffm_params=dict(ffm_type='BN_HG'),
                              clamp_values=True)

    summary(ram, (1, 3, 400, 600))

