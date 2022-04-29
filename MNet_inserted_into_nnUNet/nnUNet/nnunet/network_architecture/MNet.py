from torch import nn
import torch
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional as F
from nnunet.network_architecture.basic_module import CB3d, BasicNet



def FMU(x1, x2, mode='sub'):
    """
    feature merging unit
    Args:
        x1:
        x2:
        mode: types of fusion
    Returns:
    """
    if mode == 'sum':
        return torch.add(x1, x2)
    elif mode == 'sub':
        return torch.abs(x1 - x2)
    elif mode == 'cat':
        return torch.cat((x1, x2), dim=1)
    else:
        raise Exception('Unexpected mode')


class Down(BasicNet):
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='sub', downsample=True, min_z=8):
        """
        basic module at downsampling stage
        Args:
            in_channels:
            out_channels:
            mode: represent the streams coming in and out. e.g., ('2d', 'both'): one input stream (2d) and two output streams (2d and 3d)
            FMU: determine the type of feature fusion if there are two input streams
            downsample: determine whether to downsample input features (only the first module of MNet do not downsample)
            min_z: if the size of z-axis < min_z, maxpooling won't be applied along z-axis
        """
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.downsample = downsample
        self.FMU = FMU
        self.min_z = min_z
        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)

        if self.mode_out == '2d' or self.mode_out == 'both':
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out == '3d' or self.mode_out == 'both':
            self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

    def forward(self, x):
        if self.downsample:
            if self.mode_in == 'both':
                x2d, x3d = x
                p2d = F.max_pool3d(x2d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                if x3d.shape[2] >= self.min_z:
                    p3d = F.max_pool3d(x3d, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    p3d = F.max_pool3d(x3d, kernel_size=(1, 2, 2), stride=(1, 2, 2))

                x = FMU(p2d, p3d, mode=self.FMU)

            elif self.mode_in == '2d':
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

            elif self.mode_in == '3d':
                if x.shape[2] >= self.min_z:
                    x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        if self.mode_out == '2d':
            return self.CB2d(x)
        elif self.mode_out == '3d':
            return self.CB3d(x)
        elif self.mode_out == 'both':
            return self.CB2d(x), self.CB3d(x)


class Up(BasicNet):
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='sub'):
        """
        basic module at upsampling stage
        Args:
            in_channels:
            out_channels:
            mode: represent the streams coming in and out. e.g., ('2d', 'both'): one input stream (2d) and two output streams (2d and 3d)
            FMU: determine the type of feature fusion if there are two input streams
        """
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.FMU = FMU
        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)

        if self.mode_out == '2d' or self.mode_out == 'both':
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out == '3d' or self.mode_out == 'both':
            self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

    def forward(self, x):
        x2d, xskip2d, x3d, xskip3d = x

        tarSize = xskip2d.shape[2:]
        up2d = F.interpolate(x2d, size=tarSize, mode='trilinear', align_corners=False)
        up3d = F.interpolate(x3d, size=tarSize, mode='trilinear', align_corners=False)

        cat = torch.cat([FMU(xskip2d, xskip3d, self.FMU), FMU(up2d, up3d, self.FMU)], dim=1)

        if self.mode_out == '2d':
            return self.CB2d(cat)
        elif self.mode_out == '3d':
            return self.CB3d(cat)
        elif self.mode_out == 'both':
            return self.CB2d(cat), self.CB3d(cat)


class MNet(SegmentationNetwork):
    # ----------------------------------------parameters of NNUNet, not mine
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440
    # ----------------------------------------

    def __init__(self, in_channels, num_classes, kn=(32, 48, 64, 80, 96), ds=True, FMU='sub'):
        """

        Args:
            in_channels: channels of input
            num_classes: output classes
            kn: the number of kernels
            ds: deep supervision
            FMU: type of feature merging unit
        """
        super().__init__()
        # ----------------------------------------parameters of NNUNet
        self.conv_op = nn.Conv3d
        self._deep_supervision = self.do_ds = ds
        self.num_classes = num_classes
        # ----------------------------------------


        channel_factor = {'sum': 1, 'sub': 1, 'cat': 2}
        fct = channel_factor[FMU]



        self.down11 = Down(in_channels, kn[0], ('/', 'both'), downsample=False)
        self.down12 = Down(kn[0], kn[1], ('2d', 'both'))
        self.down13 = Down(kn[1], kn[2], ('2d', 'both'))
        self.down14 = Down(kn[2], kn[3], ('2d', 'both'))
        self.bottleneck1 = Down(kn[3], kn[4], ('2d', '2d'))
        self.up11 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', '2d'), FMU)
        self.up12 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', '2d'), FMU)
        self.up13 = Up(fct * (kn[1] + kn[2]), kn[1], ('both', '2d'), FMU)
        self.up14 = Up(fct * (kn[0] + kn[1]), kn[0], ('both', 'both'), FMU)

        self.down21 = Down(kn[0], kn[1], ('3d', 'both'))
        self.down22 = Down(fct * kn[1], kn[2], ('both', 'both'), FMU)
        self.down23 = Down(fct * kn[2], kn[3], ('both', 'both'), FMU)
        self.bottleneck2 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up21 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', 'both'), FMU)
        self.up22 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', 'both'), FMU)
        self.up23 = Up(fct * (kn[1] + kn[2]), kn[1], ('both', '3d'), FMU)

        self.down31 = Down(kn[1], kn[2], ('3d', 'both'))
        self.down32 = Down(fct * kn[2], kn[3], ('both', 'both'), FMU)
        self.bottleneck3 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up31 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', 'both'), FMU)
        self.up32 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', '3d'), FMU)

        self.down41 = Down(kn[2], kn[3], ('3d', 'both'), FMU)
        self.bottleneck4 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up41 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', '3d'), FMU)

        self.bottleneck5 = Down(kn[3], kn[4], ('3d', '3d'))

        self.outputs = nn.ModuleList(
            [nn.Conv3d(c, num_classes, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
             for c in [kn[0], kn[1], kn[1], kn[2], kn[2], kn[3], kn[3]]]
        )

    def forward(self, x):
        down11 = self.down11(x)
        down12 = self.down12(down11[0])
        down13 = self.down13(down12[0])
        down14 = self.down14(down13[0])
        bottleNeck1 = self.bottleneck1(down14[0])

        down21 = self.down21(down11[1])
        down22 = self.down22([down21[0], down12[1]])
        down23 = self.down23([down22[0], down13[1]])
        bottleNeck2 = self.bottleneck2([down23[0], down14[1]])

        down31 = self.down31(down21[1])
        down32 = self.down32([down31[0], down22[1]])
        bottleNeck3 = self.bottleneck3([down32[0], down23[1]])

        down41 = self.down41(down31[1])
        bottleNeck4 = self.bottleneck4([down41[0], down32[1]])

        bottleNeck5 = self.bottleneck5(down41[1])

        up41 = self.up41([bottleNeck4[0], down41[0], bottleNeck5, down41[1]])

        up31 = self.up31([bottleNeck3[0], down32[0], bottleNeck4[1], down32[1]])
        up32 = self.up32([up31[0], down31[0], up41, down31[1]])

        up21 = self.up21([bottleNeck2[0], down23[0], bottleNeck3[1], down23[1]])
        up22 = self.up22([up21[0], down22[0], up31[1], down22[1]])
        up23 = self.up23([up22[0], down21[0], up32, down21[1]])

        up11 = self.up11([bottleNeck1, down14[0], bottleNeck2[1], down14[1]])
        up12 = self.up12([up11, down13[0], up21[1], down13[1]])
        up13 = self.up13([up12, down12[0], up22[1], down12[1]])
        up14 = self.up14([up13, down11[0], up23, down11[1]])


        if self._deep_supervision and self.do_ds:
            features = [up14[0] + up14[1], up23, up13, up32, up12, up41, up11]
            return tuple([self.outputs[i](features[i]) for i in range(7)])
        else:
            return self.outputs[0](up14[0] + up14[1])


if __name__ == '__main__':
    MNet = MNet(1, 3, kn=(2, 2, 2, 2, 2), ds=True, FMU='sub')
    input = torch.randn((1, 1, 19, 255, 256))
    output = MNet(input)

    print([e.shape for e in output])

