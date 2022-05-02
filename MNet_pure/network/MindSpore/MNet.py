import mindspore.nn as nn
import mindspore.ops as ops


class CNA3d(nn.Cell): # conv + norm + activation
    def __init__(self, in_channels, out_channels, kSize, stride, padding=(1,1,1), bias=True, norm=True, activation=True):
        super().__init__()
        self.norm = norm
        self.activation = activation

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kSize, stride=stride, pad_mode='pad',
                              padding=padding, has_bias=bias)

        if norm: # InstanceNorm3d is not available in mindspore
            self.norm = nn.BatchNorm3d(out_channels)

        if activation:
            self.activation = nn.LeakyReLU()


    def construct(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)

        if self.activation:
            x = self.activation(x)
        return x

class CB3d(nn.Cell): # conv block 3d
    def __init__(self, in_channels, out_channels, kSize=(3,3), stride=(1,1), padding=(1,1,1), bias=True,
                 norm:tuple=(None,None), activation:tuple=(None,None)):
        super().__init__()

        self.conv1 = CNA3d(in_channels, out_channels, kSize=kSize[0], stride=stride[0],
                             padding=padding, bias=bias, norm=norm[0], activation=activation[0])

        self.conv2 = CNA3d(out_channels, out_channels,kSize=kSize[1], stride=stride[1],
                             padding=padding, bias=bias, norm=norm[1], activation=activation[1])

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


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
        return ops.Add()(x1, x2)
    elif mode == 'sub':
        return ops.Abs()(x1 - x2)
    elif mode == 'cat':
        return ops.Concat(axis=1)((x1, x2))
    else:
        raise Exception('Unexpected mode')


class Down(nn.Cell):
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='sub', downsample=True, min_z=4):
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

        if self.mode_out == '2d' or self.mode_out == 'both':
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 0, 1, 1, 1, 1),
                             norm=(True,True), activation=(True,True))

        if self.mode_out == '3d' or self.mode_out == 'both':
            self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1, 1, 1),
                             norm=(True,True), activation=(True,True))

    def construct(self, x):
        if self.downsample:
            if self.mode_in == 'both':
                x2d, x3d = x


                p2d = ops.MaxPool3D(kernel_size=(1,2,2), strides=(1,2,2), pad_mode="valid")(x2d)
                # p2d = F.max_pool3d(x2d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                if x3d.shape[2] >= self.min_z:
                    p3d = ops.MaxPool3D(kernel_size=(2, 2, 2), strides=(2, 2, 2))(x3d)
                    # p3d = F.max_pool3d(x3d, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    p3d = ops.MaxPool3D(kernel_size=(1, 2, 2), strides=(1, 2, 2))(x3d)
                    # p3d = F.max_pool3d(x3d, kernel_size=(1, 2, 2), stride=(1, 2, 2))

                x = FMU(p2d, p3d, mode=self.FMU)

            elif self.mode_in == '2d':
                x = ops.MaxPool3D(kernel_size=(1, 2, 2), strides=(1, 2, 2))(x)
                # x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

            elif self.mode_in == '3d':
                if x.shape[2] >= self.min_z:
                    x = ops.MaxPool3D(kernel_size=(2, 2, 2), strides=(2, 2, 2))(x)
                    # x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    x = ops.MaxPool3D(kernel_size=(1, 2, 2), strides=(1, 2, 2))(x)
                    # x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        if self.mode_out == '2d':
            return self.CB2d(x)
        elif self.mode_out == '3d':
            return self.CB3d(x)
        else:
            return self.CB2d(x), self.CB3d(x)


class Up(nn.Cell):
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

        if self.FMU == 'cat':
            inC_deconv = in_channels//2 - out_channels
        else:
            inC_deconv = in_channels - out_channels

        # trilinear interpolation is not available in mindspore, thus we use deconv instead
        self.deconv2d = nn.Conv3dTranspose(inC_deconv, inC_deconv, kernel_size=(1,3,3), stride=(1,2,2),
                                           pad_mode='same', padding=0, output_padding=0)

        self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                         kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 0, 1, 1, 1, 1),
                         norm=(True,True), activation=(True,True))


        # trilinear interpolation is not available in mindspore, thus we use deconv instead
        self.deconv3d = nn.Conv3dTranspose(inC_deconv, inC_deconv, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                           pad_mode='same', padding=0, output_padding=0)

        self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                         kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1, 1, 1),
                         norm=(True,True), activation=(True,True))



    def construct(self, x):
        x2d, xskip2d, x3d, xskip3d = x

        up2d = self.deconv2d(x2d)
        up3d = self.deconv3d(x3d)


        cat = ops.Concat(1)([FMU(xskip2d, xskip3d, self.FMU), FMU(up2d, up3d, self.FMU)])

        if self.mode_out == '2d':
            return self.CB2d(cat)
        elif self.mode_out == '3d':
            return self.CB3d(cat)
        else:
            return self.CB2d(cat), self.CB3d(cat)


class MNet(nn.Cell):
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
        self.ds = ds
        self.num_classes = num_classes

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


        self.outputs = nn.CellList(
            [nn.Conv3d(c, num_classes, kernel_size=(1, 1, 1), stride=1, pad_mode='valid',padding=0, has_bias=False)
             for c in [kn[0], kn[1], kn[1], kn[2], kn[2], kn[3], kn[3]]]
        )

    def construct(self, x):
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


        if self.ds:
            # outputs = [self.outputs[i](f) for i, f in enumerate([up14[0] + up14[1], up23, up13, up32, up12, up41, up11])]
            return self.outputs[0](up14[0] + up14[1]), self.outputs[1](up23), \
                   self.outputs[2](up13), self.outputs[3](up32),\
                   self.outputs[4](up12), self.outputs[5](up41),self.outputs[6](up11)
        else:
            return self.outputs[0](up14[0] + up14[1])
