from .yolov5_afpn import *
from mmyolo.models.utils import make_divisible

class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 8, 8, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x

class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_4, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_0 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input0, input1, input2, input3):
        level_0_weight_v = self.weight_level_0(input0)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :] + \
                            input3 * levels_weight[:, 3:, :, :]

        out = self.conv(fused_out_reduced)

        return out

class BlockBody(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super(BlockBody, self).__init__()

        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )
        self.blocks_scalethree1 = nn.Sequential(
            BasicConv(channels[3], channels[3], 1),
        )

        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        self.blocks_scalezero3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)

        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])

        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
        self.blocks_scalethree4 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
        x3 = self.blocks_scalethree1(x3)

        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)

        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        scalezero = self.asff_scalezero3(x0, self.upsample_scaleone3_2(x1), self.upsample_scaletwo3_4(x2), self.upsample_scalethree3_8(x3))
        scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(x0), x1, self.upsample_scaletwo3_2(x2), self.upsample_scalethree3_4(x3))
        scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(x0), self.downsample_scaleone3_2(x1), x2, self.upsample_scalethree3_2(x3))
        scalethree = self.asff_scalethree3(self.downsample_scalezero3_8(x0), self.downsample_scaleone3_4(x1), self.downsample_scaletwo3_2(x2), x3)

        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)
        scalethree = self.blocks_scalethree4(scalethree)

        return scalezero, scaleone, scaletwo, scalethree

class YOLOv8AFPN4(nn.Module):
    def __init__(self, in_channels=[128,256, 512, 1024], out_channels=[128,256, 512, 1024]):
        super(YOLOv8AFPN4, self).__init__()
        self.fp16_enabled = False

        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)
        self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1)

        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8])
        )

        self.conv00 = BasicConv(in_channels[0] // 8, out_channels[0], 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels[1], 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels[2], 1)
        self.conv33 = BasicConv(in_channels[3] // 8, out_channels[3], 1)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)
        print(out0.shape)
        print(out1.shape)
        print(out2.shape)
        print(out3.shape)
        return tuple([out0, out1, out2, out3])


@MODELS.register_module()
class YOLOv8AFPN(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], out_channels=[256, 512, 1024],widen_factor=1.0):
        super(YOLOv8AFPN, self).__init__()
        in_channels = [make_divisible(i,widen_factor) for i in in_channels]
        out_channels = [make_divisible(i,widen_factor) for i in out_channels]
        if len(in_channels) == 3:
            self.afpn = YOLOv5AFPN(in_channels,out_channels)
        else:
            self.afpn = YOLOv8AFPN4(in_channels,out_channels)
    def forward(self, x):
        return self.afpn(x)
