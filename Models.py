import torch.nn as nn
import torch
from utils.path_hyperparameter import ph
import numbers
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse


class Conv_BN_ReLU(nn.Module):
    """ Basic convolution."""

    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                                                    padding=kernel // 2, bias=False, stride=stride),
                                          nn.BatchNorm2d(out_channel),
                                          nn.ReLU(inplace=True),
                                          )

    def forward(self, x):
        output = self.conv_bn_relu(x)

        return output

def channel_split(x):
    """Half segment one feature on channel dimension into two features, mixture them on channel dimension,
    and split them into two features."""
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]

class CGSU(nn.Module):
    """Basic convolution module."""

    def __init__(self, in_channel):
        super().__init__()

        mid_channel = in_channel // 2

        self.conv1 = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )

    def forward(self, x):
        x1, x2 = channel_split(x)
        x1 = self.conv1(x1)
        output = torch.cat([x1, x2], dim=1)

        return output

class CGSU_DOWN(nn.Module):
    """Basic convolution module with stride=2."""

    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1,
                                             stride=2, bias=False),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )
        self.conv_res = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # remember the tensor should be contiguous
        output1 = self.conv1(x)

        # respath
        output2 = self.conv_res(x)

        output = torch.cat([output1, output2], dim=1)

        return output

class MultiScaleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

class MEB(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel * 2, 'the out_channel is not in_channel*2 in encoder block'
        self.conv1 = nn.Sequential(
            CGSU_DOWN(in_channel=in_channel),
            CGSU(in_channel=out_channel),
            CGSU(in_channel=out_channel)
        )
        self.conv2 = Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1)
        self.attention = MultiScaleAttention(dim=out_channel)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.attention(x)

        return output

class Decoder_Block(nn.Module):
    """Basic block in decoder."""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, de):
        de = self.up(de)
        output = self.conv(de)

        return output

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Low_Interaction(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Low_Interaction, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.iwt = DWTInverse(wave='haar', mode='zero')

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x1, x2):

        yL1, yH1 = self.wt(x1)
        yL2, yH2 = self.wt(x2)

        yL1 = self.conv1(yL1)
        yL2 = self.conv2(yL2)

        x1_t = self.iwt((yL2, yH1))
        x2_t = self.iwt((yL1, yH2))

        return x1_t, x2_t

class High_extra(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(High_extra, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = nn.Sequential(
                                    nn.Conv2d(in_ch*3, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL1, yH1 = self.wt(x)

        y_HL = yH1[0][:,:,0,::]
        y_LH = yH1[0][:,:,1,::]
        y_HH = yH1[0][:,:,2,::]

        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv(yH)

        yH = self.outconv(yH)

        return yH

class AGP(nn.Module):
    def __init__(self, dim, num_heads):
        super(AGP, self).__init__()
        self.num_heads = num_heads
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_1_3 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv1_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv1_2_3 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv2_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_1_3 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv2_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_2_3 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

    def forward(self, x1, x2):
        b, c, h, w = x2.shape
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)
        attn_113 = self.conv1_1_3(x1)
        attn_121 = self.conv1_2_1(x1)
        attn_122 = self.conv1_2_2(x1)
        attn_123 = self.conv1_2_3(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_212 = self.conv2_1_2(x2)
        attn_213 = self.conv2_1_3(x2)
        attn_221 = self.conv2_2_1(x2)
        attn_222 = self.conv2_2_2(x2)
        attn_223 = self.conv2_2_3(x2)

        out1 = attn_111 + attn_112 + attn_113 +attn_121 + attn_122 + attn_123
        out2 = attn_211 + attn_212 + attn_213 +attn_221 + attn_222 + attn_223

        out1 = self.conv(out1)
        out2 = self.conv(out2)

        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2

        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.conv(out4) + x1 + x2

        return out


class SC(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.fuse = nn.Sequential(nn.Conv2d(in_channels=in_channel + out_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, de, en):
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)

        return output


class BiLIM(nn.Module):
    """Exchange channels of two feature uniformly-spaced with 1:1 ratio."""

    def __init__(self, in_channel):
        super().__init__()
        self.low = Low_Interaction(in_channel, in_channel)
        self.AGP_T1 = AGP(in_channel, num_heads=8)
        self.AGP_T2 = AGP(in_channel, num_heads=8)

    def forward(self, x1, x2):
        x1_L, x2_L = self.low(x1, x2)

        out_x1 = self.AGP_T1(x1_L, x1)
        out_x2 = self.AGP_T2(x2_L, x2)

        return out_x1, out_x2

class HEM(nn.Module):
    def __init__(self, in_channel_1, in_channel_2):
        super(HEM, self).__init__()

        self.high1 = High_extra(in_channel_1, in_channel_2)
        self.high2 = High_extra(in_channel_1, in_channel_2)
        self.AGP_1 = AGP(in_channel_2,num_heads=8)
        self.AGP_2 = AGP(in_channel_2,num_heads=8)

    def forward(self, x1, x2, d1, d2):
        
        x1_h = self.high1(x1)
        x2_h = self.high2(x2)

        out1 = self.AGP_1(x1_h, d1)
        out2 = self.AGP_2(x2_h, d2)

        out = out1 + out2

        return out

class FASINet(nn.Module):
    def __init__(self):
        super().__init__()

        channel_list = [32, 64, 128, 256, 512]
        # encoder
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=1),
                                       CGSU(in_channel=channel_list[0]),
                                       CGSU(in_channel=channel_list[0]),
                                       )        
        self.en_block2 = MEB(in_channel=channel_list[0], out_channel=channel_list[1])
        self.en_block3 = MEB(in_channel=channel_list[1], out_channel=channel_list[2])
        self.en_block4 = MEB(in_channel=channel_list[2], out_channel=channel_list[3])
        self.en_block5 = MEB(in_channel=channel_list[3], out_channel=channel_list[4])
        self.interaction = BiLIM(in_channel=channel_list[2])

        # fusion
        self.fuse = HEM(in_channel_1=channel_list[3], in_channel_2=channel_list[4])

        # decoder
        self.fu_block = SC(in_channel=channel_list[3], out_channel=channel_list[3])

        self.de_block1 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.de_block2 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.de_block3 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])
        self.de_block4 = Decoder_Block(in_channel=channel_list[1], out_channel=channel_list[0])

        self.conv_out_change = nn.Conv2d(channel_list[0], 1, kernel_size=7, stride=1, padding=3)

    def forward(self, t1, t2):

        t1_1 = self.en_block1(t1)
        t2_1 = self.en_block1(t2)

        t1_2 = self.en_block2(t1_1)
        t2_2 = self.en_block2(t2_1)

        t1_3 = self.en_block3(t1_2)
        t2_3 = self.en_block3(t2_2)
        t1_3, t2_3 = self.interaction(t1_3, t2_3)

        t1_4 = self.en_block4(t1_3)
        t2_4 = self.en_block4(t2_3)

        t1_5 = self.en_block5(t1_4)
        t2_5 = self.en_block5(t2_4)        
       
        de5 = self.fuse(t1_4, t2_4, t1_5, t2_5)

        de4 = self.de_block1(de5)
        fu4 = t1_4 + t2_4
        de4 = self.fu_block(de4, fu4)

        de3 = self.de_block2(de4) 

        de2 = self.de_block3(de3)

        de1 = self.de_block4(de2)

        change_out = self.conv_out_change(de1)

        return change_out