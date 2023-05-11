
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """
    Implemented as described in
    Oktay, Ozan, et al. 
    "Attention u-net: Learning where to look for the pancreas." arXiv preprint arXiv:1804.03999 (2018).
    """
    def __init__(self, g_channels, x_channels, inter_channels=None):
        super(AttentionGate, self).__init__()

        if inter_channels is None:
            inter_channels = x_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        else:
            inter_channels = inter_channels

        self.g_channels = g_channels
        self.x_channels = x_channels
        self.inter_channels = inter_channels
        self.g_weights = None # size of padding only known at runtime
        
        self.x_weights = nn.Conv2d(in_channels=x_channels, out_channels=inter_channels, kernel_size=(1,1), stride=2)
        self.in_relu = nn.ReLU(inplace=True)

        self.value_weights = nn.Conv2d(in_channels=inter_channels, out_channels=1, kernel_size=(1,1))
        self.out_sigmoid = nn.Sigmoid()
        self.resampler = None

    def forward(self, x, g):

        if self.resampler is None:
            self.resampler = nn.Upsample(x.shape[2:], mode='bilinear', align_corners=False)

        if self.g_weights is None:
            # pad g to prevent shape mismatch
            padding_size_x = (x.shape[2] // 2 - g.shape[2]) // 2
            padding_size_y = (x.shape[3] // 2 - g.shape[3]) // 2
            self.g_weights = nn.Conv2d(in_channels=self.g_channels, out_channels=self.inter_channels,
                kernel_size=(1,1), padding=(padding_size_x, padding_size_y), padding_mode="replicate")

            if g.is_cuda:
                self.g_weights.to("cuda:0")

        g_weighted = self.g_weights(g)
        x_weighted = self.x_weights(x)

        # pad if necessary 
        if g_weighted.shape[2] < x_weighted.shape[2] or g_weighted.shape[3] < x_weighted.shape[3]:
            pad_x = x_weighted.shape[2] - g_weighted.shape[2]
            pad_y = x_weighted.shape[3] - g_weighted.shape[3]
            # https://stackoverflow.com/questions/48686945/reshaping-a-tensor-with-padding-in-pytorch
            g_weighted = F.pad(g_weighted, pad=(0, pad_y, 0, pad_x))

        x_g_added = x_weighted + g_weighted
        x_g_relued = self.in_relu(x_g_added)

        x_g_weighted = self.value_weights(x_g_relued)
        x_g_sigmoid = self.out_sigmoid(x_g_weighted)

        attention_values = self.resampler(x_g_sigmoid)
        output_x = attention_values * x

        return output_x

class RadarUNet(nn.Module):
    """
    Implemented as described in 
    Orr, Itai, Moshik Cohen, and Zeev Zalevsky. 
    "High-resolution radar road segmentation using weakly supervised learning." 
    Nature Machine Intelligence 3.3 (2021): 239-246.

    A good tutorial can also be found in :
    https://www.youtube.com/watch?v=u1loyDCoGbE
    """

    def double_conv(self, in_channels, out_channels, padding=0, last_relu=True):

        if last_relu:
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=padding, padding_mode="replicate"),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=padding, padding_mode="replicate"),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        else:
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=padding, padding_mode="replicate"),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=padding, padding_mode="replicate"))        
        return conv

    def crop_image(self, input_image, output_image):
        out_size_x = output_image.shape[2]
        in_size_x = input_image.shape[2]
        delta_x = in_size_x - out_size_x

        if delta_x % 2 == 0:
            delta_x_left = delta_x // 2
            delta_x_right = delta_x // 2 
        else:
            delta_x_left = delta_x // 2
            delta_x_right = delta_x // 2 + 1

        out_size_y = output_image.shape[3]
        in_size_y = input_image.shape[3]
        delta_y = in_size_y - out_size_y
        if delta_y % 2 == 0:
            delta_y_left = delta_y // 2
            delta_y_right = delta_y // 2
        else:
            delta_y_left = delta_y // 2
            delta_y_right = delta_y // 2 + 1

        cropped_image = input_image[:, :, delta_x_left:-delta_x_right, delta_y_left:-delta_y_right]
        return cropped_image

    def __init__(self, output_image_shape, input_channels=1, enable_attention=True, segmentation=False):
        """
        output_image_shape -> Shape of output image without channel shape
        input_channels -> Number of input shapes for exammple two for real and complex
        segmentation -> if segmentation is enabled a softmax is applied
        """
        super(RadarUNet, self).__init__()

        self.enable_attention = enable_attention
        self.segmentation = segmentation

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = self.double_conv(input_channels, 64)
        self.down_conv2 = self.double_conv(64, 128)
        self.down_conv3 = self.double_conv(128, 256)
        self.down_conv4 = self.double_conv(256, 512)
        self.down_conv5 = self.double_conv(512, 1024)

        self.down_drop1 = nn.Dropout2d(0.2)
        self.down_drop2 = nn.Dropout2d(0.2)
        self.down_drop3 = nn.Dropout2d(0.2)
        self.down_drop4 = nn.Dropout2d(0.2)

        self.up_trans_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=(2,2), stride=2)
        self.up_trans_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=(2,2), stride=2)
        self.up_trans_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=(2,2), stride=2)
        self.up_trans_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=(2,2), stride=2)

        self.up_conv1 = self.double_conv(1024, 512, padding=3)
        self.up_conv2 = self.double_conv(512, 256, padding=3)
        self.up_conv3 = self.double_conv(256, 128, padding=3)
        self.up_conv4 = self.double_conv(128, 64, padding=3)

        if segmentation:
            self.final_up_conv = self.double_conv(64, 2)
            self.final_softmax = nn.Softmax2d()
        else:
            self.final_up_conv = self.double_conv(64, 1, last_relu=False)
        
        self.final_upsampler = nn.Upsample(output_image_shape, mode='bilinear', align_corners=False)
        
        if enable_attention:
            # from bottom to up
            self.att_layer1 = AttentionGate(g_channels=1024, x_channels=512)
            self.att_layer2 = AttentionGate(g_channels=512, x_channels=256)
            self.att_layer3 = AttentionGate(g_channels=256, x_channels=128)
            self.att_layer4 = AttentionGate(g_channels=128, x_channels=64)

    def forward(self, x):
        """
        expecting following dimensions: (batch, channel, width, height)
        """

        # downsampling path
        x1 = self.down_conv1(x) 
        #x1 = self.down_drop1(x1)
        x2 = self.max_pool_2x2(x1) 

        x3 = self.down_conv2(x2)
        #x3 = self.down_drop2(x3)
        x4 = self.max_pool_2x2(x3) 

        x5 = self.down_conv3(x4) 
        #x5 = self.down_drop3(x5)
        x6 = self.max_pool_2x2(x5) 

        x7 = self.down_conv4(x6) 
        #x7 = self.down_drop4(x7)

        if self.enable_attention:
            x_left2 = self.att_layer2(x5, x7)
        else:
            x_left2 = x5
        x_up2 = self.up_trans_conv2(x7)
        x5_cropped = self.crop_image(x_left2, x_up2)
        x_comb2 = torch.cat((x5_cropped, x_up2), 1)
        x_comb2 = self.up_conv2(x_comb2)

        if self.enable_attention:
            x_left3 = self.att_layer3(x3, x_comb2)
        else:
            x_left3 = x3
        x_up3 = self.up_trans_conv3(x_comb2)
        x3_cropped = self.crop_image(x_left3, x_up3)
        x_comb3 = torch.cat((x3_cropped, x_up3), 1)
        x_comb3 = self.up_conv3(x_comb3)

        if self.enable_attention:
            x_left4 = self.att_layer4(x1, x_comb3)
        else:
            x_left4 = x1
        x_up4 = self.up_trans_conv4(x_comb3)
        x1_cropped = self.crop_image(x_left4, x_up4)
        x_comb4 = torch.cat((x1_cropped, x_up4), 1)
        x_comb4 = self.up_conv4(x_comb4)

        final_output = self.final_up_conv(x_comb4)
        final_output = self.final_upsampler(final_output)

        if self.segmentation:
            final_output = self.final_softmax(final_output)
        
        return final_output