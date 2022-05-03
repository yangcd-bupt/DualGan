import torch
from torch import nn


# 生成器 U-Net
class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf=64):
        """
        定义生成器的网络结构
        :param in_ch: 输入数据的通道数
        :param out_ch: 输出数据的通道数
        :param ngf: 第一层卷积的通道数 number of generator's first conv filters
        """
        super(Generator, self).__init__()

        # U-Net encoder
        self.en1 = nn.Sequential(
            nn.Conv2d(in_ch, ngf, kernel_size=4, stride=2, padding=1),
        )
        self.en2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        self.en4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        self.en5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        self.en6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        self.en7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        self.en8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        )

        # U-Net decoder
        self.de1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        self.de2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        self.de3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        self.de4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        self.de5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout(p=0.5)
        )
        self.de6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout(p=0.5)
        )
        self.de7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.Dropout(p=0.5)
        )
        self.de8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, X):
        """
        生成器模块前向传播
        :param X: 输入生成器的数据
        :return: 生成器的输出
        """
        # Encoder
        en1_out = self.en1(X)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)
        en4_out = self.en4(en3_out)
        en5_out = self.en5(en4_out)
        en6_out = self.en6(en5_out)
        en7_out = self.en7(en6_out)
        en8_out = self.en8(en7_out)

        # Decoder
        de1_out = self.de1(en8_out)
        de1_cat = torch.cat([de1_out, en7_out], dim=1)
        de2_out = self.de2(de1_cat)
        de2_cat = torch.cat([de2_out, en6_out], 1)
        de3_out = self.de3(de2_cat)
        de3_cat = torch.cat([de3_out, en5_out], 1)
        de4_out = self.de4(de3_cat)
        de4_cat = torch.cat([de4_out, en4_out], 1)
        de5_out = self.de5(de4_cat)
        de5_cat = torch.cat([de5_out, en3_out], 1)
        de6_out = self.de6(de5_cat)
        de6_cat = torch.cat([de6_out, en2_out], 1)
        de7_out = self.de7(de6_cat)
        de7_cat = torch.cat([de7_out, en1_out], 1)
        de8_out = self.de8(de7_cat)

        return de8_out
