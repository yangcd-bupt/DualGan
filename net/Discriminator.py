from torch import nn


# 辨别器 PatchGAN
class Discriminator(nn.Module):
    def __init__(self, in_ch, ndf=64):
        """
        定义判别器的网络结构
        :param in_ch: 输入数据的通道数
        :param ndf: 第一层卷积的通道数 number of discriminator's first conv filters
        """
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, X):
        """
        判别器模块正向传播
        :param X: 输入判别器的数据
        :return: 判别器的输出
        """
        layer1_out = self.layer1(X)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)

        return layer5_out
