import itertools
import os
import random
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(" -- 使用GPU进行训练 -- ")

def parseArgs():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--save_path", type=str, default="./result/")
    parser.add_argument("--data_path", type=str, default="../Data/")
    parser.add_argument("--origin", type=str, default="GT")
    parser.add_argument("--hazy", type=str, default="hazy")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    return args
## 生成器 U-Net（输入照片为256*256） ##
class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf=64):
        """
        定义生成器的网络结构
        :param in_ch: 输入数据的通道数
        :param out_ch: 输出数据的通道数
        :param ngf: 第一层卷积的通道数 number of generator's first conv filters
        """
        super(Generator, self).__init__()
        # 下面的激活函数都放在下一个模块的第一步 是为了skip-connect方便

        # 左半部分 U-Net encoder
        # 每层输入大小折半，从输入图片大小256开始
        # 256 * 256（输入）
        self.en1 = nn.Sequential(
            nn.Conv2d(in_ch, ngf, kernel_size=4, stride=2, padding=1),
            # 输入图片已正则化 不需BatchNorm
        )
        # 128 * 128
        self.en2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        # 64 * 64
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        # 32 * 32
        self.en4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 16 * 16
        self.en5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 8 * 8
        self.en6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 4 * 4
        self.en7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 2 * 2
        self.en8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
            # Encoder输出不用BatchNorm
        )

        # 右半部分 U-Net decoder
        # skip-connect: 前一层的输出+对称的卷积层
        # 1 * 1（输入）
        self.de1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 2 * 2
        self.de2 = nn.Sequential(
            nn.ReLU(inplace=True),
            # skip-connect 所以输入管道数是之前输出的2倍
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 4 * 4
        self.de3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 8 * 8
        self.de4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 16 * 16
        self.de5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout(p=0.5)
        )
        # 32 * 32
        self.de6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout(p=0.5)
        )
        # 64 * 64
        self.de7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.Dropout(p=0.5)
        )
        # 128 * 128
        self.de8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, out_ch, kernel_size=4, stride=2, padding=1),
            # Encoder输出不用BatchNorm
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
        de1_cat = torch.cat([de1_out, en7_out], dim=1)  # cat by channel
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


## 辨别器 PatchGAN（其实就是卷积网络而已） ##
class Discriminator(nn.Module):
    def __init__(self, in_ch,  ndf=64):
        """
        定义判别器的网络结构
        :param in_ch: 输入数据的通道数
        :param ndf: 第一层卷积的通道数 number of discriminator's first conv filters
        """
        super(Discriminator, self).__init__()

        # 不是输出一个表示真假概率的实数，而是一个N*N的Patch矩阵（此处为30*30），其中每一块对应输入数据的一小块
        # in_ch + out_ch 是为将对应真假数据同时输入
        # 256 * 256（输入）
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1),
            # 输入图片已正则化 不需BatchNorm
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 128 * 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64 * 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32 * 32
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 31 * 31
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        # 30 * 30（输出的Patch大小）
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


class MyDataset(Dataset):
    def __init__(self, root, subfolder, transform=None):
        """
        自定义数据集初始化
        :param root: 数据文件根目录
        :param subfolder: 数据文件子目录
        :param transform: 预处理方法
        """
        super(MyDataset, self).__init__()
        self.path = os.path.join(root, subfolder)
        self.image_list = [x for x in os.listdir(self.path)]
        self.transform = transform

    def __len__(self):
        """
        以便可以len(dataset)形式返回数据大小
        :return: 数据集大小
        """
        return len(self.image_list)

    def __getitem__(self, item):
        """
        支持索引以便dataset可迭代获取
        :param item: 索引
        :return: 索引对应的数据单元
        """
        image_path = os.path.join(self.path, self.image_list[item])
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]] # BGR -> RGB
        if self.transform is not None:
            image = self.transform(image)

        # Dataset每个数据单元要求返回一个数据一个标签 此处标签无意义（但不能直接设为None）
        lable = self.image_list[item]
        return image, lable

## 加载数据（Facades） ##
def loadData(root, subfolder, batch_size, shuffle=False):
    """
    加载数据以返回DataLoader类型
    :param root: 数据文件根目录
    :param subfolder: 数据文件子目录
    :param batch_size: 批处理样本大小
    :param shuffle: 是否打乱数据（默认为否）
    :return: DataLoader类型的可迭代数据
    """
    # 数据预处理方式
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([256,256]),
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # (0, 1) -> (-1, 1)
    ])
    # 创建Dataset对象
    dataset = MyDataset(root, subfolder, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

## 训练判别器A ##
def D_A_train(D_A: Discriminator,  G_B: Generator,X, Y,BCELoss, optimizer_D):
    """
    训练判别器
    :param D: 判别器
    :param G: 生成器
    :param X: 未分隔的数据
    :param BCELoss: 二分交叉熵损失函数
    :param optimizer_D: 判别器优化器
    :return: 判别器的损失值
    """
    # 标签转实物（右转左）
    #image_size = X.size(3) // 2
    x = X.to(device)  # 标签图
    y = Y.to(device)  # 实物图
    #print(xy.size())
    # 梯度初始化为0
    yx = torch.cat([y, x], dim=1)  # 在channel维重叠 xy!=X
    D_A.zero_grad()
    # 在真数据上
    D_output_r = D_A(yx).squeeze()
    # 在假数据上
    G_B_output = G_B(y)
    yx_fake = torch.cat([y, G_B_output], dim=1)  # 在channel维重叠 xy!=X
    D_output_f = D_A(yx_fake).squeeze()
    if random.random() < 0.1:
        D_real_loss = BCELoss(D_output_r, torch.zeros(D_output_r.size()).to(device))
        D_fake_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
    else:
        D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device))
        D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))
    # 反向传播并优化
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()

    return D_loss.data.item()
## 训练判别器B ##
def D_B_train(D_B: Discriminator,  G_A: Generator,X, Y,BCELoss, optimizer_D):
    """
    训练判别器
    :param D: 判别器
    :param G: 生成器
    :param X: 未分隔的数据
    :param BCELoss: 二分交叉熵损失函数
    :param optimizer_D: 判别器优化器
    :return: 判别器的损失值
    """
    # 标签转实物（右转左）
    #image_size = X.size(3) // 2
    x = X.to(device)  # 标签图
    y = Y.to(device)  # 实物图
    #print(xy.size())
    # 梯度初始化为0
    D_B.zero_grad()
    # 在真数据上
    xy = torch.cat([x, y], dim=1)  # 在channel维重叠 xy!=X
    D_output_r = D_B(xy).squeeze()
    # 在假数据上
    G_A_output = G_A(x)
    xy_fake = torch.cat([x, G_A_output], dim=1)  # 在channel维重叠 xy!=X
    D_output_f = D_B(xy_fake).squeeze()
    if random.random() < 0.1:
        D_real_loss = BCELoss(D_output_r, torch.zeros(D_output_r.size()).to(device))
        D_fake_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
    else:
        D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device))
        D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))

    # 反向传播并优化
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()

    return D_loss.data.item()

## 训练生成器 ##
def G_train(D_A: Discriminator,D_B: Discriminator, G_A: Generator,G_B: Generator, X,Y, BCELoss, L1, optimizer_G, lamb=100):
    """
    训练生成器
    :param D: 判别器
    :param G: 生成器
    :param X: 未分隔的数据
    :param BCELoss: 二分交叉熵损失函数
    :param L1: L1正则化函数
    :param optimizer_G: 生成器优化器
    :param lamb: L1正则化的权重
    :return: 生成器的损失值
    """
    # 标签转实物（右转左）
    #image_size = X.size(3) // 2
    x = X.to(device)  # 标签图（右半部分）
    y = Y.to(device)  # 实物图（左半部分）

    # 梯度初始化为0
    G_A.zero_grad()
    G_B.zero_grad()
    # 在假数据上
    G_A_output = G_A(x)
    xy_fake = torch.cat([x, G_A_output], dim=1)  # 在channel维重叠 xy!=X
    D_B_output_f = D_B(xy_fake).squeeze()
    G_A_BCE_loss = BCELoss(D_B_output_f, torch.ones(D_B_output_f.size()).to(device))
    G_A_L1_Loss = L1(G_A_output, y)
    # 反向传播并优化
    G_A_loss = G_A_BCE_loss + lamb * G_A_L1_Loss
    # 在假数据上
    G_B_output = G_B(y)
    yx_fake=torch.cat([y, G_B_output], dim=1)  # 在channel维重叠 xy!=X
    D_A_output_f = D_A(yx_fake).squeeze()
    G_B_BCE_loss = BCELoss(D_A_output_f, torch.ones(D_A_output_f.size()).to(device))
    G_B_L1_Loss = L1(G_B_output, x)
    # 反向传播并优化
    G_B_loss = G_B_BCE_loss + lamb * G_B_L1_Loss
    G_loss=G_A_loss+G_B_loss
    G_loss.backward(retain_graph=True)
    optimizer_G.step()

    return G_loss.data.item()
def showplt(X,name):
    array1=X[0].numpy()#将tensor数据转为numpy数据
    maxValue=array1.max()
    array1=array1*255/maxValue#normalize，将图像数据扩展到[0,255]
    mat=np.uint8(array1)#float32-->uint8
    mat=mat.transpose(1,2,0)#mat_shape: (982, 814，3)
    cv2.imshow(name,mat)

## 主函数：训练Pix2Pix网络 ##
def main():
    # 加载训练数据
    args=parseArgs()
    save_path = args.save_path
    data_path = args.data_path
    GT_path = args.origin
    hazy_path = args.hazy
    batch_size = args.batch_size
    GT_loader = loadData(data_path, GT_path, batch_size)
    hazy_loader=loadData(data_path,hazy_path,batch_size)

    # 定义结构参数
    in_ch, out_ch = 3, 3  # 输入输出图片通道数
    ngf, ndf = 64, 64  # 生成数、判别器第一层卷积通道数
    image_size = 256  # 图片大小

    # 定义训练参数
    lr_G, lr_D = 0.0005, 0.0001  # G、D的学习速率
    beta1 = 0.5  # momentum term of Adam（一般用的是0.9）
    lamb = 100  # 在生成器的目标函数中L1正则化的权重
    epochs = 200  # 训练迭代次数

    # 声明生成器、判别器
    G_A = Generator(in_ch, out_ch, ngf).to(device)#生成无雾
    G_B=Generator(in_ch, out_ch, ngf).to(device)#生成有雾
    D_A = Discriminator(in_ch+out_ch, ndf).to(device)#鉴别有雾
    D_B = Discriminator(in_ch+out_ch, ndf).to(device)#鉴别无雾

    # 目标函数 & 优化器
    BCELoss = nn.BCELoss().to(device)
    L1 = nn.L1Loss().to(device)  # Pix2Pix论文中在传统GAN目标函数加上了L1
    optimizer_G = optim.RMSprop(itertools.chain(G_A.parameters(), G_B.parameters()), lr=lr_G, alpha=0.99,eps=1e-08)
    optimizer_D_A = optim.RMSprop(D_A.parameters(), lr=lr_D, alpha=0.99,eps=1e-08)
    optimizer_D_B = optim.RMSprop(D_B.parameters(), lr=lr_D, alpha=0.99, eps=1e-08)
    # 输入数据 & ground-truth & 初始生成器的输出
    X, labelx = next(iter(hazy_loader))
    Y, labely = next(iter(GT_loader))

    g_a = G_A(X.to(device))#有雾-》无雾
    g_b = G_B(Y.to(device))#无雾-》有雾
    print(g_a.size())
    save_image(X, save_path + labelx[0].split('.')[0]+'_input.png')
    save_image(Y, save_path + labely[0].split('.')[0]+'_ground-truth.png')
    save_image(g_a, save_path+labelx[0].split('.')[0]+ '_sample_GT_0.png')
    save_image(g_b, save_path + labelx[0].split('.')[0] + '_sample_hazy_0.png')

    # 开始训练
    G_A.train()  # （区分.eval）
    G_B.train()  # （区分.eval）
    D_A.train()  # （ .train不启用BatchNorm、Dropout）
    D_B.train()  # （ .train不启用BatchNorm、Dropout）
    D_A_Loss, D_B_Loss, G_Loss, Epochs = [], [],[], range(1, epochs + 1)  # 对一次epoch的loss数据操作
    for epoch in range(epochs):
        D_A_losses,D_B_losses, G_losses, batch, d_a_l,d_b_l, g_l = [], [],[], 0, 0,0, 0  # 对一次batch的loss数据操作
        GT_iter = iter(GT_loader)
        hazy_iter = iter(hazy_loader)
        for i in range(0, len(GT_loader)):
            X, _ = next(hazy_iter)
            Y, _ = next(GT_iter)
            # 每次epoch最大为10
            batch += 1
            # 训练Discriminator并保存loss
            D_A_losses.append(D_A_train(D_A, G_B, X, Y, BCELoss, optimizer_D_A))
            D_B_losses.append(D_B_train(D_B, G_A, X, Y, BCELoss, optimizer_D_B))
            # 训练Generator
            G_losses.append(G_train(D_A,D_B, G_A,G_B, X, Y, BCELoss, L1, optimizer_G, lamb))
            if batch % 1000 == 1:
                # 打印每十次batch的平均loss
                d_a_l,d_b_l, g_l = np.array(D_A_losses).mean(),np.array(D_B_losses).mean(), np.array(G_losses).mean()
                print('[%d / %d]: batch#%d loss_d_a= %.3f  loss_d_b= %.3f  loss_g= %.3f' %
                      (epoch + 1, epochs, batch, d_a_l,d_b_l, g_l))
        # 测试每十次epoch的生成效果
        if (epoch + 1) % 10 == 0:
            X, labelx = next(iter(hazy_loader))
            Y, labely = next(iter(GT_loader))
            g_a = G_A(X.to(device))
            g_b = G_B(Y.to(device))
            save_image(g_a, save_path + labelx[0].split('.')[0] + '_sample_GT_' + str(epoch + 1) + '.jpg')
            save_image(g_b, save_path + labelx[0].split('.')[0] + '_sample_hazy_' + str(epoch + 1) + '.jpg')
        # 保存每次epoch的loss
        D_A_Loss.append(d_a_l)
        D_B_Loss.append(d_b_l)
        G_Loss.append(g_l)
    print("Done!")

    # 保存训练结果
    torch.save(G_A, 'generator_a.pkl')
    torch.save(G_B, 'generator_b.pkl')
    torch.save(D_A, 'discriminator_a.pkl')
    torch.save(D_B, 'discriminator_b.pkl')
    '''
    G = torch.load('generator.pkl')
    D = torch.load('discriminator.pkl')
    '''

    # 画出loss图
    # G的loss因为包含L1 相比D的loss太大了 画图效果不好 所以除以100
    plt.plot(Epochs, D_A_Loss, label='Discriminator_A Losses')
    plt.plot(Epochs, D_B_Loss, label='Discriminator_B Losses')
    plt.plot(Epochs, np.array(G_Loss) / 100, label='Generator_A Losses / 100')
    plt.legend()
    plt.savefig(save_path + 'loss.png')
    plt.show()


def predict():
    data_path = '../../Data/'
    save_path = './predict_512/'
    hazy_path='indoor_hazy'
    batch_size=1
    G_A = torch.load('generator_a.pkl')
    hazy_loader = loadData(data_path, hazy_path, batch_size)
    for X,labelx in iter(hazy_loader):
        x=X.to(device)
        g_a=G_A(x)
        torch.nn.functional.interpolate(g_a,(512,512),mode='bilinear', align_corners=True)
        save_image(g_a, save_path + labelx[0].split('.')[0] + '.jpg')


## 运行 ##
if __name__ == '__main__':
    main()

