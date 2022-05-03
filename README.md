# DualGan
使用Pytorch实现对偶生成对抗网络来实现图像去雾
## 使用说明
DualGan含有两个生成器和辨别器  
本项目中为同样结构，生成器为U-Net，辨别器为PatchGan的辨别器  
G_A：有雾生成无雾  
G_B: 无雾生成有雾  
D_A: 辨别G_B生成的有雾图像，输入为6通道  
D_B: 辨别G_A生成的无雾图像，输入为6通道  
train.py用来训练网络  
predict.py用来预测无雾图像  
预训练好的模型在model下
## 训练方法
将成对的图片分别放在clear和hazy下，并放在data_path下  
然后运行train.py并输入需要的参数即可  
保存路径--save_path, default="./save/"  
训练数据路径--data_path,  default="../Data/"  
训练数据清晰图像路径--clear， default="clear"  
训练数据有雾图像路径--hazy", default="hazy"  
--image_size,  default=256   
--batch_size,  default=4  
保存/加载G_A路径--g_a_path,  default="generator_a.pkl"  
保存/加载G_B路径--g_b_path,  default="generator_b.pkl"  
保存/加载D_A路径--d_a_path,  default="discriminator_a.pkl"  
保存/加载D_B路径--d_b_path,  default="discriminator_b.pkl"  
## 预测方法
将要训练的图片放到data_path的hazy下  
运行predict.py并输入需要的参数即可  


