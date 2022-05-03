import torch
from DualGan.util.parseArgs import parseArgs
from DualGan.util.pre_loader import pre_loader
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict():
    parser=parseArgs()
    data_path = parser.data_path
    save_path = parser.save_path
    hazy_path=parser.hazy
    batch_size=parser.batch_size
    image_size = parser.image_size
    G_A = torch.load('generator_a.pkl')
    hazy_loader = pre_loader(data_path, hazy_path, image_size=image_size,batch_size=batch_size)
    for X,labelx in iter(hazy_loader):
        x=X.to(device)
        g_a=G_A(x)
        torch.nn.functional.interpolate(g_a,(labelx[0][0], labelx[0][1]),mode='bilinear', align_corners=True)
        save_image(g_a, save_path + labelx[0][2].split('.')[0] + '.jpg')


## 运行 ##
if __name__ == '__main__':
    predict()