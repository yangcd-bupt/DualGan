class PreDataset(Dataset):
    def __init__(self, root, subfolder, transform=None):
        """
        自定义数据集初始化
        :param root: 数据文件根目录
        :param subfolder: 数据文件子目录
        :param transform: 预处理方法
        """
        super(PreDataset, self).__init__()
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
        lable = [image.size()[0],image.size()[1],self.image_list[item]]
        return image, lable


def pre_loader(root, subfolder, image_size=256,batch_size=4, shuffle=False):
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
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # (0, 1) -> (-1, 1)
    ])
    # 创建Dataset对象
    dataset = PreDataset(root, subfolder, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)