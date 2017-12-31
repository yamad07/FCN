class DataEncoder:

    def __init__(self, width=768, height=512):
        self.img_channels = 3
        self.pred_channels = 2
        self.width = width
        self.height = height

    def encode(self, img, pred):
        img_flatten = img.view(self.img_channels, -1)
        flags, idx = pred.max(dim=0)
        idx_flatten = idx.view(-1)
        # 人がいそうなところにフラグが立っている場合のインデックスを取得
        true_idx_flatten = idx_flatten.nonzero()
        # フラグが立っているところを0で埋める
        img_with_flags_flatten = img_flatten.index_fill_(0, true_idx_flatten, 0)
        img_with_flgs = img_with_flags_flatten.view(self.img_channels, self.width, self.height)
        return img_with_flags

    def decode(self):
        pass
