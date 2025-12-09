import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


# モデルをインスタンス化　モデル≒設計図
model = models.MyModel()#インスタンス≒設計図から設計されたもの
print(model)

# データセットのロード
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)])
)

#imageはPILではなくTensorに変換済み
image, target = ds_train[0]
image_PIL = image
#(1, H ,W)　から (1,1, H, W)へ次元を上げる
image = image.unsqueeze(0)  # バッチ次元を追加
print(image.shape, image.dtype)
# モデルにいれて結果を表示
model.eval()  # 評価モードに設定
with torch.no_grad():  # 勾配計算を無効化
    logits = model(image)

print(logits)

# ロジットモデルをグラフにする
#plt.bar(range(len(logits[0])), logits[0])
#plt.show()

# クラス確率をグラフにする 
probs = logits.softmax(dim=1)

plt.subplot(1,2,1)
plt.imshow(image[0,0,:,:], cmap='gray_r',vmin=0,vmax=1)#[1,1,28,28]の中の[0,0,:,:]image.squeeze(0)で元に戻すのも可
plt.title(f"class:{target} ({datasets.FashionMNIST.classes[target]})")

plt.subplot(1,2,2)
plt.bar(range(probs.shape[1]), probs[0])
plt.title(f"predicted class: {probs[0].argmax()}")#ニューラルネットワークがほぼないため、AIが超適当に回答している(本来は９固定のはず)
plt.ylim(0, 1)
plt.show()