import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

# データセットの読み込み
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(f'dataset size: {len(ds_train)}')

#インデックスとデータを指定して、データを取り出す
# データ（画像）とターゲット（クラス番号）のタプル
image , target = ds_train[0]

print(type(image))
print(target)

#画像を表示
#plt.imshow(image, cmap='gray_r')
#plt.title(target)
#plt.show()

fig,ax = plt.subplots()
ax.imshow(image, cmap='gray_r',vmin=0,vmax=255)
ax.set_title(target)
plt.show()

image = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image, scale=True)
plt.subplot(1,2,2)
print(type(image))
print(image.shape,image.dtype)
print(image.min(), image.max())

