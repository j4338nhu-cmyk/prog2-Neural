import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


#データセットの前処理を定義
ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

# データセットのロード
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)

ds_test = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)
#バッチ　画像をまとめて複数枚同時に与えること、分割したものはミニバッチとも呼ばれる(バッチでも可)　分割元はエポックという
#ミニバッチに分割するためのデータローダーを作成
batch_size = 64#略称bs
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True,#データをシャッフルする
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)
# バッチを取り出す実験
for image_batch, target_batch in dataloader_train:
    print(image_batch.shape, image_batch.dtype)
    print(target_batch.shape, target_batch.dtype)
    break

#モデルをインスタンス化　
model = models.MyModel()

#損失関数の選択

loss_fn = torch.nn.CrossEntropyLoss()

#最適化手法の選択
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_epochs = 20#エポック数＝周回数
x_epochs = list(range(1, n_epochs+1))
train_loss_log = []
val_loss_log = []
train_acc_log = []
val_acc_log = []
for epoch in range(n_epochs):#n_epochs回繰り返す
    print(f'epoch {epoch+1}/{n_epochs}')
    time_start = time.time()
    train_loss = models.train(model, dataloader_train, loss_fn, optimizer)
    time_end = time.time()
    print(f'    training loss: {train_loss} ({time_end - time_start}s)')
    train_loss_log.append(train_loss)

    val_loss = models.test(model, dataloader_train, loss_fn)
    print(f'    validation loss: {val_loss}')
    val_loss_log.append(val_loss)

    train_acc = models.test_accuracy(model, dataloader_train)
    print(f'    training accuracy: {train_acc*100:.3f}%')
    train_acc_log.append(train_acc)

    val_acc = models.test_accuracy(model, dataloader_test)
    print(f'    validation accuracy: {val_acc*100:.3f}%')
    val_acc_log.append(val_acc)
#グラフの表示
plt.subplot(1,2,1)
plt.plot(x_epochs,train_loss_log,label='train')
plt.plot(x_epochs,val_loss_log,label='validation')
plt.xlabel('epochs')
plt.xticks(range(1, n_epochs+1))#1づつになるように指定　任意
plt.ylabel('loss')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(x_epochs,train_acc_log,label='train')
plt.plot(x_epochs,val_acc_log,label='validation')
plt.xlabel('epochs')
plt.xticks(range(1, n_epochs+1))#1づつになるように指定　任意
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()