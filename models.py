from torch import nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()#平坦化
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 512),#線形変換
            nn.ReLU(),#活性化関数 繰り返し
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    
def test_accuracy(model, dataloader):
    # すべてのミニバッチに対して推論し、正解率を計算する
    n_corrects = 0# 正解数
    
    #モデルのデバイスを調べる
    device = next(model.parameters()).device
    model.eval()
    #モデルにいれて結果(logits)を表示
    with torch.no_grad():
        for image_batch, label_batch in dataloader:
            # ミニバッチをモデルのデバイスに転送
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
    
            logits_batch = model(image_batch)
            

            predict_batch = logits_batch.argmax(dim=1)
            n_corrects += (label_batch == predict_batch).sum().item()
    #正解＝正解率の計算
    accuracy = n_corrects / len(dataloader.dataset)

    return accuracy

def train(model, dataloader,loss_fn, optimizer):
    """1エポック分の学習を実行する関数"""

    #モデルのデバイスを調べる
    device = next(model.parameters()).device
    model.train()
    for image_batch, label_batch in dataloader:
         # ミニバッチをモデルのデバイスに転送
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)
        # 順伝播
    
        logits_batch = model(image_batch)

        # 損失の計算
        loss = loss_fn(logits_batch, label_batch)
    

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 最後のバッチのロス
    return loss.item()

def test(model, dataloader,loss_fn):
    """1エポック分のロスを計算する関数"""
    loss_total = 0.0

    #モデルのデバイスを調べる
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for image_batch, label_batch in dataloader:
            # ミニバッチをモデルのデバイスに転送
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
        # 順伝播
       
            logits_batch = model(image_batch)

        # 損失の計算
            loss = loss_fn(logits_batch, label_batch)
            loss_total += loss.item() 
        
    # 最後のバッチのロス
    return loss_total/len(dataloader)