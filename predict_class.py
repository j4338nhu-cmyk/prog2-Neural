from torchvision import io as tvio
from torchvision import models
import torchinfo

input_image = tvio.decode_image('assets/free_image_human.JPG')
print(type(input_image))
print(input_image.shape)
print(input_image.dtype)

#学習済みモデルの重みの読み込み
weights = models.AlexNet_Weights.DEFAULT

# モデルの作成
model = models.alexnet(weights=weights)
#print(model)

torchinfo.summary(model)

#モデルの前処理の方法を取得する
preprocess = weights.transforms()

#パッヂにする
batch = preprocess(input_image).unsqueeze(dim=0)
print(batch.shape)
#torch.size([1枚数, 3チャンネル, 224横, 224縦])

#推論モードに変更
model.eval()

#バッヂに対して推論(モデルの計算)を行う
output_logits = model(batch)
print(output_logits.shape, output_logits.dtype)
#パッヂ内のデータごとにクラス確率に変換
output_probs = output_logits.softmax(dim=1)

#パッチからインデックス０のデータのクラス確率を取得
#結果を表示
class_id = output_probs[0].argmax().item()
score = output_probs[0][class_id].item()
category_name = weights.meta['categories'][class_id]
print(f'{category_name}:{100*score:.1f}%')