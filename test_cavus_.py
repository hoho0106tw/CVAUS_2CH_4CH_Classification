#!/usr/bin/env python
# coding: utf-8


import torch
from torchvision import models, transforms
from PIL import Image

# ------------------------------------------------------------
# 1. 設定 label 名稱（依你訓練時順序）
# ------------------------------------------------------------
class_names = ["2ch", "4ch"]

# ------------------------------------------------------------
# 2. 載入模型
# ------------------------------------------------------------
def load_model(weight_path="densenet121_cvus.pth"):
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    print("模型載入完成！")
    return model

# ------------------------------------------------------------
# 3. Transform（務必與你訓練時相同）
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------------------------------------
# 4. 單張圖片推論
# ------------------------------------------------------------
def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    label = class_names[pred.item()]
    print(f"📌 預測結果： {label}")
    return label

# ------------------------------------------------------------
# 5. 載入模型並測試
# ------------------------------------------------------------
model = load_model("densenet121_cvus_20251201.pth")

# 修改成你的 PNG 圖片路徑
img_path = "000.png"

predict_image(model, img_path)






