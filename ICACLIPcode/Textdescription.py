import torch
import os
import clip
import numpy as np
from torchvision import transforms
import pandas as pd
from PIL import Image
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_model=".\ICmodel\IC1.pth"
save_model1=".\content\Contentmodel1.pth"
save_model2=".\color\Colormodel1.pth"
save_model3=".\distribution\Distribmodel1.pth"

train_model=torch.load(save_model,weights_only=False)
train_model1=torch.load(save_model1,weights_only=False)
train_model2=torch.load(save_model2,weights_only=False)
train_model3=torch.load(save_model3,weights_only=False)

model=train_model.clip_model
model1=train_model1.clip_model
model2=train_model2.clip_model
model3=train_model3.clip_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 标签
complexity = ["complex", "moderately complex", "simple"]
content = ["abundant", "moderate", "monotonous"]
color = ["rich", "general", "single"]
distribution = ["dense", "balanced", "sparse"]

# 文本 token 编码
text = clip.tokenize(complexity).to(device)
text1 = clip.tokenize(content).to(device)
text2 = clip.tokenize(color).to(device)
text3 = clip.tokenize(distribution).to(device)

# 文件夹路径
image_folder = ".\Manual_counts\Objects1"  # 修改为你的图片文件夹路径
output_excel =".\Manual_counts\Objects1test.xlsx"

# 结果保存列表
results = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        image_path = os.path.join(image_folder, filename)
        img = Image.open(image_path).convert("RGB")
        image = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # 属性一
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            probs = torch.matmul(image_features, text_features.T).softmax(dim=-1).cpu().numpy().squeeze()
            key = complexity[np.argmax(probs)]

            # 属性二
            image_features1 = model1.encode_image(image)
            text_features1 = model1.encode_text(text1)
            probs1 = torch.matmul(image_features1, text_features1.T).softmax(dim=-1).cpu().numpy().squeeze()
            key1 = content[np.argmax(probs1)]

            # 属性三
            image_features2 = model2.encode_image(image)
            text_features2 = model2.encode_text(text2)
            probs2 = torch.matmul(image_features2, text_features2.T).softmax(dim=-1).cpu().numpy().squeeze()
            key2 = color[np.argmax(probs2)]

            # 属性四
            image_features3 = model3.encode_image(image)
            text_features3 = model3.encode_text(text3)
            probs3 = torch.matmul(image_features3, text_features3.T).softmax(dim=-1).cpu().numpy().squeeze()
            key3 = distribution[np.argmax(probs3)]

        # 模板填充
        templates = [
            f"This is a {key} photo, because it has {key1} content, {key2} color composition, and {key3} distribution.",
            f"From this photo, we can see that it is {key}, with {key1} content, {key2} color composition, and {key3} distribution.",
            f"This photo is {key}, showcasing {key1} content, {key2} color composition, and {key3} distribution.",
            f"It looks {key}, with {key1} content, {key2} color composition, and {key3} distribution.",
            f"In this photo, we can observe a {key} theme, with {key1} content, {key2} color composition, and {key3} composition."
        ]
        description = random.choice(templates)
        print(f"{filename}: {description}")

        # 保存结果
        results.append({
            "photo name": filename,
            "Text Description": description
        })

# 写入 Excel
df = pd.DataFrame(results)
df.to_excel(output_excel, index=False)
print(f"预测结果已保存到 {output_excel}")