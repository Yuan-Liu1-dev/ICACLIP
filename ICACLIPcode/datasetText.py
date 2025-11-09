import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import BertTokenizer,CLIPTokenizer
import os



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "E:\exproject\clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_path)
class CLIPDataset(Dataset):
    def __init__(self, image_folder_path, excel_file_path, transform=None):
       
        self.image_folder_path = image_folder_path
        self.df = pd.read_excel(excel_file_path)
        self.transform = transform
        self.tokenizer = tokenizer
        self.df.columns = self.df.columns.str.strip()  # 去掉列名前后的空格

    def __len__(self):
        
        return len(self.df)

    def __getitem__(self, idx):
       
        
        image_name = self.df.at[idx, 'photo name']
        text = self.df.at[idx, 'complexity']
        # text = self.df.at[idx, 'content']
        # text = self.df.at[idx, 'color']
        # text = self.df.at[idx, 'distribution']
        image_path = os.path.join(self.image_folder_path, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        encoded_text = self.tokenizer(
            text,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts = encoded_text['input_ids'].squeeze()
        # print(image.shape)
        # print(texts)
        return image, texts # 这里假设文本标注可以直接转换为张量，可能实际中需要进一步处理文本编码等

# 示例用法
# 定义图像变换（这里可以根据实际需求调整变换内容，比如归一化的参数等）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_folder_path = r".\ICA4000train"  # 替换为实际的图像文件夹路径
excel_file_path = r".\ICA4000train.xlsx"  # 替换为实际的Excel文件路径

dataset = CLIPDataset(image_folder_path=image_folder_path, excel_file_path=excel_file_path, transform=transform)