import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import BertModel, BertTokenizer,AutoModel,AutoImageProcessor
import os
from transformers import CLIPTextModel, CLIPTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "E:\exproject\clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_path)


class ICSC3Dataset(Dataset):
    def __init__(self, image_folder_path, excel_file_path,tokenizer,transform=None):
        """
        初始化数据集
        Args:
            image_folder_path (str): 存放图像数据的文件夹路径
            excel_file_path (str): 包含图像名称和文本标注的Excel文件路径
            transform (callable, optional): 对图像进行的转换操作，例如裁剪、归一化等. Defaults to None.
        """
        self.image_folder_path = image_folder_path
        self.df = pd.read_excel(excel_file_path)
        self.transform = transform
        self.tokenizer = tokenizer
        self.device = device
        # self.df = pd.read_excel(excel_file_path, skiprows=1)  # 跳过第一行
        self.df.columns = self.df.columns.str.strip()  # 去掉列名前后的空格
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
       
        image_name = self.df.at[idx, 'photo name']
        text = self.df.at[idx, 'text'] # Adjust the text description yourself
        score = self.df.at[idx, 'score']
        image_path = os.path.join(self.image_folder_path, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        texts = self.tokenizer(
            text,
            max_length=16,  # 显式限制长度
            padding='max_length',  # 不足77则填充
            truncation=True,  # 超过77则截断
            return_tensors='pt'
        )
        texts = texts['input_ids'].squeeze()

        return image, texts ,score

# 定义图像变换（这里可以根据实际需求调整变换内容，比如归一化的参数等）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



image_folder_path = r".\ICA4000train"  # 替换为实际的图像文件夹路径
excel_file_path = r".\ICA4000train.xlsx"


ICSC3dataset = ICSC3Dataset(image_folder_path=image_folder_path, excel_file_path=excel_file_path,tokenizer = tokenizer, transform=transform)
