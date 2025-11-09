from torch import nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel, BertTokenizer,ResNetConfig, ResNetModel,AutoImageProcessor, AutoModel,ViTModel,SwinModel
import torchvision
from transformers import CLIPTextModel, CLIPTokenizer,CLIPProcessor, CLIPVisionModel
from torchvision import models


model_path = "E:\exproject\clip-vit-base-patch32"
text_encoder = CLIPTextModel.from_pretrained(model_path)
image_model = CLIPVisionModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)




# 交叉注意力融合
class CrossAttentionLayer(nn.Module):
    def __init__(self):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(512,8, batch_first=True)

    def forward(self, query, key, value):
        # query 是图像特征，key 和 value 是文本特征
        # 这里的query是来自图像的特征，key和value是来自文本的特征
        output, _ = self.attention(query, key, value)
        return output

class SCtextcross(nn.Module):
    def __init__(self, device):
        super(SCtextcross, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder =image_model
        self.device = device

        self.image512=nn.Linear(768,512)


        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),  # 输出层
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.cross_attention = CrossAttentionLayer()

    def forward(self, image, texts):
  
        image = image.to(self.device)
        texts = texts.to(self.device)

        image_output = self.image_encoder(image)
        image_hidden = image_output.last_hidden_state # [16, 50, 768]
        image_features=self.image512(image_hidden)# [16, 50, 512]

        text_output = self.text_encoder(texts)
        text_features = text_output.last_hidden_state  # [16, 16, 512],第一个是batch
        combined_features= self.cross_attention(text_features, image_features,image_features) # [16, 16, 512]

        mean_output = torch.mean(combined_features, dim=1)    
        output = self.regression_head(mean_output)

        return output
