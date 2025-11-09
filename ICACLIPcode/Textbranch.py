from torch import nn
import torch
import numpy as np

class IC_CLIP(nn.Module):
    def __init__(self, clip_model, device):
        super(IC_CLIP,self).__init__()
        self.clip_model=clip_model
        self.loss_fn=contrastive_loss(device)
        self.logit_scale = nn.Parameter(torch.ones([])* np.log(1/0.07))
        self.device = device
        for param in self.clip_model.parameters():
            param.requires_grad= True


    def forward(self,image, texts):

        image = image.to(self.device)
        texts=texts.to(self.device)
        # print("image:", image.shape)
        # print("texts:", texts.shape)
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(texts)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        loss=self.loss_fn(image_features,text_features)

        return loss




class contrastive_loss(nn.Module):
    def __init__(self,device):
        super(contrastive_loss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([])* np.log(1/0.07))
        #nn.CrossEntropyLoss()包括了 softmax 操作和对数损失的计算
        self.criterion=nn.CrossEntropyLoss()
        self.device = device

    def forward(self,image_features,text_features):

        batch_size = image_features.shape[0]
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-10)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-10)

        # 计算余弦相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        labels = torch.arange(batch_size).to(self.device)

        log_probs_image=self.criterion(logits_per_image,labels)
        log_probs_text=self.criterion(logits_per_text,labels)
        loss=(log_probs_image+ log_probs_text)/2.0

        return loss


