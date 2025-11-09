import clip
import torch
from torch.utils.data import Dataset,DataLoader
from datesets.dateset_IC3 import dataset
from datesets.dataset_ICn import dataset
from CLIP.IC_CLIPmodel import IC_CLIP
from CLIP.ICmodeln import CLIPModel,VisionTransformer
import torch.optim as optim
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name="ViT-B/32"
yu_model,_=clip.load(model_name,device=device,jit=False,download_root=r"E:\exproject\CLIP-main")#这个没写 download_root
yu_model.float()
model = IC_CLIP(yu_model,device).to(device)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True,drop_last=False,num_workers=0)#一般设置为0

# default_dtype = torch.cuda.FloatTensor().dtype
# print(f"GPU默认数据类型: {default_dtype}")

def main():

    epochs =10
    best_train_loss = 1000.0
    best_model = './save_model3\ICmodel\IC2.pth'
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range (epochs):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(dataloader)
        # print(f"Epoch: {epoch + 1}")
        # for batch_idx, (image, texts) in tqdm(enumerate(train_loader)):
        # for batch_idx, (image, texts) in enumerate(train_bar):
        for batch_idx, (image,texts) in enumerate(train_bar):
           image= image.to(device)
           texts=texts.to(device)

           train_loss = model(image, texts)
           optimizer.zero_grad()
           train_loss.backward()
           optimizer.step()
           total_loss += train_loss.item()
           print(f'Training: Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {train_loss.item():.4f}')
        
        average_loss = total_loss / len(train_bar)
        print(f'average_loss: Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss:.4f}')
        if average_loss < best_train_loss:
            best_train_loss = average_loss
            print("保存最好的模型")
            torch.save(model, best_model)
    print(f'训练结束最佳损失: {best_train_loss:.4f}')
if __name__ == '__main__':
    main()
