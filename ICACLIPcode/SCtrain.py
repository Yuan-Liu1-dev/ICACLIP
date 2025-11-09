import torch.optim as optim
import torch
from CLIP.SCwithtext import SCtext,SCtextcross,imagetext,SwinSvit,efficientnetb_6,SwinSvittext,Resnet18,SCtextZjiaquan,Resnet50,Resnet,Resnet101,efficientnetb_7,efficientnetb_5,SCcross,efficientnetb_3,Textcross,efficientnetb_4,vit32,vit16
from torch import nn
from datesets.datasetSC import train_dataset
from datesets.datasetSCn import ICSC3dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
from transformers import set_seed
from collections import defaultdict



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SCtextcross(device).to(device)
# model = imagetext(device).to(device)
# model = SCtextncross(yu_model,device).to(device) 
# model = SCtextjiaquan(yu_model,device).to(device)
# model = SCtextZjiaquan(yu_model,device).to(device)
# model = CLIPForRegression(yu_model,device).to(device)
# model = Resnet50(device).to(device)
# model = Resnet18(device).to(device)
# model = Resnet101(device).to(device)
# model = efficientnetb_7(device).to(device)
# model = efficientnetb_5(device).to(device)
# model = efficientnetb_3(device).to(device)
# model = efficientnetb_4(device).to(device)
# model = efficientnetb_6(device).to(device)
# model = vit32(device).to(device)
# model = vit16(device).to(device)
# model = SwinSvit(device).to(device)
# model = SwinSvittext(device).to(device)
# model = SCcross(device).to(device)
#model = Textcross(device).to(device)
# model = Resnet(yu_model,device).to(device)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), weight_decay=0.001, momentum=0.9, lr=0.001)#IC9600设置

train_loader = DataLoader(ICSC3dataset, batch_size=16, shuffle=True)
best_model = './save_model/NewIC15.pth'
best_train_loss = 500.0
epochs =10

for epoch in range(epochs):

    model.train()
    total_loss = 0.0
    train_bar = tqdm(train_loader)

    for batch_idx,(images,texts,labels) in enumerate(train_bar):
    # for batch_idx,(images,labels) in enumerate(train_bar):
    #for batch_idx, (texts, labels) in enumerate(train_bar):
        images= images.to(device)
        labels=labels.to(device)
        labels=labels.float()
        texts = texts.to(device)
        optimizer.zero_grad()

        outputs = model(images,texts).squeeze()  # Squeeze to match label dimensions
        # outputs = model(images).squeeze()
        #outputs = model(texts).squeeze() 

        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()
        print(f'Training: Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_bar)}], Loss: {train_loss.item():.5f}')


    average_loss = total_loss/ len(train_bar)
    print(f'average_loss: Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss:.5f}')
    
    if average_loss < best_train_loss:
            best_train_loss = average_loss
            best_epoch = epoch + 1           
            print(f"New Best Average Loss: {best_train_loss:.5f}")
            print("save best model")
            torch.save(model, best_model)

print(f'训练结束最佳损失: {best_train_loss:.5f} at epoch {best_epoch}')


