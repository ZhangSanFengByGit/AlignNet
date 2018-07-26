import os
import time

import torch
from PIL import Image
from torch.optim import Adam

from torchvision import transforms

from AlignNet import AlignNet




#arguments definition
#device = torch.device("cuda" if torch.cuda.is_available == True else "cpu")
device = torch.device("cuda")
base_path = "H:/pedestrian_RGBT/kaist-rgbt/images/"
log_interval = 1
checkpoint_interval = 5
learning_rate = 0.001
checkpoint_model_dir = "H:/model_checkPoint_save_version2/"
save_model_dir = "H:/model_finish/"
epochs = 10



#transform method
transform = transforms.Compose([
        transforms.Resize(550),
        transforms.RandomResizedCrop(512),
        transforms.RandomRotation([-15,15]),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform2 = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor()
        ])

#variable definition
alignNet = AlignNet().to(device)
optimizer = Adam(alignNet.parameters(),learning_rate)
mse_loss = torch.nn.MSELoss()
#best_loss = infinite_great

#temp Area~~~~~~~~
alignNet.load_state_dict(torch.load('H:\\model_checkPoint_save\\ckpt_epoch_0_Sequence_id_5.plk'))




for e in range(epochs):
    alignNet.train()
    turn = 0

    SetNum = len(os.listdir(base_path))
    for i in range(SetNum):
        SequenceNum = len(os.listdir(base_path+"set"+str(i+1)+"/"))
        
        for j in range(SequenceNum):
            Sequence_path = base_path+"set"+str(i+1)+"/"+"V00"+str(j)+"/"
            
            rgb_folder = Sequence_path + "visible/"
            thermal_folder = Sequence_path + "lwir/"
            imgs = os.listdir(rgb_folder)
            imgNum = len(imgs) 
            
            for imgNo in range(imgNum):
                optimizer.zero_grad()
                encodeNo = list(str(imgNo+100000))
                encodeNo.pop(0)
                encodeNo = "".join(encodeNo)
                
                img1 = Image.open(rgb_folder+"I"+encodeNo+".jpg")
                img2 = Image.open(thermal_folder+"I"+encodeNo+".jpg")
                img1 = transform2(img1).unsqueeze(0)
                img2_trans = transform(img2).unsqueeze(0)
                img2 = transform2(img2).unsqueeze(0).to(device)

                img2_masked = alignNet(img1.to(device),img2_trans.to(device))

                loss = mse_loss(img2_masked,img2)
                loss.backward()
                optimizer.step()

                log_loss = loss.item()
            turn = turn + 1

            if turn % log_interval == 0:
                mesg = "{}\tEpoch {}  Turn{}:  set {}  sequence {}  loss: {:.6f}".format(
                      time.ctime(), e + 1, turn, i, j, log_loss)
                print(mesg)


            if checkpoint_model_dir is not None and turn % checkpoint_interval == 0:
                alignNet.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_Sequence_id_" + str(turn) + ".plk"
                ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
                torch.save(alignNet.state_dict(), ckpt_model_path)
                alignNet.to(device).train()

alignNet.eval().cpu()
save_model_filename = "Finish_epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" +".model"
save_model_path = os.path.join(save_model_dir, save_model_filename)
torch.save(alignNet.state_dict(), save_model_path)

print("\nDone, trained model saved at", save_model_path)
    








'''
use torch.CUDA
optmizer = optim(AlignNet.parameters())
current_params = AlignNet.static_params()
best_loss = infinite_great


for epoch in range(25):
    for set in(6):
        for serials in(50):
            for i in (imageNum):
                img1 = get(rgb)
                img2 = get(thermal)
                img2_t = transform(img2)
                img2_masked = AlignNet(img1,img2_t)
                loss = MSE(img2_masked,img2_t)
                loss.backward()
                optmizer.step()

                if loss < best_loss:
                    best_loss = loss
                    current_params = AlignNet.static_params()

save(current_params)
'''




