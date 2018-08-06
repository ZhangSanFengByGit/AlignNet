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
checkpoint_model_dir = "H:/model_checkPoint_save_version4/"
save_model_dir = "H:/model_finish/"
epochs = 10
loss_weight = 1e5
batch_size = 4


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

imgR = list()
imgT = list()
imgT_trans = list()
imgR_batch = torch.zeros(batch_size,3,512,512)
imgT_batch = torch.zeros(batch_size,3,512,512)
imgT_trans_batch = torch.zeros(batch_size,3,512,512)

flag = 0
global log_check
global log_loss_sum

#best_loss = infinite_great

#temp Area~~~~~~~~
#alignNet.load_state_dict(torch.load('H:\\model_checkPoint_save_version3\\ckpt_epoch_3_Sequence_id_40.plk'))




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
            
            
            flag = 0
            imgR = []
            imgT = []
            imgT_trans = []
            
            log_check = 0
            log_loss_sum = 0
            
            for imgNo in range(imgNum):
                flag = flag + 1
                
                encodeNo = list(str(imgNo+100000))
                encodeNo.pop(0)
                encodeNo = "".join(encodeNo)
                
                img1 = Image.open(rgb_folder+"I"+encodeNo+".jpg")
                img2 = Image.open(thermal_folder+"I"+encodeNo+".jpg")
                img1 = transform2(img1)#.unsqueeze(0)
                img2_trans = transform(img2)#.unsqueeze(0)
                img2 = transform2(img2)#.unsqueeze(0).to(device)
                
                imgR.append(img1)
                imgT_trans.append(img2_trans)
                imgT.append(img2)
                
                if flag == batch_size:
                    
                    flag = 0
                    
                    for p in range (batch_size):
                        imgR_batch[p] = imgR[p]
                        imgT_batch[p] = imgT[p]
                        imgT_trans_batch[p] = imgT_trans[p]
                        
                    optimizer.zero_grad()
                    #img2_masked = alignNet(img1.to(device),img2_trans.to(device))
                    imgT_batch = imgT_batch.to(device)
                    imgR_batch = imgR_batch.to(device)
                    imgT_trans_batch = imgT_trans_batch.to(device)
                    
                    img_masked = alignNet(imgR_batch , imgT_trans_batch)


                    loss = mse_loss(img_masked , imgT_batch)/batch_size
                    loss = loss * loss_weight
                    loss.backward()
                    optimizer.step()

                    log_loss = loss.item()
                    log_check = log_check + 1
                    log_loss_sum = log_loss_sum + log_loss
                    
                    imgR = []
                    imgT = []
                    imgT_trans = []
            
            
            turn = turn + 1
            log_loss_sum = log_loss_sum/log_check
            if turn % log_interval == 0:
                mesg = "{}\tEpoch {}  Turn{}:  set {}  sequence {}  loss: {:.6f}".format(
                      time.ctime(), e + 1, turn, i, j, log_loss_sum)
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




