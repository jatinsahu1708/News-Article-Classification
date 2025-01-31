import argparse
import os
import sys
# sys.path.append('/content/drive/My Drive/Colab Notebooks/prediction/')
from natsort import natsorted
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from lib import Pre_dataset
from  network import  Seq2seqGRU, SASTANGen
from config import ParseGRU
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt

parse  = ParseGRU()
opt    = parse.args

autoencoder = SASTANGen(opt)
autoencoder.train()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(),
                             lr=opt.learning_rate,
                             weight_decay=1e-5)
transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Grayscale(1),
    transforms.Resize((opt.image_size[0], opt.image_size[1])),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# has all image shape,[data,label]
datat_ = Pre_dataset(opt, opt.trainset, extension=opt.img_extension, transforms=transform)  # b,label?

train_loader = DataLoader(datat_, batch_size=opt.batch_size, shuffle=True)  # if shuffle

#from torch.utils.data.dataloader import default_collate

#def custom_collate_fn(batch):
    # Safely handle None items in the batch
    #non_none_batch = [item for item in batch if item is not None and item[0] is not None]
    #if len(non_none_batch) < len(batch):
        #print(f"Removed {len(batch) - len(non_none_batch)} None items from batch.")
    #return default_collate(non_none_batch)


# Apply the custom collate function when creating the DataLoader
#train_loader = DataLoader(datat_, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

print(f"len_train {len(train_loader)}")



       
    # Print the first data and label to see what they look like
    # Optionally break after the first batch to avoid too much output
# has all image shape,[data,label]
datatest_ = Pre_dataset(opt, opt.testset, extension=opt.img_extension, transforms=transform)  # b,label?,T
test_loader = DataLoader(datatest_, batch_size=1, shuffle=False)  # if shu
#test_loader = train_loader
print(f"len_test {len(test_loader)}")
for i, (data, labels) in enumerate(train_loader):
    x = data.reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
    train = x[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
    y = labels.reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
    label = y[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
    for j in range(0,3) :
        save_image((train[j])/2+0.5,os.path.join(opt.log_folder, "lw_train_real_itr{}_img{}_no{}.tif".format(55,i, j+1)))
        save_image((label[j]/2+0.5),os.path.join(opt.log_folder, "label_real_itr{}_img{}_no{}.tif".format(55,i, j+1)))
        print("image_saved")
        print(f"Batch {i + 1}")
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
    if i==5:
      break;
    
    

 
#autoencoder = SASTANGen(opt)

# Load pre-trained weights (if available)
pretrained_model_path = '/home/jatinsahu/jatin/btp_file-20241006T170707Z-001/btp_file/model(3frame_mse).pth'
if os.path.exists(pretrained_model_path):
    # Load pre-trained model
    print(f"Loading pre-trained model from {pretrained_model_path}")
    autoencoder.load_state_dict(torch.load(pretrained_model_path, weights_only=True))

else:
    print("Pre-trained model not found, starting training from scratch.")

if opt.cuda:
    autoencoder.cuda()

losses = np.zeros(opt.num_epochs)
    
for itr in range(opt.num_epochs):
    #torch.cuda.empty_cache()
    autoencoder.train()
    for data, ydata in train_loader:
        #print(f"data:{data}")
        
        if data is None or ydata is None:
            continue
        if data.size(0) != opt.batch_size:
            break

        x = data.reshape(-1, opt.T, opt.n_channels, opt.image_size[0], opt.image_size[1])
        y = ydata.reshape(-1, opt.T,opt.n_channels, opt.image_size[0], opt.image_size[1])
        
        if opt. cuda:
            x = Variable(x).cuda()
            y = Variable(y).cuda()
        else:
            x = Variable(x)
        

        yhat = autoencoder(x)
        #print(yhat.shape)
        # print(y.shape)
        # print(f"Target labels data type: {y.dtype}")
        # 出力画像（再構成画像）と入力画像の間でlossを計算
        loss = mse_loss(yhat, y.float())
        #print(loss)
        losses[itr] = losses[itr] * (itr / (itr + 1.)) + loss.data * (1. / (itr + 1.))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #tests = y[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
                #recon = yhat[:opt.n_test].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])

                
        

    print('epoch [{}/{}], loss: {:.4f}'.format(itr + 1, opt.num_epochs, loss))
    
    if itr % opt.check_point == 0:
        autoencoder.eval()
        print("hello")
        with torch.no_grad():  # <--- Apply torch.no_grad() here
            count=1
            for data, ydata in test_loader:
                #if data.size(0) != opt.batch_size:
                    #break
                
                x = data.reshape(-1, opt.T, opt.n_channels, opt.image_size[0], opt.image_size[1])
                y = ydata.reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
                print(f"ydata_shape: {ydata.shape}")
                print(f"y_shape: {y.shape}")
                if opt.cuda:
                    x = Variable(x).cuda()
                    y = Variable(y).cuda()
                else:
                    x = Variable(x)

                yhat = autoencoder(x)
                

                tests = y[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
                recon = yhat[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
                print(f"test_shape: {tests.shape}")
                print(f"recon_shape:{recon.shape}")
                
                os.makedirs(opt.log_folder, exist_ok=True)
                for i in range(0,1):
                    print("jatin")
                    save_image((tests[-1]/2+0.5),
                               os.path.join(opt.mse_images, "real_itr{}_count{}.jpg".format(itr,count)))
                    save_image((recon[-1] / 2 + 0.5),
                               os.path.join(opt.mse_images, "recon_itr{}_count{}.jpg".format(itr,count)))
                    print('{0:.6f}'.format(mse_loss(tests, recon)))
                    print("image_saved")
                    count=count+1
                
torch.save(autoencoder.state_dict(), '/home/jatinsahu/jatin/btp_file-20241006T170707Z-001/btp_file/model(3frame_mse).pth')
#     if itr % opt.check_point == 0:
#         autoencoder.eval()
#         for data, ydata in test_loader:

#             if data.size(0) != opt.batch_size:
#                 break

#             x = data.reshape(-1, opt.T, opt.n_channels, opt.image_size[0], opt.image_size[1])
#             y = ydata.reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])

#             if opt.cuda:
#                 x = Variable(x).cuda()
#                 y = Variable(y).cuda()
#             else:
#                 x = Variable(x)

#             yhat = autoencoder(x)

# #        x0 = x[0,0,:,:,:].reshape(-1, opt.n_channels, opt.image_size, opt.image_size)
# #        x1 = x[0,1,:,:,:].reshape(-1, opt.n_channels, opt.image_size, opt.image_size)
# #        x2 = x[0,2,:,:,:].reshape(-1, opt.n_channels, opt.image_size, opt.image_size)
# #        x3 = x[0,3,:,:,:].reshape(-1, opt.n_channels, opt.image_size, opt.image_size)
#         tests = y[:opt.n_test].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
#         recon = yhat[:opt.n_test].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
        
#         print ('{0:.6f}'.format(mse_loss(tests, recon))) 
#         os.makedirs(opt.log_folder, exist_ok=True)
#         for i in range(opt.n_test):
# #            # if itr == 0:
# #            save_image((x0 / 2 + 0.5),
# #                       os.path.join(opt.log_folder + 'generated_videos', "x0_itr{}_no{}.png".format(itr, i)))
# #            save_image((x1 / 2 + 0.5),
# #                       os.path.join(opt.log_folder + 'generated_videos', "x1_itr{}_no{}.png".format(itr, i)))
# #            save_image((x2 / 2 + 0.5),
# #                       os.path.join(opt.log_folder + 'generated_videos', "x2_itr{}_no{}.png".format(itr, i)))
# #            save_image((x3 / 2 + 0.5),
# #                       os.path.join(opt.log_folder + 'generated_videos', "x3_itr{}_no{}.png".format(itr, i)))


#             # os.makedirs(os.path.dirname(os.path.join(opt.log_folder , "real_itr{}_no{}.png".format(itr, i))), exist_ok=True)
#             save_image((tests[i] / 2 + 0.5),
#                        os.path.join(opt.log_folder , "real_itr{}_no{}.png".format(itr, i)))
#             save_image((recon[i] / 2 + 0.5),
#                        os.path.join(opt.log_folder, "recon_itr{}_no{}.png".format(itr, i)))



#             print("image_saved")
#             # torch.save(autoencoder.state_dict(), os.path.join('./weights', 'G_itr{:04d}.pth'.format(itr+1)))

