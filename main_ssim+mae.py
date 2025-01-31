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

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, lambda_mae=0.01):
        super(CustomLoss, self).__init__()
        self.mae_loss = nn.L1Loss()  # Predefined MAE loss
        self.lambda_mae = lambda_mae  # Lambda weight for MAE

    def ssim(self, img1, img2, C1=1e-4, C2=9e-4):
        device = img1.device  # Get the device of img1 (either CPU or GPU)
        channels = img1.size(1)  # Number of channels
        mu1 = F.conv2d(img1, self.create_gaussian_filter(device=device, channels=channels), padding=1, groups=channels)
        mu2 = F.conv2d(img2, self.create_gaussian_filter(device=device, channels=channels), padding=1, groups=channels)
    
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
    
        sigma1_sq = F.conv2d(img1 * img1, self.create_gaussian_filter(device=device, channels=channels), padding=1, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.create_gaussian_filter(device=device, channels=channels), padding=1, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.create_gaussian_filter(device=device, channels=channels), padding=1, groups=channels) - mu1_mu2
    
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()  # Return the mean SSIM

    def create_gaussian_filter(self, kernel_size=5, sigma=1.5, device='cpu', channels=1):
        """Create a Gaussian kernel and expand it for each input channel."""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords -= (kernel_size - 1) / 2
    
        # Create a 2D Gaussian kernel
        g = torch.exp(-0.5 * (coords.unsqueeze(0) ** 2 + coords.unsqueeze(1) ** 2) / (sigma ** 2))
    
        # Normalize the kernel
        g = g / g.sum()
    
        # Expand the kernel to match the number of input channels
        g = g.view(1, 1, kernel_size, kernel_size).to(torch.float32)
        g = g.repeat(channels, 1, 1, 1)  # Repeat for each channel
    
        return g






    def forward(self, img1, img2):
        # Debugging: Print the shapes of img1 and img2
        print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")
        
        # Check the number of dimensions before reshaping
        if img1.dim() == 5:  # [Batch, Channels, Frames, Height, Width]
            # Collapse frames into the batch dimension
            img1 = img1.reshape(-1, img1.size(1), img1.size(3), img1.size(4))
            img2 = img2.reshape(-1, img2.size(1), img2.size(3), img2.size(4))
        elif img1.dim() == 4:  # [Batch, Channels, Height, Width] (No frames)
            # No reshaping needed if there are no frames
            pass
        else:
            raise ValueError(f"Unexpected tensor dimension: {img1.dim()}")
    
        # Calculate SSIM and MAE
        ssim = 1 - self.ssim(img1, img2)  # Invert SSIM to use as loss
        mae = self.mae_loss(img1, img2)
    
        # Combine the losses
        total_loss = ssim + self.lambda_mae * mae  # Combine SSIM and MAE with lambda weight
        return total_loss.mean()  # Return as scalar




        

# Example usage
loss_fn = CustomLoss(lambda_mae=0.01)

parse  = ParseGRU()
opt    = parse.args

autoencoder = SASTANGen(opt)
autoencoder.train()
#mse_loss = nn.MSELoss()
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
#for i, (data, labels) in enumerate(train_loader):
#    x = data.reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
#    train = x[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
#    y = labels.reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
#    label = y[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
#    for j in range(0,3) :
#      save_image((train[j])/2+0.5,os.path.join(opt.log_folder, "l_train_real_itr{}_no{}.tif".format(55, j+1)))
#     save_image((label[j]/2+0.5),os.path.join(opt.log_folder, "label_real_itr{}_no{}.tif".format(55, j+1)))
#      print("image_saved")
#      print(f"Batch {i + 1}")
#      print(f"Data shape: {data.shape}")
 #     print(f"Labels shape: {labels.shape}")
 #    if i==0:
  #    break;
    
for i, (data, labels) in enumerate(train_loader):
    x = data.reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
    train = x[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
    y = labels.reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
    label = y[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
    for j in range(0,3) :
        save_image((train[j])/2+0.5,os.path.join(opt.log_folder, "lw_train_real_itr{}_no{}.tif".format(55, j+1)))
        save_image((label[j]/2+0.5),os.path.join(opt.log_folder, "label_real_itr{}_no{}.tif".format(55, j+1)))
        print("image_saved")
        print(f"Batch {i + 1}")
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
    if i==0:
      break;    

 
#autoencoder = SASTANGen(opt)

# Load pre-trained weights (if available)
pretrained_model_path = '/home/jatinsahu/jatin/btp_file-20241006T170707Z-001/btp_file/model(3frame_mae+ssim).pth'
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
        # ????(?????)????????loss???
        loss = loss_fn(yhat, y.float())
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
                print("y_shape: {y.shape}")

                if opt.cuda:
                    x = Variable(x).cuda()
                    y = Variable(y).cuda()
                else:
                    x = Variable(x)

                yhat = autoencoder(x)

                tests = y[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
                recon = yhat[:3].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
                print("test_shape: {tests.shape}")
                print("recon_shape:{recon_shape}")

                
                os.makedirs(opt.log_folder, exist_ok=True)
                for i in range(0,1):
                    #print("jatin")
                    save_image((tests[-1]/2+0.5),
                               os.path.join(opt.mae_ssim_images, "real_itr{}_count{}.jpg".format(itr,count)))
                    save_image((recon[-1] / 2 + 0.5),
                               os.path.join(opt.mae_ssim_images, "recon_itr{}_count{}.jpg".format(itr,count)))
                    print('{0:.6f}'.format(loss_fn(tests, recon)))
                    print("image_saved")
                    count=count+1
                
torch.save(autoencoder.state_dict(), '/home/jatinsahu/jatin/btp_file-20241006T170707Z-001/btp_file/model(3frame_mae+ssim).pth')
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

