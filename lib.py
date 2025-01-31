import os

import torch
import cv2
import numpy as np
from natsort import natsorted
import glob
from PIL import Image
from config import ParseGRU
parse  = ParseGRU()
opt    = parse.args

class Pre_dataset(torch.utils.data.Dataset):
    def __init__(self, opt, root_folder, extension="jpg", transforms=None):
        self.videos = []  # This will hold all input sequences from all subfolders
        self.futures = []  # This will hold all label sequences from all subfolders
        self.T = opt.T
        self.transforms = transforms
        
        # Get a list of all subfolders inside the root folder
        subfolders = natsorted([f.path for f in os.scandir(root_folder) if f.is_dir()])
        
        # Loop through each subfolder
        i=0
        count=0;
        for subfolder in subfolders:
        
             # Now, look for images inside the "images" folder in each subfolder
            image_folder = os.path.join(subfolder, 'images_masked')
            print(f"Checking images_masked folder: {image_folder}")
            
            # Load all the images in the images subfolder
            frame_image = natsorted(glob.glob(image_folder + '/*.' + extension))
            print(f"Found {len(frame_image)} images in {image_folder}")
            # Load all the images in the subfolder
            #frame_list = natsorted(glob.glob(subfolder + '/*.' + extension))
            #image_count = len(frame_image)
        
            # Print the number of images in the current subfolder
            #print(f"{subfolder}: {image_count} images")
            
            # Update the total image count
            #total_images += image_count
            #subfolder_count += 1
            if len(frame_image)>=24:
              count=count+1
            
            if len(frame_image) == 0:
                print(f"Skipping {subfolder}: No images found.")
                continue  # Skip this folder if it doesn't have any images
            
            # Use all images except the last one for input
            frame_list = frame_image[:-1]  # Exclude the last image
            
            # Use all images except the first one for labels
            label_list = frame_image[1:]  # Exclude the first image
            
            
            
            ###########################################
            #frame_list = natsorted(glob.glob(video_folder + '/*.' + extension))
            #folder_path=video_folder.split('/')[-1]
            #label_list=[]
            #if folder_path=="images":
              #label_list = natsorted(glob.glob(video_folder.replace("images","masks") + '/*.' + "tif"))
              #frame_list=frame_list[0:1000]
              #label_list=label_list[0:1000]
            #elif folder_path=="image":
              #label_list = natsorted(glob.glob(video_folder.replace("image","mask") + '/*.' + "tif"))
            
            print(len(label_list))
            print(len(frame_list))
            for j in range(len(frame_list) - opt.T+1):
                # print(frames[0])
                # print(frame_list[j + opt.T+1])
                #print(f"j: {j}, opt.T: {opt.T}, len(label_list): {len(label_list)}")
                #print(frame_list[j + opt.T])
                

                video = [frame_list[j:j + opt.T][k] for k in range(opt.T)]
                #video = frame_list[j:j + opt.T]
                #future = label_list[j:j + opt.T]
                #print(video[0])
                #print(future[0])
                #print("hello")
                future = [label_list[j:j + opt.T][k] for k in range(opt.T)]
    
                # print(video)rameramet
                self.videos.append(video)
                self.futures.append(future)
            
            # Ensure there are exactly 25 images
                 
            print(f"len_video_list {len(self.videos)}")
            if(i==2):
              break
            # Append the input and label images to the dataset lists
            #self.videos.append(input_images)
            #self.labels.append(label_images)
        print(f"images with >=24months: {count}")  
    def __len__(self):
        return len(self.videos)  # Number of subfolders (or sequences) processed
        
    print("hello my names is jatin sahu")    
    
    def __getitem__(self, idx):
        video_list = self.videos[idx]
        future_list = self.futures[idx]
        
        video_frames = []
        futures = []
        
        
        
        
      
         # Process video frames
        for i in range(len(video_list)):
            #print(video_list.shape)
            try:
                video_frame = cv2.imread(video_list[i], cv2.IMREAD_UNCHANGED)
                if video_frame is None:
                    raise ValueError(f"Video frame at {video_list[i]} could not be loaded.")
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB     
                #print(video_frame.shape)
                
                video_frame = self.transforms(video_frame)  # Apply your transformations
                video_frames.append(video_frame.numpy())  # Append as numpy array
            except Exception as e:
                print(f"Error loading video frame {video_list[i]}: {e}")
                continue  # Skip bad frames if they cannot be loaded
    
        # Ensure there are valid video frames
        if len(video_frames) == 0:
            raise ValueError(f"No valid video frames found for index {idx}.")
        
        video = np.array(video_frames)  # Convert list of video frames to numpy array
    
        # Process future frames
        for i in range(len(future_list)):
            try:
                future_frame = cv2.imread(future_list[i], cv2.IMREAD_UNCHANGED)
                if future_frame is None:
                    raise ValueError(f"Future frame at {future_list[i]} could not be loaded.")
                
                future_frame = cv2.cvtColor(future_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                future_frame = self.transforms(future_frame)  # Apply your transformations
                futures.append(future_frame.numpy())  # Append as numpy array
            except Exception as e:
                print(f"Error loading future frame {future_list[i]}: {e}")
                continue  # Skip bad frames if they cannot be loaded
    
        # Ensure there are valid future frames
        if len(futures) == 0:
            raise ValueError(f"No valid future frames found for index {idx}.")
        
        label = np.array(futures)  # Convert list of future frames to numpy array
    
        return torch.from_numpy(video), torch.from_numpy(label)

    
   
    
#    def __getitem__(self, idx):
#         video_list = self.videos[idx]
#         future_list = self.futures[idx]
        
#         # Initialize empty lists for video frames and future frames
#         video_frames = []
#         #futures = np.empty((len(future_list),3, 3, opt.image_size[0], opt.image_size[1]), dtype=np.float32)  # Correct shape (channels, height, width)
#         futures=[]
        
#         # Process video frames
#         for i in range(len(video_list)):
#             # Read and convert video frame to RGB
#             video_frame = cv2.imread(video_list[i], cv2.IMREAD_UNCHANGED)
#             if video_frame is not None:
#                 video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#                 video_frame = self.transforms(video_frame)  # Apply your transformations
#                 video_frames.append(video_frame.numpy())  # Append as numpy array
        
#         video = np.array(video_frames)  # Convert list of video frames to numpy array
    
#         # Process future frames
#         for i in range(len(future_list)):
#             future = cv2.imread(future_list[i], cv2.IMREAD_UNCHANGED)  # Load the image with PIL
#             if future is not None:
#                 future = cv2.cvtColor(future, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#                 future = self.transforms(future)  # Apply your transformations
#                 futures.append(future.numpy())  # Append as numpy array
#         label=np.array(futures)


    