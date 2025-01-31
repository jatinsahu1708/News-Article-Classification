import os
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Import model, dataset, and config
from network import SASTANGen
from lib import Pre_dataset
from config import ParseGRU
parse  = ParseGRU()
opt    = parse.args

class SSIMEvaluator:
    def __init__(self, model, opt, device='cpu'):
        self.model = model
        self.opt = opt
        self.device = device

    def create_gaussian_filter(self, kernel_size=5, sigma=1.5, device='cpu', channels=3):
        """Create a Gaussian kernel and expand it for each input channel (c, h, w)."""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords -= (kernel_size - 1) / 2

        # Create a 2D Gaussian kernel
        g = torch.exp(-0.5 * (coords.unsqueeze(0) ** 2 + coords.unsqueeze(1) ** 2) / (sigma ** 2))

        # Normalize the kernel
        g = g / g.sum()

        # Expand to match the number of input channels
        g = g.view(1, 1, kernel_size, kernel_size).to(torch.float32)
        g = g.repeat(channels, 1, 1, 1)  # Repeat for each channel

        return g

    def calculate_ssim(self, img1, img2, C1=1e-4, C2=9e-4):
        device = img1.device  # Get device of img1 (either CPU or GPU)
        channels = img1.size(0)  # Get the number of channels in img1 (e.g., 3 for RGB)

        # Reshape inputs to (1, c, h, w) for conv2d compatibility
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        # Create Gaussian filter for the number of channels
        gaussian_filter = self.create_gaussian_filter(device=device, channels=channels)

        # Apply conv2d with groups set to number of channels
        mu1 = F.conv2d(img1, gaussian_filter, padding=2, groups=channels)
        mu2 = F.conv2d(img2, gaussian_filter, padding=2, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, gaussian_filter, padding=2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, gaussian_filter, padding=2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, gaussian_filter, padding=2, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return the mean SSIM value
        return ssim_map.mean()





    def evaluate_model(self, test_loader):
        ssim_scores = []
        
        with torch.no_grad():
            for data, ydata in test_loader:
                x = data.reshape(-1, self.opt.T, self.opt.n_channels, self.opt.image_size[0], self.opt.image_size[1])
                y = ydata.reshape(-1, self.opt.n_channels, self.opt.image_size[0], self.opt.image_size[1])
                x = Variable(x).to(self.device)
                y = Variable(y).to(self.device)
                tests = y.reshape(-1, self.opt.n_channels, self.opt.image_size[0], self.opt.image_size[1])
                yhat = self.model(x)
    
                # Assuming yhat should also be reshaped to match channels
                recon = yhat[:3].reshape(-1, self.opt.n_channels, self.opt.image_size[0], self.opt.image_size[1])
                
                # Make sure both tests and recon have the same number of channels
                for i in range(tests.size(0)):
                    score = self.calculate_ssim(tests[i], recon[i])
                    ssim_scores.append(score)
    
        avg_ssim = torch.mean(torch.tensor(ssim_scores)).item()  # Average SSIM
        return avg_ssim




def load_model(model_path):
    # Load your pretrained model
    model = SASTANGen(opt)  # Initialize model with config options
    
    # Use weights_only=True to avoid loading arbitrary objects
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained model
    model = load_model(opt.model_path)  # Load model from config
    model = model.to(device)

    # Load test data
    transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Grayscale(1),
    transforms.Resize((opt.image_size[0], opt.image_size[1])),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
    
    datatest_ = Pre_dataset(opt, opt.testset, extension=opt.img_extension, transforms=transform)  # b,label?,T
    test_loader = DataLoader(datatest_, batch_size=opt.test_batch, shuffle=False)  # Batch size from config

    # Instantiate the SSIM Evaluator class
    evaluator = SSIMEvaluator(model, opt, device=device)

    # Evaluate model
    avg_ssim = evaluator.evaluate_model(test_loader)
    print(f'Average SSIM: {avg_ssim}')

if __name__ == '__main__':
    main()
