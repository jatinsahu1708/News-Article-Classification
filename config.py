import argparse

from numpy import True_

class ParseGRU():
    def __init__(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument('--trainset', default="/home/jatinsahu/jatin/archive (2)/SN7_buildings_train/train")
        parser.add_argument('--valset', default="/home/jatinsahu/jatin/archive (2)/test_1/test", help='log directory')
        parser.add_argument('--testset', default="/home/jatinsahu/jatin/archive (2)/test_1/test", help='log directory')
        parser.add_argument('--img_extension', default='jpg', help='log directory')
        parser.add_argument('--num_itrs', type=int, default=10000)
        parser.add_argument('--T', type=int, default=3, help='checkpoint epoch')
        parser.add_argument('--num_layers', type=int, default=3, help='checkpoint epoch')
        parser.add_argument('--z_dim', type=int, default=1000, help='weight decay')
        parser.add_argument('--log_folder', default='/home/jatinsahu/jatin', help='log directory')
        #i am adding
        parser.add_argument('--mse_images', default='/home/jatinsahu/jatin/mse_images', help='log directory')
        parser.add_argument('--mae_ssim_images', default='/home/jatinsahu/jatin/mae_ssim_images', help='log directory')
        parser.add_argument('--model_path', type=str, default="/home/jatinsahu/jatin/btp_file-20241006T170707Z-001/btp_file/model(3frame_mse).pth", help='Path to the pretrained model file')

        #
        parser.add_argument('--batch_size', type=int,default=2)#../DATASET/UCSD/train/
        parser.add_argument('--test_batch', type=int, default=1)  # ../DATASET/UCSD/train6
        parser.add_argument('--sample_batch', type=int, default=1)  # ../DATASET/UCSD/train6
        parser.add_argument('--warmup', type=bool, default=False)  # ../DATASET/UCSD/train6
        parser.add_argument('--warmup_epochs', type=int, default=20)  # ../DATASET/UCSD/train6
        parser.add_argument('--image_size', default=(512,512))
        parser.add_argument('--conv', default=2*2*2)  # img/16
        parser.add_argument('--check_point', type=int, default=25, help='apply SpectralNorm')#SNシますか?
        parser.add_argument('--n_test', type=int, default=1, help='apply Self-atten')  # Attnシますか?
        parser.add_argument('--n_channels', type=int, default=3, help='apply Self-atten')  # Attnシますか?
        parser.add_argument('--num_epochs', type=int, default=51, help='apply Self-atten')
        parser.add_argument('--gru_dim', type=int, default=128, help='weight decay')
        parser.add_argument('--lstm_dim', type=int, default=128, help='weight decay')
        parser.add_argument('--ngru', type=int, default=100, help='dimension of latent variable')#512,128,32
        parser.add_argument('--alpha', type=int, default=25*4, help='weight decay')  # 1e-2,10
        parser.add_argument('--beta', type=float, default=2, help='weight decay')
        parser.add_argument('--lamda', type=int, default=10, help='weight decay')
        parser.add_argument('--try_', type=int, default=3, help='weight decay')
        parser.add_argument('--cuda', type=bool, default=True, help='weight decay')
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='coefficient of L_prior')  # 1e-4,1e-3
        parser.add_argument('--learning_rate_d', type=float, default=2e-5, help='coefficient of L_prior')  # 4e-4,8e-3
        parser.add_argument('--n_class', type=int, default=3, help='apply Self-atten')  # Attnシますか?

        self.args = parser.parse_args()


