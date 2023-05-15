import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.Network import Network
from utils.data_val import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size') # 
parser.add_argument('--pth_path', type=str, default='/data0/hcm/hjwNet/snapshot/model/Net_epoch_best.pth') 
parser.add_argument('--test_dataset_path', type=str, default='/data0/hcm/dataset/COD/TestDataset') 
opt = parser.parse_args()

for _data_name in ['CAMO', 'COD10K', 'CHAMELEON', 'NC4K']:
    data_path = opt.test_dataset_path+'/{}/'.format(_data_name)
    save_path = './res/{}_3/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    os.makedirs(save_path, exist_ok=True)

    model = Network(channels=96) 
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.pth_path).items()})
    model.cuda()
    model.eval()

    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        print('> {} - {}'.format(_data_name, name))
        
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        result = model(image)

        res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name,res*255)
