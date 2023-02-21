import cv2
import os
import sys
import glob
import random
import argparse
import numpy as np
from pathlib import Path
import argparse


import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from loguru import logger
from skimage.io import imread

# import trimesh
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align

sys.path.append('/home/caduser/HDD/A_MICA/')
from MICA.configs.config import get_cfg_defaults
from MICA.datasets.creation.util import get_arcface_input, get_center
from MICA.utils import util


from lightly.transforms import GaussianBlur, Jigsaw, RandomRotate, RandomSolarization
# import torch
# import torchvision
import lightly.models as models
import lightly.loss as loss
import lightly.data as data

from collections import OrderedDict
import pickle 
from pickle import load, dump
from lightly.data import LightlyDataset
from lightly.data import SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead
from lightly.models.modules import SwaVPrototypes

import torchvision.transforms as T
from typing import List, Tuple, Union



PATH = "/home/caduser/HDD/hoang_graph_matching/A_MICA/"

SAVE_PATH = ""

parser = argparse.ArgumentParser(description="Implementation of SSL MICA")

# data 
parser.add_argument('-data_path', default= PATH + 'data2D/', type=str, help='2D Data path')
parser.add_argument("--data_size", type=int, default= 3000, help="length of data")

# hyperparameters
parser.add_argument("--percent_data", type=float, default=0.8)
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate")
parser.add_argument("--epochs", type=int, default=50,
                    help="# of epochs")
parser.add_argument("--batch_size", type=int, default= 200, help="batch size")
parser.add_argument('-weight_dir', default= 'weight2', type=str, help='weight folder')

# path to save images
parser.add_argument('-i', default= PATH + 'MICA/demo/input', type=str, help='Input folder with images')
parser.add_argument('-o', default= PATH + 'MICA/demo/output', type=str, help='Output folder')
parser.add_argument('-a', default= PATH + 'MICA/demo/arcface', type=str, help='Processed images for MICA input')
parser.add_argument('-m', default= PATH + 'MICA/data/pretrained/mica.tar', type=str, help='Pretrained model path')

# other params
parser.add_argument("--seed", type=int, default=42,
                    help="seed value")
parser.add_argument("--device", type=str, default= "cuda:1", help="device index")


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)
    cudnn.deterministic = True
    cudnn.benchmark = False
                
                
def _random_rotation_transform(
    rr_prob: float,
    rr_degrees: Union[None, float, Tuple[float, float]],
) -> Union[RandomRotate, T.RandomApply]:
    if rr_degrees is None:
        # Random rotation by 90 degrees.
        return RandomRotate(prob=rr_prob, angle=90)
    else:
        # Random rotation with random angle defined by rr_degrees.
        return T.RandomApply([T.RandomRotation(degrees=rr_degrees)], p=rr_prob)

    
class SwaV(nn.Module):
    def __init__(self, backbone, device):
        super().__init__()
        self.device = device
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(5023, 512, 512)
        self.prototypes = SwaVPrototypes(512, n_prototypes=3000)

    def forward(self, data):
        # x = self.backbone(x).flatten(start_dim=1)

        images, arcface = data['image'], data['arcface']
        list_meshes = []
        for ind in range(len(images)):
            cur_images = images[ind]
            cur_arcfaces = arcface[ind]
            cur_images = cur_images.view(-1, cur_images.shape[-3], cur_images.shape[-2], cur_images.shape[-1])
            cur_arcfaces = cur_arcfaces.view(-1, cur_arcfaces.shape[-3], cur_arcfaces.shape[-2], cur_arcfaces.shape[-1])
            codedict = mica.encode(cur_images.to(self.device), cur_arcfaces.to(self.device))
            opdict = mica.decode(codedict)
            meshes = opdict['pred_canonical_shape_vertices']
            # list_meshes.append(meshes)
            x = self.projection_head(torch.mean(meshes * 1000., dim = 2))
            x = nn.functional.normalize(x, dim=1, p=2)
            p = self.prototypes(x)
            list_meshes.append(p)
        return list_meshes
    
class SSLDataset(Dataset):
    def __init__(self, 
                 list_img_path,
                 target_size,
                 transform, 
                 app):
        self.list_img_path = list_img_path
        self.transform = T.Resize(size = target_size)
        self.app = app
        self.target_size = target_size
        self.num_vertices = num_vertices
        self.crop_transforms = transform 
        self.cache = {}             

    def __len__(self):
        return len(self.list_img_path)
    def to_batch(self,img):
        # img = cv2.imread(image_path)
        bboxes, kpss = self.app.det_model.detect(img, max_num=0, metric='default')
        i = get_center(bboxes, img)
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        blob, aimg = get_arcface_input(face, img)
        image = face_align.norm_crop(img, landmark=face.kps, image_size=self.target_size[0])
        image = image / 255.
        image = cv2.resize(image, self.target_size).transpose(2, 0, 1)
        image = torch.tensor(image)
        arcface = torch.tensor(blob)
        return image, arcface

    def __getitem__(self, idx):
        image_path = self.list_img_path[idx]
#         print(image_path)
        img = cv2.imread(image_path)
        # print(image_path)
        crop_imgs_list = [transform(img) for transform in self.crop_transforms]
        # image,arcface = [self.to_batch(img.numpy()) for img in crop_imgs_list]
        images = []
        arcfaces = []
        for img_aug in crop_imgs_list :
            np_img = img_aug.permute(1,2,0).numpy() 
            np_img = (np_img * 255).astype(np.uint8)
            image,arcface = self.to_batch(np_img)
            images.append(image)
            arcfaces.append(arcface)

        data = {
            # 'image':   image.float().contiguous(),
            # 'arcface': arcface.float().contiguous(),
            'image':   images,
            'arcface': arcfaces,
            'img_path' : image_path,
        }
        self.cache[idx] = data
        return data      
    
def load_checkpoint(model, model_path):
    epoch = 0
    map_location = 'cpu'
    print('model_path: ' ,model_path)
#     model_path = '/home/caduser/HDD/hoang_graph_matching/A_MICA/MICA/data/pretrained/mica.tar'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location)
        if 'epoch' in checkpoint :
            epoch = checkpoint['epoch']
            print(f'Epoch : {epoch}')
        if 'arcface' in checkpoint:
            print('Find ARCFACE')
            model.arcface.load_state_dict(checkpoint['arcface'])
        if 'flameModel' in checkpoint:
            print('Find FLAMEMODEL')
            model.flameModel.load_state_dict(checkpoint['flameModel'])
    return model


def load_checkpoint_1(model, model_path):
    epoch = 0
    map_location = 'cpu'
    print('model_path: ' ,model_path)
#     model_path = '/home/caduser/HDD/hoang_graph_matching/A_MICA/MICA/data/pretrained/mica.tar'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location)
        if 'epoch' in checkpoint :
            epoch = checkpoint['epoch']
            print(f'Epoch : {epoch}')
        if 'model_dict' in checkpoint :
            model_dict = checkpoint['model_dict']
            if 'arcface' in model_dict:
                print('Find ARCFACE')
                model.arcface.load_state_dict(model_dict['arcface'])
            if 'flameModel' in model_dict:
                print('Find FLAMEMODEL')
                model.flameModel.load_state_dict(model_dict['flameModel'])
    return model


def restart_checkpoint(model, weight):
    if 'arcface' in weight:
        print('Find ARCFACE')
        model.arcface.load_state_dict(weight['arcface'])
    if 'flameModel' in weight:
        print('Find FLAMEMODEL')
        model.flameModel.load_state_dict(weight['flameModel'])
    return model



if __name__ == "__main__" :
    
    global args
    args = parser.parse_args()
    print(args)
    deterministic(args.seed)
#     path = '/home/caduser/HDD/hoang_graph_matching/A_MICA/data2D/'
    list_img_path = []
    list_obj_folder = glob.glob(args.data_path + '/*')
    for obj_folder in list_obj_folder :
        list_identity_name = glob.glob(obj_folder + '/*')

        for identity_name in list_identity_name :
            list_img = glob.glob(identity_name + '/*')

            for img3D_path in list_img :
                if "GT" in img3D_path :
                    continue
                list_img_path.append(img3D_path)

    # random.shuffle(list_img_path)  # len 2550 
    list_img_path = list_img_path[:args.data_size]
    data_len = len(list_img_path)
    train_len = int(0.8*data_len)
    val_len = int(0.1*data_len)
    test_len =  data_len - train_len - val_len

    # Prepare train, test,
    train_path = list_img_path[:train_len]
    val_path = list_img_path[train_len: train_len + val_len]
    test_path = list_img_path[train_len + val_len: ]
    # print('Train_path 0 : ',train_path[0])
    assert (len(train_path)  + len(test_path)  + len(val_path)) == data_len
    
    
    num_vertices = 5023
    target_size = (224,224)
    # app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app = FaceAnalysis( providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size= target_size)
    
    
    
    imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
    crop_sizes = [224*4, 224*4]
    crop_counts = [2, 2]
    crop_min_scales = [0.14, 0.05]
    crop_max_scales =  [1.0, 0.14]
    hf_prob = 0.5
    vf_prob = 0.0
    rr_prob = 0.0
    rr_degrees = None
    cj_prob = 0.8
    cj_strength = 0.8
    random_gray_scale =  0.2
    gaussian_blur= 0.
    kernel_size = 1.0
    normalize = imagenet_normalize

    color_jitter = T.ColorJitter(
        cj_strength, cj_strength, cj_strength, cj_strength/4,
    )

    transforms = T.Compose([
        T.RandomHorizontalFlip(p=hf_prob),
        T.RandomVerticalFlip(p=vf_prob),
        _random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
        T.ColorJitter(brightness=.02, hue=.02),
        # T.RandomApply([color_jitter], p=cj_prob),
        T.RandomGrayscale(p=random_gray_scale),
        GaussianBlur(kernel_size, prob=gaussian_blur),
        T.ToTensor(),
        # T.Normalize(mean=normalize['mean'], std=normalize['std'])
    ])



    crop_transforms = []
    for i in range(len(crop_sizes)):

        random_resized_crop = T.RandomResizedCrop(
            crop_sizes[i],
            scale=(crop_min_scales[i], crop_max_scales[i])
        )

        crop_transforms.extend([
            T.Compose([
                T.ToPILImage(),
                T.CenterCrop((int(224*3),int(224*3))),
                transforms,
            ])
        ] * crop_counts[i])

#     random.seed(args.seed)
    random.shuffle(train_path)  # len 2550
    
#     ssl_path = train_path[: int( args.percent_data * len(train_path) )]
    ssl_path = train_path[int( args.percent_data * len(train_path) ) : int( (args.percent_data + 0.2) * len(train_path) )]
    ssl_dataset = SSLDataset(list_img_path = ssl_path, target_size = target_size, app = app,  transform = crop_transforms)

    
    ssl_loader = DataLoader(
                ssl_dataset,
                batch_size= args.batch_size,
                shuffle=False,
                drop_last= False,
                num_workers = 10,
                pin_memory = False
                )
    print('Number of SSL images : ', len(ssl_dataset))
    print('Length of SSL loader : ', len(ssl_loader))


    cfg = get_cfg_defaults()
    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='MICA.micalib.models', model_name=cfg.model.name)(cfg, args.device)
    
    # load checkpoint from previous ssl weight to continue train ssl
    ssl_path = f'{PATH + args.weight_dir}/ssl_{int( (args.percent_data - 0.2) * 100)}.pth'
    mica = load_checkpoint(mica, ssl_path)
    model  = SwaV(mica, device = args.device)

    # loss
    criterion = SwaVLoss()
    # optimizer 
    optimizer = torch.optim.Adam(
        lr= args.lr,
        weight_decay= 0.0,
        params= model.backbone.parameters_to_optimize(),
        amsgrad=False
        )
    # restart from checkpoint
    cur_epoch = 0
    # define SAVE PATH
    save_dir = f'{PATH + args.weight_dir}/ssl_{int(args.percent_data * 100)}.pth'
    if os.path.exists(save_dir) :
        weight = torch.load(save_dir)
        if 'epoch' in weight :
            cur_epoch = weight['epoch']
        if 'model_dict' in weight :
            model_dict= weight['model_dict']
            print("Find MODEL DICT")
            mica = restart_checkpoint(mica, model_dict)
            model  = SwaV(mica, device = args.device)
        if 'optimizer' in weight :
            print("Find OPTIMIZER")
            op = weight['optimizer']
            optimizer.load_state_dict(op)
        

    model.to(args.device)

    # train model
    ssl_loss_his = []
    ssl_epochs = args.epochs
    print("Starting Training")
    for epoch in range(cur_epoch,ssl_epochs + 1):
        total_loss = 0
        n_train = len(ssl_dataset)
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{ssl_epochs}', unit='img', position=0, leave=True) as pbar:
            for it, data in enumerate(ssl_loader):
                model.prototypes.normalize()
                multi_crop_features = model(data)
                high_resolution = multi_crop_features[:2]
                low_resolution = multi_crop_features[2:]
                loss = criterion(high_resolution, low_resolution)
                total_loss += loss.detach().item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(data['image'][0].shape[0])
                pbar.set_postfix(OrderedDict({'Loss value': loss.detach().item()}))
        cur_lr = optimizer.param_groups[0]['lr']
#         print(f"Learning rate : {cur_lr}")
        model_dict = OrderedDict()
        model_dict['model_dict'] = model.backbone.model_dict()
        model_dict['optimizer'] = optimizer.state_dict()
        model_dict['epoch'] = epoch
        # save model weight
        torch.save(model_dict, f'{save_dir}')
        avg_loss = total_loss / len(ssl_loader)
        ssl_loss_his.append(avg_loss)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
        # save loss value for each epoch
        with open(f'{PATH + args.weight_dir}/ssl_loss_{int(args.percent_data * 100)}.pkl', 'wb') as file:
            dump(ssl_loss_his, file)



