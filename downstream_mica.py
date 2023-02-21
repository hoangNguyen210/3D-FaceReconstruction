import cv2
import os
import sys
import glob
import random
import trimesh
import argparse
import numpy as np
from pathlib import Path
import argparse

from collections import OrderedDict
from time import sleep 

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


parser = argparse.ArgumentParser(description="Implementation of downstream base MICA")

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

parser.add_argument("--save_name", type=str, default= 'base_mica', help="save name")


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)
    cudnn.deterministic = True
    cudnn.benchmark = False
           
    
class OurDataset(Dataset):
    def __init__(self, 
                 list_img_path,
                 target_size,
                 app):
        self.list_img_path = list_img_path
        self.transform = T.Resize(size = target_size)
        self.app = app
        self.target_size = target_size
        self.num_vertices = num_vertices
        self.cache = {}               
    def __len__(self):
        return len(self.list_img_path)
    def to_batch(self,image_path):
        # src = path.replace('npy', 'jpg')
        # if not os.path.exists(src):
        #     src = path.replace('npy', 'png')
        img = cv2.imread(image_path)
        bboxes, kpss = self.app.det_model.detect(img, max_num=0, metric='default')
        i = get_center(bboxes, img)
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        blob, aimg = get_arcface_input(face, img)
        #npy , png
        image = face_align.norm_crop(img, landmark=face.kps, image_size=self.target_size[0])
        image = image / 255.
        image = cv2.resize(image, self.target_size).transpose(2, 0, 1)
        image = torch.tensor(image)
        arcface = torch.tensor(blob)
        return image, arcface

    def __getitem__(self, idx):
        image_path = self.list_img_path[idx]
        # '/content/drive/MyDrive/3D Face Reconstruction/MICA/MICA/data_1112/dataset/11_out/CTM02845/CTM02845_3.png'
        _, image_name = os.path.split(os.path.split(image_path)[0])
        # print(image_path)
        redundant_char = image_path[60:] # for label
        label_path = image_path.replace('data2D', 'label')
        label_path = label_path.replace(redundant_char, f'{image_name}.obj')
        # '/content/drive/MyDrive/3D Face Reconstruction/MICA/MICA/data_1112/label/11_out/CTM02845.obj'
        vertex_gt = trimesh.load_mesh(label_path)
        # vertex_gt = final_normalize(vertex_gt.vertices) #normalize the vertex
        vertex_gt = torch.tensor(vertex_gt.vertices)
#         vertex_gt /= 1000.
        image,arcface = self.to_batch(image_path)
        data = {
            'image':   image.float().contiguous(),
            'arcface': arcface.float().contiguous(),
            'vertex_gt': vertex_gt.float().contiguous(),
            'img_path' : image_path,
            'label_path': label_path
        }
        self.cache[idx] = data
        return data  
    
def load_checkpoint(model, model_path):
    epoch = 0
    map_location = 'cpu'
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


if __name__ == "__main__" :
    
    global args
    args = parser.parse_args()
    deterministic(args.seed)
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
    print('Train_path 0 : ',train_path[0])
    assert (len(train_path)  + len(test_path)  + len(val_path)) == data_len
    
    
    num_vertices = 5023
    target_size = (224,224)
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     app = FaceAnalysis( providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size= target_size)
    
    
    train_dataset = OurDataset(list_img_path = train_path, target_size = target_size, app = app)
    test_dataset = OurDataset(list_img_path = test_path, target_size = target_size, app = app)
    val_dataset = OurDataset(list_img_path = val_path, target_size = target_size, app = app)
    print(f'Len of train set {len(train_dataset)}, test set {len(test_dataset)}, val set {len(val_dataset)}')
    
    train_loader = DataLoader(
                train_dataset,
                batch_size= args.batch_size,
                pin_memory = False,
                num_workers = 10,
                shuffle=True,
                drop_last= False
                )
    test_loader = DataLoader(
                test_dataset,
                batch_size= args.batch_size,
                shuffle=False,
                drop_last= False,
                pin_memory = False,
                num_workers = 10,
                )
    val_loader = DataLoader(
                val_dataset,
                batch_size= args.batch_size,
                shuffle=False,
                drop_last= False,
                pin_memory = False,
                num_workers = 10,
    )
    print(f'Len of train loader {len(train_loader)}, test loader {len(test_loader)}, val loader {len(val_loader)}')
    
    cfg = get_cfg_defaults()
    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='MICA.micalib.models', model_name=cfg.model.name)(cfg, args.device)
    mica = load_checkpoint(mica, args.m)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        lr= args.lr,
        weight_decay= 0.0,
        params= mica.parameters_to_optimize(),
        amsgrad=False
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    mica.to(args.device)

    # model
    faces = mica.flameModel.generator.faces_tensor.cpu()
    
 
    loss_train_his = []
    loss_val_his = []
    min_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        mica.train()
        loss_train = 0
        loss_eval = 0

        n_train = len(train_dataset)
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img', position=0, leave=True) as pbar:
            for batch_ndx, data_train in enumerate(train_loader):
                images, arcface, vertex_gt = data_train['image'], data_train['arcface'], data_train['vertex_gt']
                images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1])
                codedict = mica.encode(images.to(args.device), arcface.to(args.device))
                opdict = mica.decode(codedict)
                meshes = opdict['pred_canonical_shape_vertices'] * 1000.0
                loss = criterion(meshes, vertex_gt.to(args.device))
                loss_train += loss.detach().item() 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(images.shape[0])
                pbar.set_postfix(OrderedDict({'Loss value': loss.detach().item()}))

        mica.eval()
        n_val = len(val_dataset)
        y_total = []
        predict_total = []
        # print ("Starting Evaluation")
        with torch.no_grad():
            with tqdm(total=n_val, desc='Evaluation Progress', unit='img', position=0, leave=True) as pbar_val:
                for batch_ndx, data_val in enumerate(val_loader):
                    #pred = target_model(x)
                    images, arcface, vertex_gt = data_val['image'], data_val['arcface'], data_val['vertex_gt']
                    images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1])
                    codedict = mica.encode(images.to(args.device), arcface.to(args.device))
                    opdict = mica.decode(codedict)
                    meshes = opdict['pred_canonical_shape_vertices'] * 1000.0
                    loss = criterion(meshes, vertex_gt.to(args.device))
                    loss_eval += loss.detach().item()
                    pbar_val.update(images.shape[0])
                    pbar_val.set_postfix(OrderedDict({'Loss value': loss.detach().item()}))
                    sleep(0.1)
                    # Append to compute total
        cur_train_loss = loss_train / float(len(train_loader))
        cur_val_loss =   loss_eval / float(len(val_loader))
        print('Loss Train:', cur_train_loss)
        print('Loss Eval:', cur_val_loss)
        loss_train_his.append(cur_train_loss)
        with open(f'{PATH + args.weight_dir}/train_{args.save_name}_loss.pkl', 'wb') as file:
            dump(loss_train_his, file)
        loss_val_his.append(cur_val_loss)
        
        with open(f'{PATH + args.weight_dir}/val_{args.save_name}_loss.pkl', 'wb') as file:
            dump(loss_val_his, file)
            
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate : {cur_lr}")
        if min_val_loss > cur_val_loss:
            print(f'Validation Loss Decreased({min_val_loss:.6f}--->{cur_val_loss:.6f}) \t Saving The Model')
        
            min_val_loss = cur_val_loss
            model_dict = OrderedDict()
            model_dict['model_dict'] = mica.model_dict()
            model_dict['opt'] = optimizer.state_dict()
            model_dict['scheduler'] = scheduler.state_dict()
            model_dict['epoch'] = epoch + 1
            torch.save(model_dict, f'{PATH + args.weight_dir}/train_{args.save_name}.pth.tar')
            
#         scheduler.step()


    save_path =  f'{PATH + args.weight_dir}/train_{args.save_name}.pth.tar'
    mica = load_checkpoint(mica,save_path)
    mica.eval()
    n_test = len(test_dataset)
    cur_test_loss = 0 
    with torch.no_grad():
        loss_test = 0 
        with tqdm(total=n_test, desc='Testing Progress', unit='img', position=0, leave=True) as pbar_test:
            for batch_ndx, data_test in enumerate(test_loader):
                #pred = target_model(x)
                images, arcface, vertex_gt = data_test['image'], data_test['arcface'], data_test['vertex_gt']
                images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1])
                codedict = mica.encode(images.to(args.device), arcface.to(args.device))
                opdict = mica.decode(codedict)
                meshes = opdict['pred_canonical_shape_vertices']
                loss = criterion(meshes* 1000, vertex_gt.to(args.device))
                loss_test += loss.detach().item()
                pbar_test.update(images.shape[0])
                pbar_test.set_postfix(OrderedDict({'Loss value': loss.detach().item()}))
                sleep(0.1)
                # Append to compute total
    cur_test_loss = loss_test / float(len(test_loader))
    print('Test Loss : ', cur_test_loss)
