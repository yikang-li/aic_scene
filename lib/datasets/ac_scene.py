import torch
import json
import torch.utils.data as data
import os.path as osp
import os
import cv2
from PIL import ImageFile, Image

class ac_scene(data.Dataset):

	normalize = {'mean': [0.485, 0.456, 0.406],
               'std': [0.229, 0.224, 0.225]}

    def __init__(self, split, opts, is_test=False):

        global normalize 
        self.opts = opts
        self.root_dir = opts['root_dir']
        self.testing = is_test
        # loading training annotations
        if 'train' in split: # loading train and val set
        	with open(osp.join(opts['root_dir'], opts['train']['annotation']), 'r') as f:
        		annotations = json.load(f)
        	for item in annotations:
        		item['subfolder'] = opts['train']['im_dir']
        	self.annotations = annotations
	        if split == 'trainval':
	        	with open(osp.join(opts['root_dir'], opts['val']['annotation']), 'r') as f:
	        		annotations = json.load(f)
	        	for item in annotations:
	        		item['subfolder'] = opts['val']['im_dir']
	        	self.annotations.append(annotations)
        else:
        	self.annotations = []
        	for item in os.listdir(osp.join(self.root_dir, opts['test']['im_dir'])):
        		if os.path.isfile(osp.join(self.root_dir, opts['test']['im_dir'], item)):
        			self.annotations.append({'image_id': item, 
        					'subfolder': opts['test']['im_dir']})

        # image transform
        if is_test:
        	self.transform = transforms.Compose([ 
            	transforms.Scale(opts['scale_size']), # we scale the image in advance
                transforms.CenterCrop(opts['img_size']),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize)
            ])
        else:
            self.transform = transforms.Compose([ 
            	transforms.Scale(opts['scale_size']), # We scale the image in advance
                transforms.RandomCrop(opts['img_size']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize)
            ])


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        #  pdb.set_trace()
        item = {}
        img = cv2.imread(osp.join(self.root_dir, self.annotations[index]['subfolder'], self.annotations[index]['image_id']))
        img = Image.fromarray(img)
        img = self.transform(img)
        item['visual'] = item
        item['image_id'] = self.annotations[index]['image_id']
        try:
        	item['label'] = torch.LongTensor([int(self.annotations[index]['label_id'])])
        except:
        	pass
        return item