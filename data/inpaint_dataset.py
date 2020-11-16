import torch
import numpy as np
import cv2
import os
import math
from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

ALLMASKTYPES = ['centerbox','halfbox','single_bbox', 'bboxes', 'free_form']

class InpaintDataset(Dataset):
    """
    Dataset for Inpainting task
    Params:
        img_flist_path(str): The file which contains img file path list (e.g. test.flist)
        mask_flist_paths_dict(dict): The dict contain the files which contains the pkl or xml file path for
                                generate mask. And the key represent the mask type (e.g. {"bbox":"bbox_flist.txt", "seg":..., "random":None})
        resize_shape(tuple): The shape of the final image (default:(256,256))
        transforms_oprs(list) : Determine which transformation used on the imgae (default:['random_crop', 'to_tensor'])
        random_bbox_shape(tuple): if use random bbox mask, it define the shape of the mask (default:(32,32))
        random_bbox_margin(tuple): if use random bbox, it define the margin of the bbox which means the distance between the mask and the margin of the image
                                    (default:(64,64))
    Return:
        img, *mask
    """
    def __init__(self, opt):
        assert opt.mask_type in ALLMASKTYPES
        self.opt=opt
        self.img_flist_path=opt.flist
        with open(self.img_flist_path, 'r') as f:
            self.img_paths = f.read().splitlines()
        self.transform = transforms.Compose([
            #transforms.Resize(int(opt.imgsize * 1.12), Image.BICUBIC),
            #transforms.RandomCrop(opt.imgsize, opt.imgsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # create the paths for images and masks

        img_path = self.img_paths[index]
        error = 1
        while not os.path.isfile(img_path) or error == 1:
            try:
                img = self.transform(self.read_img(img_path))
                error = 0
            except:
                index = np.random.randint(0, high=len(self))
                img_path = self.img_paths[index]
                error = 1

        img = self.transform(self.read_img(img_path))

        if self.opt.mask_type=='centerbox':
            mask=self.centerbox(shape = self.opt.imgsize)
        if self.opt.mask_type=='halfbox':
            mask=self.halfbox(shape=self.opt.imgsize)
        if self.opt.mask_type=='single_bbox':
            mask=self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
        if self.opt.mask_type == 'bboxes':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
        if self.opt.mask_type == 'free_form':
            #mask = self.random_ff_mask(shape = self.opt.imgsize, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num)
            mask= self.random_form(shape = self.opt.imgsize, min_width=self.opt.min_width, max_width=self.opt.max_width)

        return img, torch.from_numpy(mask.transpose((2,0,1)))

    def read_img(self, path):
        """
        Read Image
        """
        img = Image.open(path).convert("RGB")
        return img

    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.

        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        Returns:
            tuple: (top, left, height, width)

        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low=ver_margin, high=maxt)
        l = np.random.randint(low=hor_margin, high=maxl)
        h = height
        w = width
        return (t, l, h, w)

    def random_ff_mask(self, shape, max_angle = 4, max_len = 100, max_width = 40, times = 15):
        """Generate a random free form mask with configuration.

        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        Returns:
            tuple: (top, left, height, width)
        """

        h = shape
        w = shape
        mask = np.zeros((h,w))
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(w//2)
            start_y = np.random.randint(h//2)
            for j in range(1+np.random.randint(5)):
                angle = 0.01+np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 20+np.random.randint(max_len)
                brush_w = 20+np.random.randint(max_width)
                end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, w)
                end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, h)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
                #cv2.circle(mask, (start_x,start_y),brush_w//2,1.0,-1)
        assert np.mean(mask)!=0       
        return mask.reshape(mask.shape+(1,)).astype(np.float32)

    def random_form(self, shape, min_width=12, max_width = 40):
        H = shape
        W = shape
        mean_angle = 2*math.pi / 5
        angle_range = 2*math.pi / 15
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)
        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(4, 12)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            vertex.append((int(np.random.randint(0+W/4, W-W/4)), int(np.random.randint(0+H/4, H-H/4))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, W)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, H)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        return mask.reshape(mask.shape+(1,))

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.

        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

        Returns:
            tf.Tensor: output with shape [1, H, W, 1]

        """
        height=shape
        width=shape
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        mask = np.zeros(( height, width), np.float32)
        #print(mask.shape)
        for bbox in bboxs:
            h = int(0.1*bbox[2])+np.random.randint(int(bbox[2]*0.2+1))
            w = int(0.1*bbox[3])+np.random.randint(int(bbox[3]*0.2)+1)
            mask[bbox[0]+h:bbox[0]+bbox[2]-h,
                 bbox[1]+w:bbox[1]+bbox[3]-w] = 1.
        #print("after", mask.shape)
        return mask.reshape(mask.shape+(1,)).astype(np.float32)

    def centerbox(self, shape):
        height=shape
        width=shape
        mask=np.zeros(( height, width), np.float32)
        mask[height//4:3*height//4, width//4:3*width//4]=1.
        return mask.reshape(mask.shape+(1,)).astype(np.float32)

    def halfbox(self,shape):
        height=shape
        width=shape
        mask=np.zeros(( height, width), np.float32)
        mask[height//8:7*height//8, width//8:7*width//8]=1.
        return mask.reshape(mask.shape+(1,)).astype(np.float32)