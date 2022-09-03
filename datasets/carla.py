import os
import random
import numpy as np
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from datasets.data_io import get_transform, read_all_lines


class CarlaDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.middle_filenames, self.right_filenames, self.disp_filenames = self.load_path(
            list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        middle_images = [x[1] for x in splits]
        right_images = [x[2] for x in splits]
        disp_images = [x[3] for x in splits]
        return left_images, middle_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = io.imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        middle_img = self.load_image(os.path.join(self.datapath, self.middle_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            middle_img = middle_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            processed = get_transform()
            left_img = processed(left_img)
            middle_img = processed(middle_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "middle": middle_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            middle_img = middle_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            middle_img = processed(middle_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "middle": middle_img,
                    "right": right_img,
                    "disparity": disparity,
                    "left_filename": self.left_filenames[index],
                    "middle_filename": self.middle_filenames[index],
                    "right_filename": self.right_filenames[index],
                    "disparity_filename": self.disp_filenames[index]}
