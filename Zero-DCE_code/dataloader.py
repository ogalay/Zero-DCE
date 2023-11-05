import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(lowlight_images_path):

	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")

	train_list = image_list_lowlight

	random.shuffle(train_list)

	return train_list

	

class lowlight_loader(data.Dataset):
	def __init__(self, images_name, dataset_folder):
		self.image_name_list = images_name
		self.size = 256
		self.dataset_folder = dataset_folder

	def __getitem__(self, index):
		data_lowlight = Image.open(f'{self.dataset_folder}/low/{self.image_name_list[index]}')
		data_highlight = Image.open(f'{self.dataset_folder}/high/{self.image_name_list[index]}')

		data_lowlight = data_lowlight.resize((self.size, self.size), Image.LANCZOS)
		data_highlight = data_highlight.resize((self.size, self.size), Image.LANCZOS)

		data_lowlight = (np.asarray(data_lowlight) / 255.0)
		data_lowlight = torch.from_numpy(data_lowlight).float()

		data_highlight = (np.asarray(data_highlight) / 255.0)
		data_highlight = torch.from_numpy(data_highlight).float()

		return data_lowlight.permute(2, 0, 1), data_highlight.permute(2, 0, 1)

	def __len__(self):
		return len(self.image_name_list)
