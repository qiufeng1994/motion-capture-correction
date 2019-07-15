from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import random
from path import Path

class Skeleton_Data(data.Dataset):
	"""
	Args:
		root (string): Root directory of dataset
		train (bool, optional): If True, creates dataset from ``training.pt``,
			otherwise from ``test.pt``.
		download (bool, optional): If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
	"""
	training_list = 'training.txt'
	test_list = 'valid.txt'
	# valid_file = 'validation_dataset/validation_data'
	
	def __init__(self, root, list_path, train=True, transform=None, target_transform=None, download=False, if_seq = True):
		self.root = root
		self.transform = transform
		self.if_seq = if_seq
		self.target_transform = target_transform
		self.train = train  # training set or test set
		if self.train:
			self.train_root = (Path(self.root) / self.training_list)
			self.train_samples = self.collect_samples(self.root, self.training_list,self.if_seq)
		else:
			self.test_root = (Path(self.root) / self.test_list)
			self.test_samples = self.collect_samples(self.root, self.test_list,self.if_seq)
	
	def collect_samples(self, root, list, if_seq):
		data = []
		window = 2
		list = self.txt_reader(Path(root) / list)
		list.sort()
		for l in list:
			before_file = root + '/before/' + l
			after_file = root + '/after/' + l
			before_data = self.txt_reader(before_file)
			after_data = self.txt_reader(after_file)
			if not if_seq:
				for i in range(len(before_data)):
					data.append([before_data[i], after_data[i]])
			else:
				if not self.train:
					# the first two frames
					data.append([before_data[2 - window:2 + window+1], after_data[2 - window: 2 + window +1]])
					data.append([before_data[3 - window:3 + window+1], after_data[3 - window: 3 + window +1]])
					
				for i in range(window, len(before_data) - window):
					data.append([before_data[i - window:i + window+1], after_data[i - window: i + window +1]])
				if not self.train:
					last = len(before_data)-window
					data.append([before_data[last - window:last + window+1], after_data[last - window: last + window +1]])
					data.append([before_data[last+1 - window:last+1 + window+1], after_data[last+1 - window: last +1+ window +1]])
		return data
	
	def collect_samples_seq(self,root,list):
		window = 5 // 2
		data = []
		list = self.txt_reader(Path(root)/list)
		list.sort()
		for l in list:
			before_file = root+'/before/' + l
			after_file = root+ '/after/' + l
			before_data = self.txt_reader(before_file)
			after_data = self.txt_reader(after_file)

			for i in range(window, len(before_data)-window-1):
				data.append([before_data[i-window:i+window+1], after_data[i-window: i+window+1]])
			return data
		
		
		
		
	def txt_reader(self, file):
		with open(file) as f:
			data = f.readlines()
		return [d.strip('\n') for d in data]
	
	def transform_skeleton(self, data):
		skt = []
		s = []
		for i in range(len(data)):
			if i % 5 == 0 and i != 0:
				skt.append(s)
				s = []
			s.append(data[i])
		return np.array([list(map(float,e[1:])) for e in skt]) # convert str to ndarray
		
	def load_samples(self, s):
		before = s[0].split(',')[8:]
		after = s[1].split(',')[8:]
		before = self.transform_skeleton(before)
		after = self.transform_skeleton(after)
		
		return [before, after]
	
	def load_samples_seq(self, s):
		# str to list
		before = [e.split(',')[8:] for e in s[0]]
		after = [e.split(',')[8:] for e in s[1]]
		# list to array
		before = [self.transform_skeleton(e) for e in before]
		after = [self.transform_skeleton(e) for e in after]
		# list to array
		before = np.array([l.reshape([-1]) for l in before])
		after = np.array([l.reshape([-1]) for l in after])
		return [np.expand_dims(before,0), np.expand_dims(after,0)]
		
	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		
		if self.train:
			s = self.train_samples[index]
		else:
			s = self.test_samples[index]
		if not self.if_seq:
			input, target = self.load_samples(s)
			input = np.expand_dims(input, 0)
			target = target.reshape(-1)
		else:
			input, target = self.load_samples_seq(s)

		if self.transform is not None:
			input = self.transform(input)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return input, target
	
	def __len__(self):
		if self.train:
			return len(self.train_samples)
		else:
			return len(self.test_samples)
	
	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		tmp = 'train' if self.train is True else 'test'
		fmt_str += '    Split: {}\n'.format(tmp)
		fmt_str += '    Root Location: {}\n'.format(self.root)
		tmp = '    Transforms (if any): '
		fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		tmp = '    Target Transforms (if any): '
		fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		return fmt_str



if __name__ == '__main__':
	root = 'D:\Data\Skeleton'
	data_loader = Skeleton_Data(root, list_path= '1',train=True, transform=None, target_transform=None,)
	for input, target in data_loader:
		print(input)
		print(target)