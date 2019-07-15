from path import Path
def reader():
	root = Path('D:/Data/Skeleton')
	file_before = root / 'before' / 'code_07' / 'code_07_53_1_skeleton.txt'
	print((root / 'before').glob('*'))
	with open(file_before) as f:
		skeleton = f.readlines()
		
	skeleton = skeleton[0].strip('\n').split(',')[8:]
	ske = []
	s = []
	for i in range(len(skeleton)):
		if i % 5 == 0 and i != 0:
			ske.append(s)
			s = []
		s.append(skeleton[i])
		
	print(skeleton)
	
def generate_list():
	root = Path('D:/Data/Skeleton/before')
	dirs = root.glob('*')
	save_path = 'D:/Data/Skeleton/training.txt'
	txt_list = []
	for d in dirs:
		t = d.glob('*skeleton.txt')
		t = [l.replace('\\', '/')[24:] for l in t]
		txt_list.extend(t)
	with open(save_path, 'w') as f:
		for item in txt_list:
			f.write('{:s}\n'.format(item))
	print('training.txt saving to {}'.format(save_path))
	
if __name__ == '__main__':
	generate_list()
