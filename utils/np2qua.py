from path import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

skeleton_name = ["Bip001", 'Bip001 Pelvis', 'Bip001 Spine', 'Bip001 Spine1', 'Bip001 Spine2', 'Bip001 Neck',
                 'Bip001 Head', 'Bip001 HeadNub', 'Bip001 L Clavicle', 'Bip001 L UpperArm', 'Bip001 L Forearm',
                 'Bip001 L ForeTwist', 'Bip001 L Hand', 'Bip001 L Finger0', 'Bip001 L Finger01', 'Bip001 L Finger02',
                 'Bip001 L Finger0Nub','Bip001 L Finger1', 'Bip001 L Finger11', 'Bip001 L Finger12',
                 'Bip001 L Finger1Nub', 'Bip001 L Finger2','Bip001 L Finger21', 'Bip001 L Finger22',
                 'Bip001 L Finger2Nub', 'Bip001 L Finger3', 'Bip001 L Finger31', 'Bip001 L Finger32',
                 'Bip001 L Finger3Nub', 'Bip001 L Finger4', 'Bip001 L Finger41', 'Bip001 L Finger42',
                 'Bip001 L Finger4Nub', 'Bip001 R Clavicle', 'Bip001 R UpperArm', 'Bip001 R Forearm',
                 'Bip001 R ForeTwist', 'Bip001 R Hand', 'Bip001 R Finger0', 'Bip001 R Finger01', 'Bip001 R Finger02',
                 'Bip001 R Finger0Nub', 'Bip001 R Finger1', 'Bip001 R Finger11', 'Bip001 R Finger12',
                 'Bip001 R Finger1Nub', 'Bip001 R Finger2', 'Bip001 R Finger21', 'Bip001 R Finger22',
                 'Bip001 R Finger2Nub', 'Bip001 R Finger3', 'Bip001 R Finger31', 'Bip001 R Finger32',
                 'Bip001 R Finger3Nub', 'Bip001 R Finger4', 'Bip001 R Finger41', 'Bip001 R Finger42',
                 'Bip001 R Finger4Nub', 'Bip001 R Thigh', 'Bip001 R Calf', 'Bip001 R Foot', 'Bip001 R Toe0',
                 'Bip001 R Toe0Nub', 'Bip001 L Thigh', 'Bip001 L Calf', 'Bip001 L Foot', 'Bip001 L Toe0',
                 'Bip001 L Toe0Nub']

unchange = {'Bip001 Pelvis': [-0.5, 0.5, 0.5, 0.5],
            'Bip001 HeadNub': [0, 0, 0, 1],
            # 'Bip001 L Forearm':[0,0,*,*],
            # 'Bip001 L ForeTwist':[*,1,1,*],
            'Bip001 L Finger0Nub': [0, 0, 0, 1],
            'Bip001 L Finger1Nub': [0, 0, 0, 1],
            'Bip001 L Finger2Nub': [0, 0, 0, 1],
            'Bip001 L Finger3Nub': [0, 0, 0, 1],
            'Bip001 L Finger4Nub': [0, 0, 0, 1],
            # 'Bip001 R Forearm'     : [0,0,*,*],
            # 'Bip001 R ForeTwist'   : [*,0,0,*],
            'Bip001 R Finger0Nub': [0, 0, 1, 0],
            'Bip001 R Finger1Nub': [0, 0, 1, 0],
            'Bip001 R Finger2Nub': [0, 0, -1, 0],
            'Bip001 R Finger3Nub': [0, 0, 1, 0],
            'Bip001 R Finger4Nub': [0, 0, 1, 0],
            # 'Bip001 R Calf', [0,0,*,*],
            'Bip001 R Toe0': [0, 0, -0.707107, 0.707107],
            'Bip001 R Toe0Nub': [0, 0, 0, 1],
            # 'Bip001 L Calf', [0,0,*,*],
            'Bip001 L Toe0': [0, 0, -0.707107, 0.707107],
            'Bip001 L Toe0Nub': [0, 0, 1, 0],
            }

keep_same = ['Bip001 Spine', 'Bip001 Spine1', 'Bip001 Spine2', 'Bip001 Neck',
             'Bip001 Head', 'Bip001 L Clavicle', 'Bip001 R Clavicle', 'Bip001 R Thigh',
             'Bip001 R Calf', 'Bip001 R Foot', 'Bip001 L Thigh', 'Bip001 L Calf', 'Bip001 L Foot',
             ]

root = 'D:/Data/Skeleton'
result_file = 'D:/Data/Skeleton/training_result/result.txt'
result_file_np = 'D:/Data/Skeleton/training_result/result.npy'
list_file = 'D:/Data/Skeleton/valid.txt'
def transform_result():
	
	with open(result_file) as f:
		result = f.readlines()
	with open(list_file) as f:
		data_list = f.readlines()
	result_np = np.load(result_file_np)
	cnt_lines = 0
	for l in data_list:
		l = l.strip('\n')
		file_path = Path(root + '/before/' + l)
		with open(file_path) as f:
			data = f.readlines()
		
		num_frames = len(data)
		saving_file = Path(root + '/training_result/' + l)
		
		# result_formated = format_trans(result[cnt_lines * 67: (cnt_lines + num_frames) * 67], data)
		result_formated = format_trans(result_np[cnt_lines:cnt_lines + num_frames], data, num_frames)
		cnt_lines += num_frames
		with open(saving_file, 'w') as f:
			for item in result_formated:
				f.writelines('%s\n' % item)
			print('writing to {}'.format(saving_file))
	
	


def format_trans(before_trans, data, num_frames):
	after_trans = []
	before_trans = before_trans.squeeze()
	# num_frames = len(before_trans) // 67
	before_trans_smooth = before_trans[:2]
	smooth = []
	for i in range(268):
		point_smooth = savgol_filter(before_trans[2:num_frames-2, i].squeeze(), args.window, args.order)
		smooth.append(point_smooth)
	smooth = np.array(smooth).transpose((1,0))
	before_trans_smooth = np.vstack((before_trans_smooth, smooth))
	before_trans_smooth = np.vstack((before_trans_smooth, before_trans[num_frames-2:]))
	for f in range(num_frames):
		base = ''
		for b in data[f].split(',')[1:8]: # base params
			base += b + ','
		l = skeleton_name[0] + ',' + base
		pos = before_trans_smooth[f].reshape(-1,4)
		for p_i in range(67):
			# pos = before_trans[f * 67 + p].strip('\n').strip('[').strip(']').strip(' ').replace('  ', ',').replace(' ',',')
			# optimation: fixed some constant values
			if skeleton_name[p_i+1] in unchange.keys():
				pos_i = ['{:f}'.format(p) for p in unchange[skeleton_name[p_i+1]]]
			elif skeleton_name[p_i+1] in keep_same:
				data_frame = data[f].split(',')[8 + p_i * 5: 13 + p_i * 5]
				assert data_frame[0] == skeleton_name[p_i+1]
				pos_i = data_frame[1:]
			else:
				pos_i = ['{:f}'.format(p) for p in pos[p_i]]
			pos_i = ",".join(pos_i)
			l += skeleton_name[p_i+1] + ',' + pos_i + ','
		after_trans.append(l)
	return after_trans

def savgol_test():
	# result_np = np.load(result_file_np)
	# y = result_np[:118, :, 1].squeeze()
	# y = result_file_n p
	# x = np.linspace(0,100,118)
	# windows = 15
	# polyorder = 9
	# yhat = savgol_filter(y, windows, polyorder)
	# plt.plot(x,y, label='origin')
	# plt.plot(x, yhat, color = 'red', label = 'savgol_filter:{},{}'.format(windows, polyorder))
	# plt.legend()
	# plt.show()
	x = np.arange(1, 11)
	y = 2 * x + 5
	plt.title("Matplotlib demo")
	plt.xlabel("x axis caption")
	plt.ylabel("y axis caption")
	plt.plot(x, y)
	plt.show()
	
if __name__ == '__main__':
	# import argparse
	# parser = argparse.ArgumentParser(description='numpy to qua format')
	# parser.add_argument('-w', '--window', default=7, type=int, help='window length of filter')
	# parser.add_argument('-o', '--order', default=3, type=int, help='polyorder of filter')
	# args = parser.parse_args()
	savgol_test()
	# transform_result()
