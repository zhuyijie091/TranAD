import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB','CCFD','PSM']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)


def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	if dataset == 'synthetic':
		train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
		test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
		dat = pd.read_csv(train_file, header=None)
		split = 10000
		train = normalize(dat.values[:, :split].reshape(split, -1))
		test = normalize(dat.values[:, split:].reshape(split, -1))
		lab = pd.read_csv(test_labels, header=None)
		lab[0] -= split
		labels = np.zeros(test.shape)
		for i in range(lab.shape[0]):
			point = lab.values[i][0]
			labels[point-30:point+30, lab.values[i][1:]] = 1
		test += labels * np.random.normal(0.75, 0.1, test.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'SMD':
		dataset_folder = 'data/SMD'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
	elif dataset == 'UCR':
		dataset_folder = 'data/UCR'
		file_list = os.listdir(dataset_folder)
		for filename in file_list:
			if not filename.endswith('.txt'): continue
			vals = filename.split('.')[0].split('_')
			dnum, vals = int(vals[0]), vals[-3:]
			vals = [int(i) for i in vals]
			temp = np.genfromtxt(os.path.join(dataset_folder, filename),
								dtype=np.float64,
								delimiter=',')
			min_temp, max_temp = np.min(temp), np.max(temp)
			temp = (temp - min_temp) / (max_temp - min_temp)
			train, test = temp[:vals[0]], temp[vals[0]:]
			labels = np.zeros_like(test)
			labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
	elif dataset == 'NAB':
		dataset_folder = 'data/NAB'
		file_list = os.listdir(dataset_folder)
		with open(dataset_folder + '/labels.json') as f:
			labeldict = json.load(f)
		for filename in file_list:
			if not filename.endswith('.csv'): continue
			df = pd.read_csv(dataset_folder+'/'+filename)
			vals = df.values[:,1]
			labels = np.zeros_like(vals, dtype=np.float64)
			for timestamp in labeldict['realKnownCause/'+filename]:
				tstamp = timestamp.replace('.000000', '')
				index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
				labels[index-4:index+4] = 1
			min_temp, max_temp = np.min(vals), np.max(vals)
			vals = (vals - min_temp) / (max_temp - min_temp)
			train, test = vals.astype(float), vals.astype(float)
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			fn = filename.replace('.csv', '')
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
	elif dataset == 'MSDS':
		dataset_folder = 'data/MSDS'
		df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
		df_test  = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
		df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
		_, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
		train, _, _ = normalize3(df_train, min_a, max_a)
		test, _, _ = normalize3(df_test, min_a, max_a)
		labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
		labels = labels.values[::1, 1:]
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
	elif dataset == 'SWaT':
		dataset_folder = 'data/SWaT'
		file = os.path.join(dataset_folder, 'series.json')
		df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
		df_test  = pd.read_json(file, lines=True)[['val']][7000:12000]
		train, min_a, max_a = normalize2(df_train.values)
		test, _, _ = normalize2(df_test.values, min_a, max_a)
		labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset in ['SMAP', 'MSL']:
		dataset_folder = 'data/SMAP_MSL'
		file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
		values = pd.read_csv(file)
		values = values[values['spacecraft'] == dataset]
		filenames = values['chan_id'].values.tolist()
		for fn in filenames:
			train = np.load(f'{dataset_folder}/train/{fn}.npy')
			test = np.load(f'{dataset_folder}/test/{fn}.npy')
			train, min_a, max_a = normalize3(train)
			test, _, _ = normalize3(test, min_a, max_a)
			np.save(f'{folder}/{fn}_train.npy', train)
			np.save(f'{folder}/{fn}_test.npy', test)
			labels = np.zeros(test.shape)
			indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
			indices = indices.replace(']', '').replace('[', '').split(', ')
			indices = [int(i) for i in indices]
			for i in range(0, len(indices), 2):
				labels[indices[i]:indices[i+1], :] = 1
			np.save(f'{folder}/{fn}_labels.npy', labels)
	elif dataset == 'WADI':
		dataset_folder = 'data/WADI'
		ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
		train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
		test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
		train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
		train.fillna(0, inplace=True); test.fillna(0, inplace=True)
		test['Time'] = test['Time'].astype(str)
		test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
		labels = test.copy(deep = True)
		for i in test.columns.tolist()[3:]: labels[i] = 0
		for i in ['Start Time', 'End Time']: 
			ls[i] = ls[i].astype(str)
			ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
		for index, row in ls.iterrows():
			to_match = row['Affected'].split(', ')
			matched = []
			for i in test.columns.tolist()[3:]:
				for tm in to_match:
					if tm in i: 
						matched.append(i); break			
			st, et = str(row['Start Time']), str(row['End Time'])
			labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
		train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
		print(train.shape, test.shape, labels.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'MBA':
		dataset_folder = 'data/MBA'
		ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
		train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
		test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
		train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		print(ls)
		ls = ls.values[:,1].astype(int)
		print("标签形状", ls.shape)
		labels = np.zeros_like(test)
		print("np.zeros_like(test).shape:", labels.shape)
		for i in range(-20, 20):
			labels[ls + i, :] = 1
		print(labels[:100])
		print("labels shape:", labels.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
		print("---------------")
		print(test)
	elif dataset == "ADSB":
		dataset_folder = 'data/ADSB'
		train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
		test = pd.read_csv(os.path.join(dataset_folder, 'test-ghost.csv'))
		labels = test.values[1:, -1].reshape(-1, 1)
		labels = (labels == True).astype(int)
		labels = np.tile(labels, (1, 5))
		train, test = train.values[1:8000,1:].astype(float), test.values[1:,1:-1].astype(float) # test.values[1:,1:-1] 最后一列为标签，因此这里忽略。
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		# 保存训练、测试数据和标签为 .npy 文件
		for file, data in zip(['train', 'test', 'labels'], [train, test, labels]):
			np.save(os.path.join(folder, f'{file}.npy'), data)
	elif dataset == 'CCFD':
		dataset_folder = 'data/CCFD'
		# 读取CSV文件
		df = pd.read_excel(os.path.join(dataset_folder, 'creditcard.xlsx'))

		# 分离特征和标签
		features = df.iloc[:, 1:-1]  # 忽略Time列，保留V1-V28和Amount
		labels = df['Class'].values

		# 划分训练集（仅正常数据）和测试集（正常+异常）
		normal_idx = np.where(labels == 0)[0]
		abnormal_idx = np.where(labels == 1)[0]

		# 使用80%的正常数据作为训练集
		split_idx = int(len(normal_idx) * 0.8)
		train_idx = normal_idx[:split_idx]
		test_idx = np.concatenate([normal_idx[split_idx:], abnormal_idx])

		# 提取训练集和测试集数据
		train_data = features.iloc[train_idx].values
		test_data = features.iloc[test_idx].values
		test_labels = labels[test_idx]

		# 使用训练集的统计量进行归一化
		min_val = np.min(train_data, axis=0)
		max_val = np.max(train_data, axis=0)
		range_val = max_val - min_val + 1e-12  # 避免除以零

		train_data = (train_data - min_val) / range_val
		test_data = (test_data - min_val) / range_val

		# 重塑标签形状以匹配其他数据集格式 (n_samples, n_features)
		test_labels = np.tile(test_labels.reshape(-1, 1), (1, train_data.shape[1]))

		print(
			f"CCFD数据集 - 训练集形状: {train_data.shape}, 测试集形状: {test_data.shape}, 标签形状: {test_labels.shape}")

		# 保存处理后的数据
		np.save(os.path.join(folder, 'train.npy'), train_data.astype('float32'))
		np.save(os.path.join(folder, 'test.npy'), test_data.astype('float32'))
		np.save(os.path.join(folder, 'labels.npy'), test_labels.astype('float32'))

	elif dataset == 'PSM':
		dataset_folder = 'data/PSM'

		# 1. 读取训练数据
		train_file = os.path.join(dataset_folder, 'train.csv')
		if not os.path.exists(train_file):
			raise FileNotFoundError(f"PSM训练文件不存在: {train_file}")

		train_df = pd.read_csv(train_file)

		# 自动检测特征列
		feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
		if not feature_cols:
			feature_cols = [col for col in train_df.columns if col not in ['timestamp_(min)', 'label']]

		# 2. 数据清洗 - 处理缺失值
		# 检查并处理NaN值
		nan_counts = train_df[feature_cols].isna().sum()
		if nan_counts.sum() > 0:
			print(f"警告: 训练数据中存在 {nan_counts.sum()} 个NaN值")
			# 用前一行的值填充NaN
			train_df[feature_cols] = train_df[feature_cols].fillna(method='ffill')
			# 如果第一行有NaN，用后一行的值填充
			train_df[feature_cols] = train_df[feature_cols].fillna(method='bfill')

		# 3. 读取测试数据和标签
		test_file = os.path.join(dataset_folder, 'test.csv')
		if not os.path.exists(test_file):
			raise FileNotFoundError(f"PSM测试文件不存在: {test_file}")

		test_df = pd.read_csv(test_file)

		# 检查并处理测试数据中的NaN值
		nan_counts_test = test_df[feature_cols].isna().sum()
		if nan_counts_test.sum() > 0:
			print(f"警告: 测试数据中存在 {nan_counts_test.sum()} 个NaN值")
			test_df[feature_cols] = test_df[feature_cols].fillna(method='ffill')
			test_df[feature_cols] = test_df[feature_cols].fillna(method='bfill')

		label_file = os.path.join(dataset_folder, 'test_label.csv')
		if not os.path.exists(label_file):
			raise FileNotFoundError(f"PSM标签文件不存在: {label_file}")

		label_df = pd.read_csv(label_file)

		# 4. 对齐测试数据和标签
		merged_test = pd.merge(test_df, label_df, on='timestamp_(min)', how='inner')

		if len(merged_test) != len(test_df) or len(merged_test) != len(label_df):
			print(f"警告: 数据不匹配 - 测试数据: {len(test_df)}, 标签: {len(label_df)}, 交集: {len(merged_test)}")

		# 5. 提取特征和标签
		train_data = train_df[feature_cols].values
		test_data = merged_test[feature_cols].values
		test_labels = merged_test['label'].values

		# 6. 数据验证
		print(f"训练集形状: {train_data.shape}, 测试集形状: {test_data.shape}, 标签形状: {test_labels.shape}")
		print(f"异常比例: {test_labels.mean():.4f}")

		# 7. 数据稳定性检查
		# 检查是否存在无穷大或NaN值
		if np.any(np.isinf(train_data)) or np.any(np.isnan(train_data)):
			print("训练数据中存在无穷大或NaN值，进行清理...")
			train_data = np.nan_to_num(train_data, nan=0.0, posinf=np.finfo(np.float32).max,
									   neginf=np.finfo(np.float32).min)

		if np.any(np.isinf(test_data)) or np.any(np.isnan(test_data)):
			print("测试数据中存在无穷大或NaN值，进行清理...")
			test_data = np.nan_to_num(test_data, nan=0.0, posinf=np.finfo(np.float32).max,
									  neginf=np.finfo(np.float32).min)

		# 8. 改进的归一化处理
		# 使用RobustScaler处理异常值
		from sklearn.preprocessing import RobustScaler

		scaler = RobustScaler()
		train_data = scaler.fit_transform(train_data)
		test_data = scaler.transform(test_data)

		# 9. 添加特征平滑（可选）
		# 对时间序列数据进行滑动平均平滑
		window_size = 5

		def smooth_features(data, window_size):
			smoothed = np.zeros_like(data)
			for i in range(data.shape[1]):
				smoothed[:, i] = np.convolve(data[:, i], np.ones(window_size) / window_size, mode='same')
			return smoothed

		train_data = smooth_features(train_data, window_size)
		test_data = smooth_features(test_data, window_size)

		# 10. 重塑标签形状
		test_labels = np.tile(test_labels.reshape(-1, 1), (1, train_data.shape[1]))

		print(f"PSM预处理完成: 训练集: {train_data.shape}, 测试集: {test_data.shape}, 标签: {test_labels.shape}")

		# 11. 保存处理后的数据
		np.save(os.path.join(folder, 'train.npy'), train_data.astype('float32'))
		np.save(os.path.join(folder, 'test.npy'), test_data.astype('float32'))
		np.save(os.path.join(folder, 'labels.npy'), test_labels.astype('float32'))

	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")