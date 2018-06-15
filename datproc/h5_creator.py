import os
import h5py
import argparse
import numpy as np
import random
from scipy import misc


# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('dbs', metavar='dbs', nargs='+')
parser.add_argument('-shuffle', default=40, required=False)
parser.add_argument('-normalize', action='store_true')
parser.add_argument('-b', type=int, default=20000, required=False)
parser.add_argument('-name', default='db.h5', required=True)
parser.add_argument('-maxn', type=int, default=None, required=False)
args = parser.parse_args()
print('[ info ] starting..')


# process dir
def create_data_file(hdf5_file, dir, size):
	i = 0;
	batch = args.b
	NPARR = np.zeros((len(ARR),size,size,3), dtype=np.uint8)
	for fn in ARR:
		if (i % batch == 0):
			print('[ ok ]', dir, '/', i)
		IMG = misc.imread(os.path.join(dir,fn))
		if (len(IMG.shape) == 2):
			IMG = np.stack((IMG,)*3, -1)
		elif (len(IMG.shape) == 3 and IMG.shape[2] == 4):
			IMG = IMG[:,:,:3]
		# error
		if (IMG.shape != (size,size,3)):
			print('[ ERR ] shape mismatch: '+fn+' '+str(IMG.shape))
		# assign
		NPARR[i] = IMG.reshape(1,size,size,3)
		i += 1
	# write/normalize
	if (args.normalize):
		print('[ info ] normalizing..')
		h5f.create_dataset(dir, dtype=np.float32, shape=NPARR.shape)
		chunk_num = (NPARR.shape[0]) // batch
		for i in range(chunk_num+1):
			if (i < chunk_num):
				converted = NPARR[i*batch:(i+1)*batch].astype(np.float32)/255
			else:
				converted = NPARR[i*batch:].astype(np.float32)/255
			h5f[dir][i*batch:i*batch+converted.shape[0]] = converted
			print('[ ok ]', dir, '/', i*batch)
	else:
		h5f.create_dataset(dir, data=NPARR)


# PREPARE
ARR = []
shuffle = True
try: _tmp = int(args.shuffle)
except ValueError: shuffle = False
#
print('[ info ] createing file list..')
if (shuffle):
	maxn = 0;
	for filename in os.listdir(args.dbs[0]):
		ARR.append(filename)
		if (args.maxn and ++maxn == args.maxn): break
	N = len(ARR)
	if (args.shuffle > 0):
		print('[ info ] shuffling file order..')
		for i in range(args.shuffle):
			random.shuffle(ARR)
else:
	with open(args.shuffle) as f:
		ARR = [x.strip() for x in f.readlines()]
		if (args.maxn):
			ARR = ARR[0:args.maxn]
		f.close()


# CREATE DATA FILE
print('[ info ] processing..')
h5f = h5py.File(args.name, 'w')
h5f.create_dataset('order', data=[a.encode('ascii', 'ignore') for a in ARR])
for db in args.dbs:
	print('[ info ]', '"'+db+'"', 'processing..')
	create_data_file(h5f, db, int(db.split('x')[0]))
	print('[ ok ]', '"'+db+'"', 'done')
h5f.close()


# TEST
print('__________________________________________________')
with h5py.File(args.name, 'r') as h5f:
	for key in h5f.keys():
		print(key, ' ', h5f[key].dtype, ' ', h5f[key].shape)
	h5f.close()
print('__________________________________________________')

