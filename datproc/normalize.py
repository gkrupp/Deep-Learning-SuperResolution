import os
import h5py
import numpy as np


CHUNK_SIZE = 20000;

h5f_r = h5py.File('data_all_final.hdf5', 'r')
h5f_w = h5py.File('data_all_normalized.hdf5', 'w')
for dbname in list(h5f_r.keys()):
	print('')
	print(dbname)
	if (h5f_r[dbname].dtype == np.uint8):
		h5f_w.create_dataset(dbname, dtype=np.float32, shape=h5f_r[dbname].shape)
		chunk_num = (h5f_r[dbname].shape[0])//CHUNK_SIZE
		for i in range(chunk_num+1):
			print('read')
			if (i < chunk_num):
				print(i*CHUNK_SIZE, (i+1)*CHUNK_SIZE-1)
				converted = h5f_r[dbname][i*CHUNK_SIZE:(i+1)*CHUNK_SIZE].astype(np.float32)/255
			else:
				print(i*CHUNK_SIZE, ':')
				converted = h5f_r[dbname][i*CHUNK_SIZE:].astype(np.float32)/255
			print('write')
			h5f_w[dbname][i*CHUNK_SIZE:i*CHUNK_SIZE+converted.shape[0]] = converted
	else:
		print('read')
		print('write')
		h5f_w.create_dataset(dbname, data=h5f_r[dbname])
h5f_r.close()
h5f_w.close()


# CHECK
print('')
print('CHECK')
h5f = h5py.File('data_all_normalized.hdf5', 'r')
for dbname in list(h5f.keys()):
	print(dbname, h5f[dbname].shape)
h5f.close()



