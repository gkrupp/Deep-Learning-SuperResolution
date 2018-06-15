import os
import sys
import h5py
import numpy as np
import argparse
from PIL import Image

from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, LambdaCallback


class SRBase:

	def __init__(self):
		
		self.modelname = sys.argv[0].split('/')[-1].split('.')[0]
		
		# args
		parser = argparse.ArgumentParser()
		parser.add_argument('-train', action='store_true')
		parser.add_argument('-predindex', type=int, nargs='+', required=False)
		parser.add_argument('-n', type=int, default=10000, required=False)
		parser.add_argument('-v', type=int, default=1000, required=False)
		parser.add_argument('-b', type=int, default=128, required=False)
		parser.add_argument('-data', default='data/data.h5', required=False)
		parser.add_argument('-model', default='models/'+self.modelname+'.h5', required=False)
		parser.add_argument('-chkimg', type=int, nargs='+', default=None, required=False)
		parser.add_argument('-imgpath', default='./', required=False)
		self.args = parser.parse_args()
		
		# data
		self.input_db = '64x64lanczos'
		self.output_db = '64x64'
		self.x_train = None
		self.y_train = None
		self.x_test  = None
		self.y_test  = None
		
		# helper
		self.callbacks = [
			EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0),
			LearningRateScheduler(self.LRScheduler),
			ModelCheckpoint(self.args.model, save_best_only=True)
		]
		if (self.args.chkimg != None):
			self.callbacks.append(LambdaCallback(on_epoch_end=self.SaveChkImg))
		self.model = None;
		
	
	def loadData(self):
		h5f = h5py.File(self.args.data, 'r')
		self.x_train = h5f[self.input_db][0:self.args.n]
		self.y_train = h5f[self.output_db][0:self.args.n]
		self.x_test = h5f[self.input_db][-1*self.args.v:]
		self.y_test = h5f[self.output_db][-1*self.args.v:]
		h5f.close()
	
	def printDataShape(self):
		print('x_train', self.input_db, self.x_train.shape)
		print('y_train', self.input_db, self.y_train.shape)
		print('x_test', self.output_db, self.x_test.shape)
		print('y_test', self.output_db, self.y_test.shape)

	def LRScheduler(self, epoch, lr=0.001):
		base_lr = 0.001
		return base_lr / np.sqrt(2*np.clip(epoch, 1, None))
	
	def compile(self):
		self.model.compile(optimizer='adam',
				  loss='mean_squared_error')
		self.model.summary()

	def fit(self):
		self.model.fit(self.x_train, self.y_train,
				  validation_data=(self.x_test, self.y_test),
				  batch_size=self.args.b,
				  epochs=256,
				  callbacks=self.callbacks)
		#score = self.model.evaluate(self.x_test, self.y_test, batch_size=self.args.b)
		#print('Score:', score)
		
	def save(self):
		self.model.save(self.args.model)
	
	def run(self):
		#train
		if (self.args.predindex != None):
			self.model = load_model(self.args.model)
			for i in self.args.predindex:
				self.predictIndex(i)
			print('[ ok ]', self.args.predindex, 'index saved')
		elif (self.args.train):
			if (self.x_train == None or self.y_train == None):
				self.loadData()
			self.printDataShape()
			if (os.path.isfile(self.args.model)):
				self.model = load_model(self.args.model)
			else:
				self.compile()
			self.fit()
			self.save()

	def predict(self, data):
		return self.model.predict(data)
	
	def predictOne(self, data):
		return self.predict(data.reshape((1,64,64,3)))
	
	def pred_to_imgmx(self, data):
		data = data.clip(0, 1)
		return data/np.max(data)*255
		
	def predictIndex(self, index):
		# load
		h5f = h5py.File(self.args.data, 'r')
		x32  = h5f['32x32'][index]*255
		x64  = h5f['64x64'][index]*255
		x64s = h5f['64x64lanczos'][index]*255
		pred = h5f['64x64lanczos'][index]
		h5f.close()
		# pred
		pred = self.predictOne(pred)
		pred = self.pred_to_imgmx(pred)[0]
		# save
		prefix = os.path.join(self.args.imgpath, self.modelname + '_' + str(index) + '_')
		Image.fromarray(x32.astype(np.uint8),  'RGB').save(prefix + '32.png')
		Image.fromarray(x64.astype(np.uint8),  'RGB').save(prefix + '64.png')
		Image.fromarray(x64s.astype(np.uint8), 'RGB').save(prefix + '64s.png')
		Image.fromarray(pred.astype(np.uint8), 'RGB').save(prefix + 'pred.png')
	
	def SaveChkImg(self, epoch, logs):
		for i in self.args.chkimg:
			h5f = h5py.File(self.args.data, 'r')
			data = h5f[self.input_db][i]*255
			h5f.close()
			pred = self.predictOne(data)
			pred = self.pred_to_imgmx(pred)[0]
			prefix = os.path.join(self.args.imgpath, self.modelname + '_' + str(i) + '_e' + str(epoch) + '_')
			Image.fromarray(pred.astype(np.uint8), 'RGB').save(prefix + 'pred.png')
		