from base import SRBase
from keras.models import Sequential
from keras.layers import Conv2D

class SRCNN(SRBase):
	
	def __init__(self):
		super().__init__()
		
		# data
		self.input_db = '64x64lanczos'
		self.output_db = '64x64'
		
		# model
		self.model = Sequential([
			Conv2D(64, 3, input_shape=(64,64,3),
				activation='relu',
				kernel_initializer='RandomNormal',
				padding='same'),
			Conv2D(32, 3,
				activation='relu',
				kernel_initializer='RandomNormal',
				padding='same'),
			Conv2D(3,  3,
				activation='relu',
				kernel_initializer='RandomNormal',
				padding='same')
		])

# SRCNN
net = SRCNN()
net.run()