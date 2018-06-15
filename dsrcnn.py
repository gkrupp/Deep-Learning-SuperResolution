from base import SRBase
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Average, Add

class DSRCNN(SRBase):
	
	def __init__(self):
		super().__init__()
		
		# data
		self.input_db = '64x64lanczos'
		self.output_db = '64x64'
		
		# model
		inlay = Input(shape=(64,64,3))
		x_1_1 = Conv2D(64, 3, activation='relu', padding='same')(inlay)
		#2
		x_2_1 = Conv2D(64, 3, activation='relu', padding='same')(x_1_1)
		#3
		x_3_1 = Conv2DTranspose(64, 3, activation='relu', padding='same')(x_2_1)
		#/3
		x_2_2 = Add()([x_3_1, x_2_1])
		x_2_3 = Conv2DTranspose(64, 3, activation='relu', padding='same')(x_2_2)
		#/2
		x_1_2 = Add()([x_2_3, x_1_1])
		outlay = Conv2D(3, 5, activation='linear', padding='same')(x_1_2)
		#
		self.model = Model(inputs=inlay, outputs=outlay)

# SRCNN
net = DSRCNN()
net.run()