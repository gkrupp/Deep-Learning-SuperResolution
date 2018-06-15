from base import SRBase
from keras.models import Model
from keras.layers import Input, Conv2D, Average

class ESRCNN(SRBase):
	
	def __init__(self):
		super().__init__()
		
		# data
		self.input_db = '64x64lanczos'
		self.output_db = '64x64'
		
		# model
		inlay = Input(shape=(64,64,3))
		x = Conv2D(64, 9, activation='relu', padding='same')(inlay)
		x = Average()([
			Conv2D(32, 1, activation='relu', padding='same')(x),
			Conv2D(32, 3, activation='relu', padding='same')(x),
			Conv2D(32, 5, activation='relu', padding='same')(x)
		])
		outlay = Conv2D(3, 5, activation='relu', padding='same')(x)
		#
		self.model = Model(inputs=inlay, outputs=outlay)

# SRCNN
net = ESRCNN()
net.run()