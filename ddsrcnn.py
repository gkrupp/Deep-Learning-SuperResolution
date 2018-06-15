from base import SRBase
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Average, Add

class DSRCNN(SRBase):
	
	def __init__(self):
		super().__init__()
		
		# data
		self.input_db = '64x64lanczos'
		self.output_db = '64x64'

		# model
		inlay = Input(shape=(64,64,3))
		c_1_1 = Conv2D(64, 3, activation='relu', padding='same')(inlay)
		c_1_2 = Conv2D(64, 3, activation='relu', padding='same')(c_1_1)
		#2
		m_2_1 = MaxPooling2D(2)(c_1_2)
		c_2_2 = Conv2D(128, 3, activation='relu', padding='same')(m_2_1)
		c_2_3 = Conv2D(128, 3, activation='relu', padding='same')(c_2_2)
		#3
		x_3_1 = MaxPooling2D(2)(c_2_3)
		c_3_2 = Conv2D(256, 3, activation='relu', padding='same')(x_3_1)
		u_3_3 = UpSampling2D(2)(c_3_2)
		c_3_4 = Conv2D(128, 3, activation='relu', padding='same')(u_3_3)
		c_3_5 = Conv2D(128, 3, activation='relu', padding='same')(c_3_4)
		#/3
		m_2_4 = Add()([c_3_5, c_2_3])
		u_2_5 = UpSampling2D()(m_2_4)
		c_2_6 = Conv2D(64, 3, activation='relu', padding='same')(u_2_5)
		c_2_7 = Conv2D(64, 3, activation='relu', padding='same')(c_2_6)
		#/2
		m_1_3 = Add()([c_2_7, c_1_2])
		outlay = Conv2D(3, 5, activation='linear', padding='same')(m_1_3)
		#
		self.model = Model(inputs=inlay, outputs=outlay)

# SRCNN
net = DSRCNN()
net.run()