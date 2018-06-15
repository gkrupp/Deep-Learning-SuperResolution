import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

model = load_model('models/srcnn.h5')
for layer in model.layers:
	weights = layer.get_weights() # list of numpy arrays
	shape = weights[0].shape
	if (shape[3] < 10): continue;
	print(shape)
	conf_arr = np.swapaxes(np.swapaxes(weights[0], 3, 0), 1, 2)[15][0]
	print(conf_arr.shape)

	norm_conf = []
	for i in conf_arr:
		a = 0
		tmp_arr = []
		a = sum(i, 0)
		for j in i:
			tmp_arr.append(float(j)/float(a))
		norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
					interpolation='nearest')

	width, height = conf_arr.shape


cb = fig.colorbar(res)
plt.show()