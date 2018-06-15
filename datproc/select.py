import os
from PIL import Image

SOURCE = './files2'
DESTINATION = './min64'
MINRES = 64;

for filename in os.listdir(SOURCE):
	file = os.path.join(SOURCE, filename)
	with Image.open(file) as img:
		width, height = img.size
		if (width >= MINRES and height >= MINRES):
			os.rename(file, os.path.join(DESTINATION, filename));