import numpy as np
import os
from keras.models import load_model
from PIL import Image


model = load_model(r"mdl.h5")
classes = { 0:'real',
            1:'fake'}
i = r"genuine\10.png"

print(i)
image = Image.open(i)
image = image.convert('RGB')
image = np.expand_dims(image, axis=0)
image = np.array(image)
image = image.astype('float32')
image /= 255
print(image.shape)
print(model.predict(image))
print(np.argmax(model.predict(image),axis=-1))
