#import modules and libraries
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
import os
from keras.utils import to_categorical

def train_model(X_train, y_train):
    print(X_train.shape)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train[0].shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=8)
    return model

#To evaluate the model
def detect_spoofing(fingerprint, model, threshold):
    prob_spoofing = model.predict(np.expand_dims(fingerprint, axis=0))
    if prob_spoofing > threshold:
        return True  
    else:
        return False  

#To scrape out the images from the directory
img_data = []
labels = ["genuine","spoofed"]
img_labels = []
img_classes = 2       
curr_path = r"C:\Users\thisi\Desktop\mini final"#Retrieving the images and their labels 
for i in range(img_classes):
    path = os.path.join(curr_path,labels[i])
    print(path)
    train_img = os.listdir(path)
    for a in train_img:
        try:
            img = Image.open(path + '\\'+ a)
            img = img.convert('RGB')
            img = np.array(img)                    #converts the image data to array
            img_data.append(img)
            img_labels.append(i)
        except Exception as e:
            print("Error loading image",e)
img_data = np.array(img_data)                      #images of different classes
img_labels = np.array(img_labels)                  #folder containing images based on classification
print(img_data.shape, img_labels.shape)            #dimension of the array

X_train, X_test, y_train, y_test = train_test_split(img_data, img_labels, test_size=0.2, random_state=0)#splitiing the data into test and train
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalizing the data to help with the training
X_train /= 255
X_test /= 255
#one hot encoding
y_train = to_categorical(y_train,img_classes)
y_test = to_categorical(y_test,img_classes)
print(y_train)
model = train_model(X_train, y_train)#train the model
model.save("mdl.h5")#save the model

