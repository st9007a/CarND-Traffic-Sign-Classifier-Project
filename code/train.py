#!/usr/bin/env python3
import pickle

training_file = 'data/train.p'
validation_file = 'data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

from keras.utils import to_categorical

def normalize(arr):
    return (arr - 128) / 128

x_train = normalize(X_train)
x_valid = normalize(X_valid)
x_test = normalize(X_test)

y_train = to_categorical(y_train, num_classes = 43)
y_valid = to_categorical(y_valid, num_classes = 43)
y_test = to_categorical(y_test, num_classes = 43)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = (32, 32, 3)))
model.add(Conv2D(filters = 10, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 10, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 10, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(filters = 128, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 10, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 10, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 128, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(filters = 256, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 10, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 10, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 256, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation = 'softmax'))

model.summary()

model_ckpt = ModelCheckpoint('models/test.h5', save_best_only = True, verbose = 1, monitor = 'val_loss')

model.compile(optimizer = Adam(lr = 1e-3), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, validation_data = (x_valid, y_valid), epochs = 100, batch_size = 256, callbacks = [model_ckpt])
