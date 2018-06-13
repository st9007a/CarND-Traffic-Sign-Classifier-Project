#!/usr/bin/env python3
import sys
import pickle
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score

testing_file = 'data/test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']

def normalize(arr):
    return (arr - 128) / 128

x_test = normalize(X_test)

model = load_model(sys.argv[1])
proba = model.predict(x_test)
proba = np.argmax(proba, axis = 1)

print(accuracy_score(y_test, proba))
