"""STEP 1: Preliminary language-specific commands"""
import os
import numpy as np # same modules as before
import pandas as pd
from sklearn import neural_network, model_selection, metrics

import tensorflow as tf # for deep learning modules
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras import optimizers, metrics

"""STEP 2: Load the data"""
"""STEP 5: Shuffle the samples and split into train and test (step order based on run_ann.py)"""
# keras' built-in datasets already take care of train/test split
(train_in, train_out), (test_in, test_out) = mnist.load_data()

"""STEP 4: Scale and reshape the data"""
# Data must be four-dimensional to can work with the Keras API
train_in = train_in.reshape(train_in.shape[0], train_in.shape[1], train_in.shape[2], 1)
test_in = test_in.reshape(test_in.shape[0], test_in.shape[1], test_in.shape[2], 1)
train_in = train_in.astype("float32")
test_in = test_in.astype("float32")

# Scaling
train_in /= 255
test_in /= 255

# using 10 here because that is the number of possible classifications (10 unique digits)
train_out = np_utils.to_categorical(train_out, 10)
test_out = np_utils.to_categorical(test_out, 10)

"""STEP 3: Determine the CNN hyperparameters"""
# here, we must build each layer of the CNN
cnn = Sequential()
cnn.add(Conv2D(8, (3, 3), activation="relu", input_shape=(28, 28, 1)))
cnn.add(Conv2D(16, (3, 3), activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(32, (3, 3), activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, (3, 3), activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(128, (1, 1), activation="relu"))
cnn.add(MaxPooling2D(pool_size=(1,1)))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dense(10, activation="softmax"))

method = optimizers.SGD(learning_rate=0.01) # set training method and learning rate

"""STEP 6: Train the ANN"""
# select type of loss (cross-entropy) and metric
# F1 score is not available but you can obtain precision and recall, then calculate F1 manually
cnn.compile(optimizer=method, loss="categorical_crossentropy", metrics=["accuracy"])

cnn.fit(train_in, train_out, epochs=20, batch_size=128)

"""STEP 7: Predict training outputs"""
pred_train_out = cnn.predict(train_in)

"""STEP 8: Get the training score"""
train_score = cnn.evaluate(train_in, train_out, verbose=0)

"""STEP 9: Predict testing outputs"""
pred_test_out = cnn.predict(test_in)

"""STEP 10: Get the testing score"""
test_score = cnn.evaluate(test_in, test_out, verbose=0)

"""STEP 11: Save evaluation results and outputs to a file"""
results = np.array([["Training Loss (%): ", str(100 * train_score[0])], ["Training Accuracy (%): ", str(100 * train_score[1])], ["Testing Loss (%): ", str(100 * test_score[0])], ["Testing Score (%): ", str(100 * test_score[1])]])

results_file = pd.DataFrame(results)
# predicted values versus actual values on training data
train_compare = pd.DataFrame((np.vstack((pred_train_out,train_out))))
# predicted values versus actual values on testing data
test_compare = pd.DataFrame((np.vstack((pred_test_out,test_out))))

# filepath to "Saved Files" folder
savedir = "Saved Files" + os.sep
# export evaluation results
results_file.to_csv(savedir + "score.csv", index = False, header = False)
# export training outputs
train_compare.to_csv(savedir + "Predicted Training Outputs.csv", index = False, header = None)
# export test outputs
test_compare.to_csv(savedir + "Predicted Test Outputs.csv", index = False, header = None)

"""STEP 12: Display results to the console"""
print("Deep Neural Network (DNN) Implementation","\n-------------------------------------------")
#print('Average training loss: ',"%.2f" %(100 * train_score[0]),
#    '%','\nAverage training accuracy: ',"%.2f" %(100 * train_score[1]),'%')
#print('Average testing loss: ',"%.2f" %(100 * test_score[0]),
#    '%','\nAverage testing accuracy: ',"%.2f" %(100 * test_score[1]),'%')

for elt in results: print(*elt, sep="\n")    