#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
import pickle

from absl import flags

from dataclasses import dataclass, field, InitVar
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

# classify CIFAR10
# achieve performance similar to state of the art (99.5%)

#data recieved using https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

#method to unpickle data into dicts
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

@dataclass
class Data:
    rng: InitVar[np.random.Generator]
    itrain: np.ndarray
    itrainlab: np.ndarray
    itest: np.ndarray
    itestlab: np.ndarray

    #Training set
    train: np.ndarray = field(init=False)
    train_labels: np.ndarray = field(init=False)

    #Validation set
    val:np.ndarray = field(init=False)
    val_labels: np.ndarray = field(init=False)

    #Test Set
    test: np.ndarray = field(init=False)
    test_labels: np.ndarray = field(init=False)

    def __post_init__(self,rng):
        self.train = self.itrain[:40000].reshape(-1,32,32,3).astype('float32')/255.0
        self.train_labels = self.itrainlab[:40000]

        self.val = self.itrain[40000:].reshape(-1,32,32,3).astype('float32')/255.0
        self.val_labels = self.itrainlab[40000:]

        self.test = self.itest.reshape(-1,32,32,3).astype('float32')/255.0
        self.test_labels = self.itestlab

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 50000, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 50, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 5000, "Number of forward/backward pass iterations")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate/initial step size")
flags.DEFINE_integer("random_seed", 31415, "Random seed for reproducible results")
flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")

flags.DEFINE_integer("num_epochs", 70, "Number of epochs")
FLAGS(sys.argv)
epochs = FLAGS.num_epochs
#learning rate decay
def poly_decay(epoch):
    maxEpochs = epochs
    baseLR = 0.005
    power = 1.0
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha

#CNN made based off of https://www.tensorflow.org/tutorials/images/cnn
def Model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same',input_shape = (32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32,(3,3), activation = 'relu',padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0))
    model.add(layers.Conv2D(64,(3,3), activation = 'relu',padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0))
    model.add(layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0))

    #add Dense layers for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    # size 10 output layer for 10 classes
    model.add(layers.Dense(10, activation='softmax', activity_regularizer = regularizers.L2(0.001)))

    #add dropout layer to prevent overfitting, higher rate means more parameters are dropped out
    model.add(layers.Dropout(.4))
    callbacks = [LearningRateScheduler(poly_decay)]
    #model has optimzeer and loss function built in now
    model.compile(optimizer=Adam(lr=0.005),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

    return model

def main():

    #parse flags before we use them

    #set seed for reproducible results
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2) #spawn 2 sequences for 2 threads
    np_rng =np.random.default_rng(np_seed)

    #data from CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    #unpickle all training
    npdata10 = np.empty((0,3072))
    nplabels10 = np.empty((0))
    for  i in range(0,5):
        batchdict = unpickle("./cifar-10-batches-py/data_batch_" + str(i+1))
        npdata10 = np.append(npdata10, batchdict[b'data'], axis=0)
        trainlabels = np.array(batchdict[b'labels'])
        nplabels10= np.append(nplabels10, trainlabels, axis = 0)

    #npdata10 is now 50000 x 3072
    #nplabels10 is 50000 (1 dim)
    #add dim to np labels
    #nplabels10 = np.expand_dims(nplabels10, axis = 1)

    #unpickle test
    testdict10 = unpickle("./cifar-10-batches-py/test_batch")
    # 10000 x 3072
    nptestdata10 = testdict10[b'data']
    # 10000 (1 dim)
    #nptestlabels10 = np.expand_dims(np.array(testdict10[b'labels']), axis = 1)
    nptestlabels10 = np.array(testdict10[b'labels'])
    #call Data class to properly shape data
    data = Data(rng = np_rng, itrain = npdata10, itrainlab = nplabels10, itest = nptestdata10, itestlab = nptestlabels10)

    #print(data.train.shape, data.train_labels.shape)
    model = Model()
    print(model.summary())
    history = model.fit(data.train, data.train_labels, epochs=FLAGS.num_epochs, batch_size = 256,
    validation_data=(data.val, data.val_labels))

    #PLOTTING
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.ylim([0, 7])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('./epochloss.pdf')

    test_loss, test_acc = model.evaluate(data.test, data.test_labels, verbose=2)


if __name__ == "__main__":
    main()
