#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
import pickle

from absl import flags

from dataclasses import dataclass, field, InitVar
from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# classify CIFAR10
# achieve performance similar to state of the art (99.5%)

#data recieved using https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

#method to unpickle data into dicts
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
#This was a big problem in my original model. I was reshaping my data incorrectly and the max my model would ever reach was 66% Thank you Husam for helping me reshape my data!
def rgb_stack(imagedata):
    r= imagedata[:, 0:1024].reshape(-1,32,32)
    g = imagedata[:, 1024:2048].reshape(-1,32,32)
    b = imagedata[:, 2048:].reshape(-1, 32,32)

    stackrgb = np.stack([r,g,b], axis= -1)
    print(stackrgb.shape)
    return stackrgb

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
        stacked = rgb_stack(self.itrain)
        self.train = stacked[:40000].astype('float32')/255.0
        self.train_labels = self.itrainlab[:40000]

        self.val = stacked[40000:].astype('float32')/255.0
        self.val_labels = self.itrainlab[40000:]

        teststacked = rgb_stack(self.itest)
        self.test = teststacked.astype('float32')/255.0
        self.test_labels = self.itestlab

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 50000, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 128, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 5000, "Number of forward/backward pass iterations")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate/initial step size")
flags.DEFINE_integer("random_seed", 31415, "Random seed for reproducible results")
flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")

#residual network built off https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
#take input tensor and add relu and batch norm
def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

#take input tensor x pass thorugh conv to get y.
#adds x to y
#add relu and batch norm and returns that tensor

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    print(type(x))
    y = Conv2D(kernel_size = kernel_size,
            strides = (1 if not downsample else 2),
            filters = filters,
            padding = "same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
            strides = 1,
            filters = filters,
            padding = "same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                strides = 2,
                filters = filters,
                padding = "same")(x)
    out = Add()([x,y])
    out = relu_bn(out)
    return out

def create_res_net():
    inputs = Input(shape=(32,32,3))
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size =3,
            strides = 1,
            filters = num_filters,
            padding = "same")(t)
    t = relu_bn(t)

    num_blocks_list = [2,5,5,2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters = num_filters)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(100, activation = 'softmax')(t)

    model = Model(inputs, outputs)

    model.compile(
            optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
            )

    return model


def main():

    #parse flags before we use them
    FLAGS(sys.argv)
    batch_size = FLAGS.batch_size
    #set seed for reproducible results
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2) #spawn 2 sequences for 2 threads
    np_rng =np.random.default_rng(np_seed)

    #data from CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    #unpickle all training
    traindict100 = unpickle("./cifar-100-python/train")
    nptraindata100 = traindict100[b'data']
    nptrainlabels100 = np.array(traindict100[b'fine_labels'])


    #unpickle all test
    testdict100 = unpickle("./cifar-100-python/test")
    nptestdata100 = testdict100[b'data']
    nptestlabels100 = np.array(testdict100[b'fine_labels'])
    #call Data class to properly shape data
    data = Data(rng = np_rng, itrain = nptraindata100, itrainlab = nptrainlabels100, itest = nptestdata100, itestlab = nptestlabels100)

    #data augmentation https://aigeekprogrammer.com/convolutional-neural-network-4-data-augmentation/
    #Thank you Lucia for sending me the link!
    #make generator for training data
    datagen = ImageDataGenerator(rotation_range = 10,
            horizontal_flip = True,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            )

    #make generator object for validation data does nothing
    valgen = ImageDataGenerator()

    model = create_res_net()
    print(model.summary())

    train_generator = datagen.flow(data.train, data.train_labels, batch_size=batch_size)
    val_generator = valgen.flow(data.val, data.val_labels, batch_size=batch_size)
    history = model.fit(train_generator,
            steps_per_epoch = len(data.train)//batch_size,
            epochs=30,
            batch_size=batch_size,
            validation_data=val_generator,
            validation_freq =1,
            validation_steps = data.val.shape[0]//batch_size,
            verbose = 1)

    #PLOTTING
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.ylim([0, 4])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('./epochloss100.pdf')

    test_loss, test_acc, test_sparse = model.evaluate(data.test, data.test_labels, verbose=2)


if __name__ == "__main__":
    main()
