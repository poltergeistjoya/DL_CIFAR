import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

npdata = np.empty((0,3072))
for  i in range(0,5):
    batchdict = unpickle("./cifar-10-batches-py/data_batch_" + str(i+1))
    npdata = np.append(npdata, batchdict[b'data'], axis=0)
    #nplabels = batchdict[b'labels']
    #print(type(nplabels), len(nplabels))

print(npdata.shape)
