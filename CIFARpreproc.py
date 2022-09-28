import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

hey= unpickle("./cifar-10-batches-py/batches.meta")
print(hey)

batchdict = unpickle("./cifar-10-batches-py/data_batch_1")
print('keys',batchdict.keys())
npdata = batchdict[b'data']
print(npdata, type(npdata), npdata.shape)
nplabels = batchdict[b'labels']
print(type(nplabels), len(nplabels))
