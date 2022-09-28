import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

#unpickle test
testdict10 = unpickle("./cifar-10-batches-py/test_batch")
# 10000 x 3072
nptestdata10 = testdict10[b'data']
# 10000 (1 dim)
nptestlabels10 = np.array(testdict10[b'labels'])

np.savetxt("traindata10.txt", npdata10)
np.savetxt("trainlabel10.txt", nplabels10)
np.savetxt("testdata10.txt", nptestdata10)
np.savetxt("testlabel10.txt", nptestlabels10)

#now CIFAR100
traindict100 = unpickle("./cifar-100-python/train")
print(traindict100.keys())
nptraindata100 = traindict100[b'data']
print(nptraindata100.shape)
# 10000 (1 dim)
nptrainlabels100 = np.array(traindict100[b'fine_labels'])
print(nptrainlabels100.shape)

testdict100 = unpickle("./cifar-100-python/test")
# 10000 x 3072
nptestdata100 = testdict100[b'data']
# 10000 (1 dim)
nptestlabels100 = np.array(testdict100[b'fine_labels'])

np.savetxt("traindata100.txt", nptraindata100)
np.savetxt("trainlabel100.txt", nptrainlabels100)
np.savetxt("testdata100.txt", nptestdata100)
np.savetxt("testlabel100.txt", nptestlabels100)



