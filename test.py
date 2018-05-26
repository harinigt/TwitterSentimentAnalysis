import numpy as np

train_data = np.load('data/train_encoded_array.npy')
test_data = np.loadtxt('data/test_target_array')
train_target = np.load('data/train_target_array.npy')
# print(np.shape(train_data))
print(np.shape(train_data))
print(np.shape(train_target))