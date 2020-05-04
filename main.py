import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from DataProcessing import get_data
from LanguageModel import BiLSTM
import numpy as np
import gc
# ======================================================================================================================
# Data preprocessing
# The processed data are stored in the file 'temp_data'
# If you want process the data personally, just delete the two ''', and the processed data will be saved.
'''
get_data = get_data()
X_train, X_val, X_test, Y_train, Y_val, Y_test = get_data.word2vec()

np.save("./temp_data/X_train.npy",X_train)
np.save("./temp_data/X_test.npy",X_test)
np.save("./temp_data/X_val.npy",X_val)
np.save("./temp_data/Y_train.npy",Y_train)
np.save("./temp_data/Y_test.npy",Y_test)
np.save("./temp_data/Y_val.npy",Y_val)
'''
X_train = np.load("./temp_data/X_train.npy")
X_test = np.load("./temp_data/X_test.npy")
X_val = np.load("./temp_data/X_val.npy")
Y_train = np.load("./temp_data/Y_train.npy")
Y_test = np.load("./temp_data/Y_test.npy")
Y_val = np.load("./temp_data/Y_val.npy")

# ======================================================================================================================
#Bi-LSTM model implementation
model = BiLSTM(batch_size=100,n_epoch=18)    # Build model object.
acc_train, acc_val= model.Train(X_train, X_val, Y_train, Y_val) # Train model based on the training set
acc_test = model.Test(X_test,Y_test)    # Test model based on the test set.
del X_train, X_val, X_test, Y_train, Y_val, Y_test  #Free memory
gc.collect()

# ======================================================================================================================
#Print results
print('acc_train, acc_val, acc_test :{},{},{};'.format(acc_train, acc_val,acc_test))
