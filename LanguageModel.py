import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
import matplotlib.pyplot as plt
import numpy as np

class BiLSTM:
    lstm_out = 50
    model = Sequential()    #Build the language model
    model.add(Bidirectional(LSTM(lstm_out)))
    model.add(Dense(30))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __init__(self,batch_size,n_epoch):
        self.batch_size = batch_size
        self.n_epoch = n_epoch

    def Train(self,X_train,X_val,Y_train,Y_val):
        with tf.device('/cpu:0'):   #Use CPU to train, as we do not have acess to GPU
            #========================================================================================================
            #Plot the accuracy curve and loss curve to determine the value of epoch
            #It will cost a lot of time, and the model will not be automatically saved.
            '''
            history = BiLSTM.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=50,validation_data=(X_val, Y_val))
            # Summarize history for accuracy, plot accuracy curve
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
            # Summarize history for loss, plot loss curve
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
            '''
            #===============================================================================================================
            #Use determined epoch to train the model, epoch is 18,
            #This is for training Bi-LSTM model. If you want to train the model personally, just delete these two '''.
            '''
            BiLSTM.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.n_epoch, validation_data=(X_val, Y_val))
            BiLSTM.model.save('./BiLSTM_model.h5')  #Save the trained model 
            '''
            #===============================================================================================================

            estimator = load_model('./BiLSTM_model.h5') #Load the trained Bi-LSTM model
            score_train, acc_train = estimator.evaluate(X_train, Y_train, batch_size=self.batch_size)
            score_val, acc_val = estimator.evaluate(X_val, Y_val, batch_size=self.batch_size)
        return acc_train, acc_val

    def Test(self,X_test,Y_test):
        estimator = load_model('./BiLSTM_model.h5')
        score_test, acc_test = estimator.evaluate(X_test, Y_test, batch_size=self.batch_size)
        return acc_test
