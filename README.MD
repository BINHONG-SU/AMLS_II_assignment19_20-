This project is implemented by Python 3.5, and this project requires a lot of libraries: numpy, pandas, re, sklearn, bs4, nltk, gensim, keras, matplotlib, tensorflow

If your computer has not download the nltk corpus, you need to use command 'nltk.donwload()' to donwload all the packages.

Please run the 'main.py' directly, it will use the data in 'temp_data' and the trained model 'BiLSTM_model.h5' to produce the training error, validation error, and testing error.

If you want to process the data personally, please follow the instructions marked in 'main.py'

If you want to train the model personally, please follow the instructions marked in 'DataProcessing.py'

The processed data are store in folder 'temp_data', they are in numerical form.

The trained language model is stores as 'BiLSTM_model.h5'.

Folder 'original_data from Kaggle' contains all the original data provided for this competition: labeledTrainData, testData, unlabeledTrainData

In folder 'useful_data' contains the data: labeledTrainData, labeledTestData, unlabeledTrainData, additionalData. The labeledTrainData and labeledTestData in this folder are divided from the labeledTrainData in 'original_data from Kaggle' folder

'Word2Vec_model' is a trained model for word embedding.

'DataProcessing.py' is used to process the raw data.

'LanguageModel.py' is used to build, train and test model.