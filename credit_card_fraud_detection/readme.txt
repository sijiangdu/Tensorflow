All the source code is in one file: "creditcard_with_summaries.py"

Input data are in "creditcard.csv".  it is downloaded from https://www.kaggle.com/dalpozz/creditcardfraud
The creditcard.csv needs be copied to the py folder.

This program is for Mac OS. The log folder and internal data folder are automatically created at: 
/tmp/tensorflow/creditcard/logs/creditcard_with_summaries
/tmp/tensorflow/creditcard/

The plotting result PNG image file is at:
/tmp/tensorflow/creditcard/result

For detail documentation, please see the PDF file.

Tensorflow input_pipeline is being used to streamline and randomlize inputs for the training and testing session.

To start: call the train() function, e.g. "train(5,100)", means 5 hidden layers, each layer has 100 neurons.  


