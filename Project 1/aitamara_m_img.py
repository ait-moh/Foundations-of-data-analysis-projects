# Solution for task 2 (Image Classifier) of lab assignment for FDA WS22 by [NAME]
# imports here
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# define additional functions here


def train_predict(X_train, y_train, X_test):
    # check that the input has the correct shape
    assert X_train.shape == (len(X_train), 784)
    assert y_train.shape == (len(y_train), 1)
    assert X_test.shape == (len(X_test), 784)

    # --------------------------
    # add your data preprocessing, model definition, training and prediction between these lines
    #data preprocessing-----------------------------------------
    #if the inputs are not ndarray we changed them
    if str(type(X_train))!= "<class 'numpy.ndarray'>" :
        X_train = X_train.to_numpy()
        
    if str(type(y_train))!= "<class 'numpy.ndarray'>" :
        y_train = y_train.to_numpy()
        
    if str(type(X_test))!= "<class 'numpy.ndarray'>"  :
        X_test = X_test.to_numpy()

    # Changing string labels to float        
    enc = OrdinalEncoder()
    y_train = enc.fit_transform(y_train) 
    
    #reshaping the inputs
    X_train = X_train.reshape(len(X_train),28,28,1)
    X_test = X_test.reshape(len(X_test),28,28,1)
    y_train = y_train.reshape(-1,)
    

    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    #training---------------------------------------------------
    #I inspired my model from this video : https://www.youtube.com/watch?v=7HPwo4wnJeA&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=24
    cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # I add the dropout to minimize the overfitting
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),   
    layers.Dense(94, activation='softmax')
    ])
    
    cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    # training the model
    cnn.fit(X_train, y_train, batch_size=32, epochs=6,validation_split=0.2)

    
    
    #prediction------------------------------------------------
    #transforming the predected probabilities to predected y
    y_pred = np.argmax(cnn.predict(X_test), axis=1)

    y_pred= y_pred.reshape(-1, 1)
    # transform y_pred labels to string again
    y_pred = enc.inverse_transform(y_pred)

    # --------------------------

    # test that the returned prediction has correct shape
    assert y_pred.shape == (len(X_test),) or y_pred.shape == (len(X_test), 1)

    return y_pred


if __name__ == "__main__":
    # load data (please load data like this and let every processing step happen inside the train_predict function)
    # (change path if necessary)
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.1, random_state=48)
    prediction = train_predict(X_train,y_train,X_test)

    print(accuracy_score(y_test, prediction))



