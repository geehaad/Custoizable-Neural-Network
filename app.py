import streamlit as st

st.title("Custoizable Neural Network")

#Create the sidebars in the webpage to let users change num_neurons, num_epochs, activation function
num_neurons = st.sidebar.slider("Number of neurons in hidden layer:", 1, 66)
num_epochs = st.sidebar.slider("Number of epochs", 1, 10)
activation = st.sidebar.text_input("Activation function")

#When click the button, do the following
if st.button('Train the model'):
    #Import nucessarly packages
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Sequential 
    from tensorflow.keras.callbacks import ModelCheckpoint

    #Preprocessing function to normalize normalize the values to range from 0 to 1
    def preprocess_image(images):
      images = images / 255
      return images

    #Split the data set
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #apply preprocess_image on training and testing data set
    X_train = preprocess_image(X_train)
    X_test = preprocess_image(X_test)

    #Create the model 
    model = Sequential()
    model.add(InputLayer((28,28)))
    model.add(Flatten())
    model.add(Dense(num_neurons, activation))
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    #Save the best model 
    save_cp = ModelCheckpoint('model', save_best_only = True)
    history_cp = tf.keras.callbacks.CSVLogger('history.csv', separator=',')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, callbacks=[save_cp, history_cp])

#When click the button, do the following
if st.button('Evaluate the model'):

    import pandas as pd 
    import matplotlib.pyplot as plt

    #Draw Model accuracy vs epochs graph of train and validation set
    history = pd.read_csv('history.csv')
    fig = plt.figure()
    plt.plot(history['epoch'], history['accuracy'])
    plt.plot(history['epoch'], history['val_accuracy'])
    plt.title('Model accuracy vs epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    fig