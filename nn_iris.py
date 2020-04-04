# ML of the Iris dataset

# Keras library
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import plot_model

# SKLearn dataset stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv(file_name):
    url = './datasets/' + file_name
    dataset = pd.read_csv(url)
    dataset = dataset.values
    return dataset

def preprocess_dataset(dataset):
    x = dataset[:, 0:4]
    y = dataset[:, 4]
    category_list = np.unique(y)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    #print(np.unique(y))
    y = to_categorical(y)
    #print(y[50])
    #print(np.argmax(y[50]))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test, category_list

def build_model():
    model = Sequential()
    model.add(Dense(10, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_accuracy_loss(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def main():
    file_name = 'bezdekIris.csv'
    dataset = load_csv(file_name)
    x_train, x_test, y_train, y_test, category_list = preprocess_dataset(dataset)
    model = build_model()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=10)
    plot_accuracy_loss(history)
    print(model.predict_classes(x_test))

if __name__ == "__main__":
    main()