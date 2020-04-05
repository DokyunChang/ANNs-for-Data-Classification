import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

data = pandas.read_csv('data.csv')
# Gets columns 3 - 32 from all rows; the data
X = data.iloc[:, 2:32].values 
# Gets 2nd column from all rows; the results
y = data.iloc[:, 1].values 

# Encodes results. Converts results from B or M to 1 or 0
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# Splits data set into training and test data. 
#   test_size = 0.3: 30% data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Standardize all data from different columns
sc = StandardScaler()
# fit_transform calls fit() then transform(). Fit is required to run once before all transforms
X_train = sc.fit_transform(X_train)
# fit() already generated, can use just transform()
X_test = sc.transform(X_test)


# Start ANN
network = Sequential()
# Creates 4 layers; input, hidden 1, hidden 2, output
#   Input: Dimension of 30; input_dim=30. 30 variables per case in data
#   Hidden 1: Dimension of 16.
#             Activation function: relu. Cheap and fast to use, generally effective.
#   Hidden 2: Dimension of 32.
#             Activation function: relu. Cheap and fast to use, generally effective.
#   Output: Dimension of 1. Needs to be a single prediction for diagnosis
#           Activation function: sigmoid. Sigmoid is especially suited for finalizing predictions that are binary in nature.
network.add(Dense(16, activation='relu', input_dim=30))
network.add(Dense(32, activation='relu'))
network.add(Dense(1, activation='sigmoid'))
# Compile model
network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fitting the ANN to the Training set
fitModel = network.fit(X_train, y_train, epochs=150, batch_size=100)

# Uses trained network to predict the test data
y_pred = network.predict(X_test)
# Converts prediction odds to T/F
y_pred = (y_pred > 0.5)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Calcualte and print the accuracy
total_tests = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]
accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / total_tests
print("Accuracy is {}".format(accuracy))

# Create plot for accuracy (error) vs epoch
plt.plot(fitModel.history['accuracy'])
plt.title('accuracy vs epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
