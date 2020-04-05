import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

data = pandas.read_csv('bank.csv', sep=';')

# 0: Age        - Numeric
# 1: Job        - Catagorical
# 2: Marital    - Catagorical
# 3: Education  - Catagorical
# 4: Default    - Catagorical
# 5: Balance    - Numeric
# 6: Housing    - Catagorical
# 7: Loan       - Catagorical
# 8: Contact    - Ignore, not relavent
# 9: Day        - Ignore, not relavent
# 10: Month     - Ignore, not relavent
# 11: Duration  - Ignore, not useful
# 12: Campaign  - Numeric
# 13: PDays     - Numeric
# 14: Previous  - Numeric
# 15: POutcome  - Catagorical

# Gets columns stated columns; the data
# Certain columns have un-useful or irrelavent data, so they are ignored
X = data.iloc[:, [0,1,2,3,4,5,6,7,12,13,14,15]].values 
# Gets column 16 from all rows; the results
y = data.iloc[:, 16].values 

# Encoder, converts catagorical data to numerical data
labelencoder = LabelEncoder()
# Encodes the results
y = labelencoder.fit_transform(y)
# Encoder for catagorical variables
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 4] = labelencoder.fit_transform(X[:, 4])
X[:, 6] = labelencoder.fit_transform(X[:, 6])
X[:, 7] = labelencoder.fit_transform(X[:, 7])
X[:, 11] = labelencoder.fit_transform(X[:, 11])


# Splits data set into training and test data. 
#   test_size = 0.3: 30% data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Standardize all data from different columns
sc = StandardScaler()
# fit_transform calls fit() then transform(). Fit is required to run once before all transforms
X_train = sc.fit_transform(X_train)
# fit() already generated, can use just transform()
X_test = sc.transform(X_test)


# Start ANN
network = Sequential()
# Creates 4 layers; input, hidden 1, hidden 2, output
#   Input: Dimension of 12; input_dim=12. 12 variables per case in data
#   Hidden 1: Dimension of 16.
#             Activation function: relu. Cheap and fast to use, generally effective.
#   Hidden 2: Dimension of 32.
#             Activation function: relu. Cheap and fast to use, generally effective.
#   Output: Dimension of 1. Needs to be a single value for prediction
#           Activation function: sigmoid. Sigmoid is especially suited for finalizing predictions that are binary in nature.
network.add(Dense(12, activation='relu', input_dim=12))
network.add(Dense(24, activation='relu'))
network.add(Dense(36, activation='relu'))
network.add(Dense(1, activation='sigmoid'))
# Compile model
network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fitting the ANN to the Training set
fitModel = network.fit(X_train, y_train, epochs=100, batch_size=64)

# Uses trained network to predict the test data
y_pred = network.predict(X_test)
# Converts prediction odds to T/F
y_pred = (y_pred > 0.5)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Calcualte and print the accuracy
total_tests = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]
accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / total_tests
print(conf_matrix)
print("Accuracy is {}".format(accuracy))

# Create plot for accuracy (error) vs epoch
plt.plot(fitModel.history['accuracy'])
plt.title('accuracy vs epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
