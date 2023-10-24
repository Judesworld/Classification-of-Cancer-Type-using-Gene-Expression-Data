'''
Jude Tear
Student Number: 20128768
Sunday Oct 8 2023
'''

import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Filter specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# Reset seed to try and ensure reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


'''
Helper Function #1 - Report performance and print
'''
def calc_performance_and_print(confusion_matrix, name):
    # Extract values from the confusion matrix
    tn, fp, fn, tp = confusion_matrix.ravel()
    
    # Calculate accuracy, sensitivity (true positive rate), and specificity (true negative rate)
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100

    # Print the results
    print(f"{name} Accuracy: {accuracy:.2f}%")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.2f}%")
    print(f"Specificity (True Negative Rate): {specificity:.2f}%\n")



'''
Helper Function #2 - Plot confusion matrix
'''
def plot_confusion_matrix(confusion_matrix, name):
    # Plot confusion matrix using heatmap() and make available as helper function
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()



'''
Helper Function #3 - Plot ANN history
'''
def plot_ann_history(ann_history, title):
    # Extract training and validation data from history
    training_loss = ann_history.history['loss']
    training_accuracy = ann_history.history['accuracy']
    validation_loss = ann_history.history['val_loss']
    validation_accuracy = ann_history.history['val_accuracy']
    epochs = range(1, len(training_loss) + 1)

    # Plotting the training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_loss, 'g', label='Training Loss')
    plt.plot(epochs, validation_loss, 'b', label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracy, 'g', label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, 'b', label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


'''
Task 1.1 - Show the distribution of data in two classes (e.g., using a barplot) in the combined dataset
'''
def showDistribution(data):
    # Show the distribution of data in two classes using a barplot
    class_distribution = data['cancer'].value_counts()
    plt.bar(class_distribution.index, class_distribution.values)
    plt.xlabel('Cancer Type')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()



'''
Task 1.2 - Encode the labels
'''
def encodeBinaryLabels(data, n1, n2):
    # Encode the labels (AML: 0, ALL: 1)
    encode_labels = {'AML': n1, 'ALL': n2}
    data['cancer'] = data['cancer'].map(encode_labels)
    return data



'''
Task 1.3 - Remove all the "Call" columns from both data files
'''
def removeColWithName(data, name):
    data = data.loc[:, ~data.columns.str.contains(f'{name}')]
    return data



'''
Task 1.4 - Associate the train and test data to the labels
'''
def associateData(labels, train, test):
    num_label_rows, num_label_cols = labels.shape
    num_train_rows, num_train_cols = train.shape

    # Reorder the training data columns
    desired_col_order_train = [str(i) for i in range(1, num_train_cols+1)]
    train = train[desired_col_order_train].copy()

    # Reorder the test data columns
    desired_col_order_test = [str(i) for i in range(num_train_cols+1, num_label_rows+1)]
    test = test[desired_col_order_test].copy()

    # Extract the labels for train_data and test_data
    train_labels = labels['cancer'].iloc[:38].values
    test_labels = labels['cancer'].iloc[38:].values

    # Add the labels as new rows at the bottom of train_data and test_data
    train.loc['labels'] = train_labels
    test.loc['labels'] = test_labels

    return train, test, train_labels, test_labels



'''
Task 1.5 - Compute and display summary statistics for the data
'''
def computSummaryStats(data, type):
    # # Use describe() on the selected columns
    column_descriptions = data.describe()
    print(f"------------ Beneath is a description of various statistical relevant features for {type} features")
    print(column_descriptions)



'''
Task 2.2 - Use PCA from sklearn to select features that account for 90% of data variance in trainset
'''
def performPCA(data):
    # Convert column names to strings
    data.columns = data.columns.astype(str)
    
    # Standardize the data (mean=0, std=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA to select features explaining 90% of variance
    pca = PCA(n_components=0.90)
    pca_features = pca.fit_transform(scaled_data)
    
    # Return the PCA-transformed features
    return pca_features



'''
Task 2.3 - Visualize the trainset in 3D space when the first 3 PCA components are selected
'''
def visualizePCA(pca_features, labels):
    colors = ['black' if label == 0 else 'white' for label in labels]
    # Visualize the trainset in 3D space using the first 3 PCA components
    fig = plt.figure()
    fig.set_facecolor('pink')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#e0e0e0')

    # Changing color of the axis lines
    ax.xaxis.line.set_color("purple")
    ax.yaxis.line.set_color("red")
    ax.zaxis.line.set_color("blue")

    # Changing color of the box representing the 3D space
    ax.xaxis.set_pane_color((0.5, 0, 0.5, 0.5))  # x-axis plane
    ax.yaxis.set_pane_color((0.5, 0, 0.5, 0.5))  # y-axis plane
    ax.zaxis.set_pane_color((0.5, 0, 0.5, 0.5))  # z-axis plane

    # Scatter plot with star markers for data points
    ax.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], c=colors, marker='*')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')

    # Set the title to be bold and in cursive style
    plt.title('PCA Visualization of Trainset (First 3 Components)', fontweight='bold', fontstyle='italic')
    plt.show()



'''
Task 3.1 - Establish a simple baseline model, by assigning the label of the majority class to all data and
calculating the accuracy on test set. Your models should not perform worse than this
'''
def establishBaseline(train_labels, test_labels):
    # Find the majority class from train data (1)
    majority_class = train_labels.value_counts().idxmax()
    baseline_prediction = [majority_class] * len(test_labels)

    # Find the baseline accuracy
    baseline_accuracy = accuracy_score(test_labels, baseline_prediction) * 100
    return baseline_accuracy
        


'''
Task 3.2 - Logistic regression is a statistical method for predicting binary classes. The outcome or target
variable is dichotomous in nature. Dichotomous means there are only two possible classes. For
example, it can be used for cancer detection problems. It computes the probability of an event
occurrence
'''
def logisticRegression(train_data, test_data, train_labels, test_labels, parameters):
    # Determine best parameters using GridSearchCV
    model = LogisticRegression()
    grid_search = GridSearchCV(model, parameters)
    grid_search.fit(train_data, train_labels)

    best_params = grid_search.best_params_
    solver_saga = parameters['solver'][0]

    # Train model on data using best parameters 
    model_lr = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver=solver_saga)
    model_lr.fit(train_data, train_labels)

    # Make prediction about test data using model
    test_pred_lg = model_lr.predict(test_data)

    # Compute accuracy and confusion_lr matrix
    accuracy = accuracy_score(test_labels, test_pred_lg) * 100
    confusion = confusion_matrix(test_labels, test_pred_lg)

    return confusion, accuracy, best_params['C'], best_params['penalty']



'''
Task 3.4 - Use a decision tree from sklearn to predict the cancer type and report its performance as in
subtask 2
'''
def decisonTree(train_data, test_data, train_labels, test_labels, parameters):
    # Determine best parameter
    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(model, parameters)
    grid_search.fit(train_data, train_labels)

    # Associate values
    best_params = grid_search.best_params_

    # Train model using DT classifier
    model_dt = DecisionTreeClassifier(max_depth=best_params['max_depth'])
    model_dt.fit(train_data, train_labels)

    # Make predictions on test data using model
    test_pred_dt = model_dt.predict(test_data)

    accuracy = accuracy_score(test_labels, test_pred_dt) * 100
    confusion = confusion_matrix(test_labels, test_pred_dt)

    return confusion, accuracy, best_params['max_depth']



'''
Task 3.5 - It has been mentioned in class, that a Random Forest classifier is an ensemble of several decision
trees. Use a random forest from sklearn to predict the cancer type and report its performance as
in subtask 2
'''
def randomForestClassifier(train_data, test_data, train_labels, test_labels, parameters):
    # Determine best parameters using GridSearchCV
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, parameters)
    grid_search.fit(train_data, train_labels)

    # Associate best parameters
    best_params = grid_search.best_params_

    # Train model
    model_rfc = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
    model_rfc.fit(train_data, train_labels)

    # Make prediction
    test_pred_rfc = model_rfc.predict(test_data)

    accuracy= accuracy_score(test_labels, test_pred_rfc) * 100
    confusion = confusion_matrix(test_labels, test_pred_rfc)

    return confusion, accuracy, best_params['max_depth'], best_params['n_estimators']



'''
Part 3.6 - Build an ANN (using tensorflow-keras) to predict the cancer type and report its performance as in
subtask 2. Choose an appropriate architecture. Explain how you have decided to choose the
parameters such as learning rate, batch size, number of epochs, size of the hidden layer, number
of hidden layers, etc.

As the epochs go by, we expect that its error on the training set naturally goes down. But we are
not actually sure that overfitting is happening or not. One thing we can do, is to further split the
training set into train and validation and monitor the validation loss as we do for the train loss. If
after a while, the validation error stops decreasing, this indicates that the model has started to
overfit the training data. With Early Stopping, you just stop training as soon as the validation error
reaches the minimum. Try to add Early stopping using keras callbacks to your model.
'''
def runANN(train_data, test_data, train_labels, test_labels):
    # Split the training data into 80% train and 20% validation set
    (train_data_ann, val_data_ann, 
    train_data_labels_ann, val_data_labels_ann) = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    # Building the model
    # Create a Sequential model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=train_data_ann.shape[1])) # Input layer
    model.add(Dense(32, activation='relu')) # Hidden layers
    model.add(Dense(1, activation='sigmoid')) # Output layer

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                loss='binary_crossentropy', 
                metrics=['accuracy'])

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(train_data_ann, train_data_labels_ann, epochs=50, batch_size=64, 
                        validation_data=(val_data_ann, val_data_labels_ann), callbacks=[early_stopping], verbose=0)

    # Evaluate the model on the test set
    test_loss, test_accuracy_ann = model.evaluate(test_data, test_labels)

    # Make predictions using ANN 
    test_pred_ann = model.predict(test_data)

    # Convert the predicted probabilities to binary labels (0 or 1) based on a threshold (e.g., 0.5)
    threshold = 0.5
    test_pred_binary_ann = (test_pred_ann > threshold).astype(int)

    # Calculate the confusion matrix
    accuracy = accuracy_score(test_labels, test_pred_binary_ann) * 100
    confusion = confusion_matrix(test_labels, test_pred_binary_ann)

    return confusion, accuracy, test_loss, test_accuracy_ann, history





if __name__ == "__main__":
    # Load the data
    testing_data = pd.read_csv("data_set_ALL_AML_independent.csv")
    training_data = pd.read_csv("data_set_ALL_AML_train.csv")

    # Load the labels
    labels = pd.read_csv("actual.csv")


    # 1.1 - Show distribution of classes
    # showDistribution(labels)


    # # 1.2 - Encode Lables
    labels = encodeBinaryLabels(labels, 0, 1)


    # 1.3 - Remove columns with 'call'
    training_data = removeColWithName(training_data, 'call')
    testing_data = removeColWithName(testing_data, 'call')


    # 1.4 - Associate the train and test data with labels
    # Remove gene information
    train_data = training_data.iloc[:, 2:]
    test_data = testing_data.iloc[:, 2:]

    (train_data, test_data,
     train_labels, test_labels) = associateData(labels, train_data, test_data)


    # 1.5 - Compute and display data summary stats
    trans_train = train_data.transpose()
    cols_to_describe_train = trans_train.iloc[:, :-1]
    trans_test = test_data.transpose()
    cols_to_describe_test = trans_test.iloc[:, :-1]

    # computSummaryStats(cols_to_describe_train, "Training")
    # computSummaryStats(cols_to_describe_test, "Testing")


    # 2.2 - PCA
    # pca_features = performPCA(cols_to_describe_train)


    # 2.3 - Visualize PCA (for train data)
    # visualizePCA(pca_features, train_labels)


    # 3.1 - Establish baseline
    # Convert the NumPy array back to a pandas DataFrame
    # Train Labels (X Test)
    train_labels = pd.DataFrame(train_labels)
    train_labels.columns = ["Train Labels"]

    # Test Labels (Y Test)
    test_labels = pd.DataFrame(test_labels)
    test_labels.columns = ["Test Labels"]

    baseline_accuracy = establishBaseline(train_labels, test_labels)


    #### #### #### #### #### #### #### #### ####
    # Drop label column and transpose data 
    train_data = train_data.drop("labels")
    test_data = test_data.drop("labels")

    train_data = train_data.transpose()
    test_data = test_data.transpose()
    #### #### #### #### #### #### #### #### ####


    # 3.2 (Logistic Regression)
    parameters_lr = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2', 'elasticnet', None], 'solver': ['saga']}
    confusion_lr, accuracy_lr, c_lr, penalty_lr = logisticRegression(train_data,
                                                                test_data,
                                                                train_labels.values.ravel(),
                                                                test_labels.values.ravel(),
                                                                parameters_lr)

    # 3.4 (Decision Tree)
    parameters_dt = {'max_depth': [5, 10, 15, 20]}
    confusion_dt, accuracy_dt, depth_dt = decisonTree(train_data,
                                              test_data, 
                                              train_labels, 
                                              test_labels,
                                              parameters_dt)

    # 3.5 (Random Forest Classifier)
    parameters_rfc = {'n_estimators': [50, 75, 100, 150], 'max_depth': [5, 10, 15, 25]}
    confusion_rfc, accuracy_rfc, depth_rfc, n_est_rfc = randomForestClassifier(train_data,
                                                                    test_data,
                                                                    train_labels.values.ravel(), 
                                                                    test_labels.values.ravel(),
                                                                    parameters_rfc)

    # 3.6 (ANN)
    (confusion_ann, accuracy_ann,
     test_loss, test_accuracy_ann, history_ann) = runANN(train_data, 
                                            test_data, 
                                            train_labels, 
                                            test_labels)

    # 3.9 - Repeat Logistic Regression, Decision Tree, Random Forest Classifier
    # and ANN using dimensionally reduced data
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data) 

    # Perform PCA to reduce dimensionality while retaining 90% of the variance
    pca = PCA(n_components=0.90)
    pca_train_data = pca.fit_transform(scaled_train_data)
    pca_test_data = pca.transform(scaled_test_data)

    # (Logistic Regression)
    (confusion_lr_pca, accuracy_lr_pca, 
    c_lr_pca, penalty_lr_pca) = logisticRegression(pca_train_data,
                                                        pca_test_data,
                                                        train_labels.values.ravel(),
                                                        test_labels.values.ravel(),
                                                        parameters_lr)

    # (Decision Tree)
    confusion_dt_pca, accuracy_dt_pca, depth_dt_pca = decisonTree(pca_train_data,
                                                                  pca_test_data,
                                                                  train_labels, 
                                                                  test_labels,
                                                                  parameters_dt)

    # (Random Forest Classifier)
    (confusion_rfc_pca, accuracy_rfc_pca,
      depth_rfc_pca, n_est_rfc_pca) = randomForestClassifier(pca_train_data, 
                                                                pca_test_data, 
                                                                train_labels.values.ravel(), 
                                                                test_labels.values.ravel(),
                                                                parameters_rfc)

    # (ANN)
    (confusion_ann_pca, accuracy_ann_pca,
     test_loss_pca, test_accuracy_ann_pca, history_ann_pca) = runANN(pca_train_data, 
                                                    pca_test_data, 
                                                    train_labels, 
                                                    test_labels)

    # Format Output: 
    print("\n")
    print("---------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------")
    print("Results from models/classifiers when raw data is used")
    print(f"Baseline prediction accuracy: {baseline_accuracy:.2f}% \n")

    calc_performance_and_print(confusion_lr, "Logistic Regression")
    print(f"Value used for 'C' = {c_lr} and value used as 'Penalty' = {penalty_lr}\n")
    
    calc_performance_and_print(confusion_dt, "\nDecision Tree")
    print(f"Value used for 'Depth' = {depth_dt}\n")

    calc_performance_and_print(confusion_rfc, "\nRandom Forest Classifier")
    print(f"Value used for 'Depth' = {depth_rfc} and value used as 'Estimators' = {n_est_rfc}\n")
    
    calc_performance_and_print(confusion_ann, "\nANN")
    print(f"ANN Test Loss: {test_loss:.2f}\n")

    # Plot confusion matrices
    plot_confusion_matrix(confusion_lr, "Logistic Regression")
    plot_confusion_matrix(confusion_dt, "Decision Tree")
    plot_confusion_matrix(confusion_rfc, "Random Forest Classifier")
    plot_confusion_matrix(confusion_ann, "ANN")
    plot_ann_history(history_ann, 'Training and Validation Loss')

    # Format Output: 
    print("---------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------")
    print("Results from models/classifiers when using dimensionally reduced data")

    calc_performance_and_print(confusion_lr_pca, "Logistic Regression - PCA")
    print(f"Value used for 'C' = {c_lr_pca} and value used as 'Penalty' = {penalty_lr_pca}\n")

    calc_performance_and_print(confusion_dt_pca, "\nDecision Tree - PCA")
    print(f"Value used for 'Depth' = {depth_dt_pca}\n")

    calc_performance_and_print(confusion_rfc_pca, "\nRandom Forest Classifier - PCA")
    print(f"Value used for 'Depth' = {depth_rfc_pca} and value used as 'Estimators' = {n_est_rfc_pca}\n")
    
    calc_performance_and_print(confusion_ann_pca, "\nANN - PCA")
    print(f"ANN Test Loss: {test_loss_pca:.2f}\n")

    # Plot confusion matrices
    plot_confusion_matrix(confusion_lr_pca, "Logistic Regression")
    plot_confusion_matrix(confusion_dt_pca, "Decision Tree")
    plot_confusion_matrix(confusion_rfc_pca, "Random Forest Classifier")
    plot_confusion_matrix(confusion_ann_pca, "ANN")  
    plot_ann_history(history_ann_pca, 'Training and Validation Loss')