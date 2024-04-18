from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn import svm

x_train= np.array([[30, 21], [12, 24], [23, 29], [23, 38], [32, 37], [22, 34], [21, 47], [5, 43], [5, 47], [21, 24]])
y_train = np.array([-1,  1,  1, -1,  1,  1,  1,  1, -1, -1])

clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(x_train, y_train)

clf_nonlinear = svm.SVC(kernel='rbf', gamma='auto')
clf_nonlinear.fit(x_train, y_train)
# Linear SVM predictions
linear_predictions = clf_linear.predict(x_train)

# Non-linear SVM with RBF kernel predictions
nonlinear_predictions = clf_nonlinear.predict(x_train)

# Compute metrics for Linear SVM
linear_accuracy = accuracy_score(y_train, linear_predictions)
linear_precision = precision_score(y_train, linear_predictions, zero_division=1) # Add zero_division parameter
linear_recall = recall_score(y_train, linear_predictions, zero_division=1) # Add zero_division parameter
linear_f1 = f1_score(y_train, linear_predictions, zero_division=1) # Add zero_division parameter
linear_confusion_matrix = confusion_matrix(y_train, linear_predictions)

# Compute metrics for Non-linear SVM with RBF kernel
nonlinear_accuracy = accuracy_score(y_train, nonlinear_predictions)
nonlinear_precision = precision_score(y_train, nonlinear_predictions, zero_division=1) # Add zero_division parameter
nonlinear_recall = recall_score(y_train, nonlinear_predictions, zero_division=1) # Add zero_division parameter
nonlinear_f1 = f1_score(y_train, nonlinear_predictions, zero_division=1) # Add zero_division parameter
nonlinear_confusion_matrix = confusion_matrix(y_train, nonlinear_predictions)

# Print the metrics
print("Linear SVM Metrics:")
print("Accuracy:", linear_accuracy)
print("Precision:", linear_precision)
print("Recall:", linear_recall)
print("F1-score:", linear_f1)
print("Confusion Matrix:\n", linear_confusion_matrix)

print("\nNon-linear SVM with RBF Kernel Metrics:")
print("Accuracy:", nonlinear_accuracy)
print("Precision:", nonlinear_precision)
print("Recall:", nonlinear_recall)
print("F1-score:", nonlinear_f1)
print("Confusion Matrix:\n", nonlinear_confusion_matrix)