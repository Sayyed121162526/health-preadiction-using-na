import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
import math

warnings.filterwarnings("ignore")

# Load the dataset using pandas
df = pd.read_csv('heartdisease.csv')

# Features (X) and Labels (y)
X = df.iloc[:, :-1].values  # All features except the last column (label)
y = df.iloc[:, -1].values   # The last column as the label

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Function to calculate mean and standard deviation of column values for each class
def mean(columnvalues):
    return sum(columnvalues) / float(len(columnvalues))

def stdev(columnvalues):
    avg = mean(columnvalues)
    variance = sum([(x - avg) ** 2 for x in columnvalues]) / (len(columnvalues) - 1)
    return math.sqrt(variance)

# Function to calculate Gaussian probability
def calculate_probability(x, mean, stdev):
    if stdev == 0:
        stdev += 1e-9
    exponent = math.exp(-((x - mean) ** 2) / (2 * stdev ** 2))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Implementing 5-fold cross-validation for more robust evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n\n\nCross Validation Fold {fold}\n\n\n")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Separate data by class
    separated = {}
    for i in range(len(X_train)):
        vector = X_train[i]
        class_value = y_train[i]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(vector)

    # Summarize dataset by calculating mean and standard deviation for each class
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = [(mean(attribute), stdev(attribute)) for attribute in zip(*instances)]

    # Predict function using Naive Bayes
    def predict(summaries, input_vector):
        probabilities = {}
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = 1
            for i in range(len(class_summaries)):
                mean, stdev = class_summaries[i]
                x = input_vector[i]
                probabilities[class_value] *= calculate_probability(x, mean, stdev)
        return max(probabilities, key=probabilities.get)

    # Get predictions
    y_pred = [predict(summaries, X_test[i]) for i in range(len(X_test))]

    # Accuracy calculation
    accuracy = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i]) / float(len(y_test)) * 100
    print(f"\n\nAccuracy: {accuracy:.2f}%")

    # Confusion matrix, F1 Score, Precision, and Recall
    cf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # ROC Curve and AUC
    n_classes = len(np.unique(y))
    y_test_bin = np.zeros((len(y_test), n_classes))
    y_pred_bin = np.zeros((len(y_pred), n_classes))

    for i in range(len(y_test)):
        y_test_bin[i, int(y_test[i])] = 1
    for i in range(len(y_pred)):
        y_pred_bin[i, int(y_pred[i])] = 1

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC (area = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'black'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
