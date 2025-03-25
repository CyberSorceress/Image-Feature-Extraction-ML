import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import time

# Load classifier results
results = np.load("reports/classifier_results.npy", allow_pickle=True).item()

# Convert results to DataFrame
results_df = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-Score", "Computation Time"])

# Evaluate classifiers
for classifier, data in results.items():
    y_true, y_pred, y_proba, computation_time = data["y_true"], data["y_pred"], data["y_proba"], data["time"]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    results_df = results_df.append({"Classifier": classifier, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "Computation Time": computation_time}, ignore_index=True)
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {classifier}")
    plt.show()
    
    # Compute and plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {classifier}")
    plt.legend(loc="lower right")
    plt.show()

# Save results table
results_df.to_csv("reports/classifier_comparison.csv", index=False)
print("Evaluation report generated and saved.")
