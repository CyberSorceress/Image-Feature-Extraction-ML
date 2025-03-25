import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load classifier results
results = np.load("reports/classifier_results.npy", allow_pickle=True).item()

# Convert results to DataFrame
results_df = pd.DataFrame(list(results.items()), columns=["Classifier", "Accuracy"])

# Plot accuracy comparison
plt.figure(figsize=(10, 5))
sns.barplot(x="Classifier", y="Accuracy", data=results_df, palette="viridis")
plt.title("Classifier Performance Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()

# Save results table
results_df.to_csv("reports/classifier_comparison.csv", index=False)
print("Comparison report generated and saved.")
