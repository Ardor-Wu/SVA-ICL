python
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
 
full_data = pd.read_csv("vulnerability_data.csv")
full_data["date"] = pd.to_datetime(full_data["date"])
 
# 设置截止日期并筛选测试数据
cutoff_date = "2023-07-01"
post_cutoff_data = full_data[full_data["date"] > cutoff_date]
 
true_labels = post_cutoff_data["true_label"].values
predictions = post_cutoff_data["predicted_label"].values
 
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average="binary")
mcc = matthews_corrcoef(true_labels, predictions)
 
results = pd.DataFrame({
    "Metric": ["Accuracy", "F1-Score", "MCC"],
    "Post-Cutoff Performance": [accuracy, f1, mcc]
})
results.to_csv("post_cutoff_evaluation.csv", index=False)
print(results)
