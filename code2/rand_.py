python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
 
similarity_df = pd.read_csv("similarity_matrix.csv", index_col=0)
true_labels = pd.read_csv("true_labels.csv", header=None).values.flatten()
relevant_predictions = pd.read_csv("relevant_selection_predictions.csv", header=None).values.flatten()
 
similarity_matrix = similarity_df.values
n_test, n_candidates = similarity_matrix.shape
k_selection = 4  # 示例选择数量
n_runs = 100
 
def random_selection_baseline(similarity_matrix, true_labels, k_selection, n_runs):
    random_predictions = []
    for _ in range(n_runs):
        # 为每个测试样本随机选择k个不同示例
        selected_indices = np.array([
            np.random.choice(
                n_candidates, 
                size=k_selection,
                replace=False
            ) for _ in range(n_test)
        ])
        votes = np.zeros(n_test)
        for i in range(k_selection):
            candidate_votes = np.random.choice([0, 1], size=n_test, p=[0.6, 0.4])
            votes += candidate_votes
        random_pred = (votes > k_selection/2).astype(int)
        random_predictions.append(random_pred)
    return np.array(random_predictions)
 
random_predictions = random_selection_baseline(similarity_matrix, true_labels, k_selection, n_runs)
 
def calculate_metrics(true, pred):
    return {
        "accuracy": accuracy_score(true, pred),
        "f1_score": f1_score(true, pred),
        "mcc": matthews_corrcoef(true, pred)
    }
 
random_metrics = {
    k: np.mean([calculate_metrics(true_labels, preds)[k] for preds in random_predictions])
    for k in ["accuracy", "f1_score", "mcc"]
}
 
relevant_metrics = calculate_metrics(true_labels, relevant_predictions)

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "F1-Score", "MCC"],
    "Random Selection": [random_metrics[k] for k in ["accuracy", "f1_score", "mcc"]],
    "Relevant Selection": [relevant_metrics[k] for k in ["accuracy", "f1_score", "mcc"]]
})
 
comparison.to_csv("selection_baseline_comparison.csv", index=False)
print(comparison)
