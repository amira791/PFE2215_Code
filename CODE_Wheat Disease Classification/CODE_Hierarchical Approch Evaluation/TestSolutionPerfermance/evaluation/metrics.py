from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

def print_metrics(metrics):
    print("\n✅ Evaluation Metrics on Test Set:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Recall:   {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")