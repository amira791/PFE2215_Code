from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(10, 10))
    disp.plot(cmap='Blues', xticks_rotation=45) 
    plt.title("Confusion Matrix for Final Model")
    plt.tight_layout()
    plt.show()