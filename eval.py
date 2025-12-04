import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
    confusion_matrix,
    classification_report,
)
from utils import *

from all_pass_classifier import all_pass_classify
from heuristic_classifier import heursitic_model
from rf_st_classifier import rf_st_classify
from tfidf_classifier import tfidf_classify
from xgboost_st_classifier import xgboost_st_classify



#def print_report(name, y_true, y_pred, y_proba):
def print_report(name, y_true, preds, num_mis = 10): 
    y_pred = preds['pass']
    y_proba = preds['pass_prob']

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {name} ---")
    print("Confusion Matrix [TN FP; FN TP]:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print(f"\nTop {num_mis} Misclassifications:")

    df = pd.DataFrame({
        "title": m_test["title"].values,
        "true": y_true,
        "pred": y_pred
    })

    # Include probabilities if available
    if y_proba is not None:
        df["proba"] = y_proba
        df["conf_error"] = abs(df["proba"] - df["true"])
    else:
        df["proba"] = np.nan
        df["conf_error"] = 1 

    mis = df[df["true"] != df["pred"]]

    if mis.empty:
        print("No misclassfications")
        return

    # Sort by highest confidence error (or fallback)
    mis_top = mis.sort_values("conf_error", ascending=False).head(num_mis)
    
    print(mis_top[["title","true", "pred", "proba", "conf_error"]])

    # save all rows to csv
    full_csv = f"./evals/{name}_full_predictions.csv"
    df.to_csv(full_csv, index=False)
    print(f"\nSaved predictions to {full_csv}")



if __name__ == '__main__':
    
    print("--------- RECOMMENDER EVALUATIONS ---------")
    
    print("Splitting the data...\n")

    m_train, m_test = read_and_split_data()

    
    
    y_true = m_test['bechdel_pass']

    models = {
        "all_pass": all_pass_classify,
        "heuristic": heursitic_model,
        "tfidf": tfidf_classify,
        "rf_st_transform": rf_st_classify,
        "xgboost_st": xgboost_st_classify,
    }

    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        print(f"---------- Running {name} classifier... ----------")
        predictions[name] = model(m_train, m_test)
        y_pred = predictions[name]['pass']
        y_actual = y_true
        
        fpr, tpr, _ = roc_curve(y_actual, y_pred)
        roc_auc= roc_auc_score(y_actual, y_pred)
        print(f"ROC-AUC = {roc_auc}")

        # -- PR-AUC --
        precision, recall, _ = precision_recall_curve(y_actual, y_pred)
        pr_auc = auc(recall, precision)
        print(f"PR-AUC = {pr_auc}")


        plt.figure(figsize=(10,5))

        # --- ROC Curve ---
        plt.subplot(1,2,1)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], linestyle='--', color='gray')  # random baseline
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)

        # --- Precision-Recall Curve ---
        plt.subplot(1,2,2)
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})', color='green')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()

        new_name = "_".join([n.lower() for n in name.split()])
        plt.savefig(P(f'./figures/{new_name}_eval.png'))
        
        print()

    
        
    for name in models.keys():  

        print_report(name, y_true, predictions[name])




    