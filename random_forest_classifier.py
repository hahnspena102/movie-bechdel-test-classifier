
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from typing import List

from pathlib import Path

from utils import *

def random_forest_classify(m_train, m_test):
    FEATURES = ['year','popularity']
    train_data = m_train
    X = train_data[FEATURES]
    y = train_data["bechdel_pass"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,  
        random_state=42
    )
    rf.fit(X_train, y_train)


    y_val_pred = rf.predict_proba(X_val)[:, 1]
    #print(f"Validation ROC AUC: {roc_auc_score(y_val, y_val_pred):.3f}")
    
    
    y_val_pred =rf.predict_proba(X_val)[:, 1]
    #print(f"Validation ROC AUC: {roc_auc_score(y_val, y_val_pred):.3f}")
    
    m_pred = m_test.copy()
    m_pred["pass_prob"] = rf.predict_proba(m_test[FEATURES])[:, 1]
    m_pred["pass"] = m_pred["pass_prob"] > 0.5
    
    print("\nFeature Importance: ")
    print(list(zip(rf.feature_importances_, X.columns)))
    print()
    
  
    return m_pred
    



if __name__ == "__main__":
    
    
    m_train, m_test = read_and_split_data()

    predictions = random_forest_classify(m_train, m_test)
    print(predictions)

