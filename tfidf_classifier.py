
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from pathlib import Path

from utils import *

def tfidf_classify(m_train, m_test):
    movies_train = m_train.copy()
    movies_test  = m_test.copy()

    # Combine all text to one column
    def combine_text(df):
        return (
            df["title"].fillna("") + " " +
            df["overview"].fillna("") + " " +
            df["tagline"].fillna("")
        )

    movies_train["words"] = combine_text(movies_train)
    movies_test["words"]  = combine_text(movies_test)

    # Fit on only training
    vect = TfidfVectorizer(stop_words="english", max_features=500)
    train_tfidf = vect.fit_transform(movies_train["words"])
    test_tfidf  = vect.transform(movies_test["words"])   

    tfidf_cols = vect.get_feature_names_out()

    tfidf_train_df = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_cols, index=movies_train.index)
    tfidf_test_df  = pd.DataFrame(test_tfidf.toarray(),  columns=tfidf_cols, index=movies_test.index)

    # merge tfidf back
    movies_train = pd.concat([movies_train, tfidf_train_df], axis=1)
    movies_test  = pd.concat([movies_test,  tfidf_test_df],  axis=1)
    
    #add in genres using multilable binarizer
    mlb = MultiLabelBinarizer()

    genre_train = mlb.fit_transform(movies_train['all_genres'])
    genre_test  = mlb.transform(movies_test['all_genres'])

    genre_cols = mlb.classes_ #all the unique genres

    movies_train = pd.concat(
        [movies_train, pd.DataFrame(genre_train, columns=genre_cols, index=movies_train.index)],
        axis=1
    )
    movies_test = pd.concat(
        [movies_test, pd.DataFrame(genre_test, columns=genre_cols, index=movies_test.index)],
        axis=1
    )

    FEATURES = list(tfidf_cols) + list(genre_cols)

    X = movies_train[FEATURES]
    y = movies_train["bechdel_pass"]

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
    m_pred["pass_prob"] = rf.predict_proba(movies_test[FEATURES])[:, 1]
    m_pred["pass"] = m_pred["pass_prob"] > 0.5
    
    print("\nSorted Feature Importances:")
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print(importances.head(10))
  
    return m_pred
    

if __name__ == "__main__":
    
    
    m_train, m_test = read_and_split_data()

    predictions = tfidf_classify(m_train, m_test)
    print(predictions)
