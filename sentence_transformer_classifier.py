import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from utils import *

# Keyword lists
female_keywords = ['she', 'her', 'woman', 'girl', 'mother', 'daughter', 'female', 'women', 'sisters']
male_keywords   = ['he', 'him', 'man', 'boy', 'father', 'son', 'male', 'men', 'brothers']

def keyword_counts(text, keywords):
    text_lower = text.lower()
    return sum(text_lower.count(word) for word in keywords)

def sentence_transformer_classify(m_train, m_test):
    movies_train = m_train.copy()
    movies_test  = m_test.copy()

    # Combine all text
    def combine_text(df):
        return df["title"].fillna("") + " " + df["overview"].fillna("") + " " + df["tagline"].fillna("")

    movies_train["words"] = combine_text(movies_train)
    movies_test["words"]  = combine_text(movies_test)

    # genre features
    mlb = MultiLabelBinarizer()
    genre_train = mlb.fit_transform(movies_train['all_genres'])
    genre_test  = mlb.transform(movies_test['all_genres'])
    genre_cols  = mlb.classes_

    movies_train = pd.concat([movies_train, pd.DataFrame(genre_train, columns=genre_cols, index=movies_train.index)], axis=1)
    movies_test  = pd.concat([movies_test, pd.DataFrame(genre_test, columns=genre_cols, index=movies_test.index)], axis=1)

    # keyword features
    for df in [movies_train, movies_test]:
        df['female_kw_count'] = df['words'].apply(lambda x: keyword_counts(x, female_keywords))
        df['male_kw_count']   = df['words'].apply(lambda x: keyword_counts(x, male_keywords))
        df['female_ratio']    = df['female_kw_count'] / (df['female_kw_count'] + df['male_kw_count'] + 1e-6)

    keyword_features = ['female_kw_count', 'male_kw_count', 'female_ratio']

    # sentence transformer embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_embeddings = model.encode(movies_train['words'].tolist(), show_progress_bar=True)
    test_embeddings  = model.encode(movies_test['words'].tolist(), show_progress_bar=True)

    # Scale embeddings
    scaler = StandardScaler()
    train_embeddings_scaled = scaler.fit_transform(train_embeddings)
    test_embeddings_scaled  = scaler.transform(test_embeddings)

    # combine all features
    FEATURES = ['release_year', 'popularity'] + list(genre_cols) + keyword_features
    X_train_basic = movies_train[FEATURES].values
    X_test_basic  = movies_test[FEATURES].values

    X_train = np.hstack([X_train_basic, train_embeddings_scaled])
    X_test  = np.hstack([X_test_basic, test_embeddings_scaled])

    y_train = movies_train["bechdel_pass"]

    # perform random forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    m_pred = movies_test.copy()
    m_pred["pass_prob"] = rf.predict_proba(X_test)[:, 1]
    m_pred["pass"] = m_pred["pass_prob"] > 0.5

    print("\nTop 30 Feature Importances:")
    importances = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf.feature_importances_[:len(FEATURES)]
    }).sort_values(by="importance", ascending=False)
    print(importances.head(10))

    return m_pred

if __name__ == "__main__":
    m_train, m_test = read_and_split_data()
    predictions = sentence_transformer_classify(m_train, m_test)
    print(predictions)
