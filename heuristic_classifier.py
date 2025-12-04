import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import List

from pathlib import Path

from utils import *

def heursitic_model(m_train, m_test):
    
    m_pred = m_test.copy()
    fail_genres = ["Adventure", "Action", "Crime", "War", "Western", "History"]

    m_pred["pass"] = m_pred["all_genres"].apply(
    lambda genres: all(g not in fail_genres for g in genres)
        if isinstance(genres, list)  
        else False                   
    )   

    m_pred["pass_prob"] = m_pred["pass"].apply(lambda x: 1 if True else 0)

    print(m_pred[["pass", "pass_prob"]])
    
    return m_pred
    

if __name__ == "__main__":
    
    m_train, m_test = read_and_split_data()

    predictions = heursitic_model(m_train, m_test)
    print(predictions)

    
    

