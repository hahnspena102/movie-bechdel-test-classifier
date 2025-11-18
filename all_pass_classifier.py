
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import List

from pathlib import Path

from utils import *

def all_pass_classify(m_train, m_test):
    m_pred = m_test.copy()
    m_pred["pass"] = True
    m_pred["pass_prob"] = 1
    return m_pred



if __name__ == "__main__":
    
    
    m_train, m_test = read_and_split_data()

    predictions = all_pass_classify(m_train, m_test)
    print(predictions)

