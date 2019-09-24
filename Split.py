import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
bank_df = pd.read_csv("/home/munib/Desktop/parkinsons_updrs.csv")
train, test = train_test_split(bank_df, test_size=0.3)
train.to_csv("parkinsons_train.csv")
test.to_csv("parkinsons_test.csv")
