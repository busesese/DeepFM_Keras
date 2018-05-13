import pandas as pd 
import numpy as np
import config

#featrue engineering : missing_feat and ps_car_13_x_ps_reg_03
def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

#feat_dict :caculate the unique values of every categorical feautres
def preprocess_data(df):
	feat_dict = {}
	for col in df.columns:
		if col in config.CATEGORECIAL_COLS:
			features_num = df[col].unique()
			feat_dict[col] = len(features_num)
	return feat_dict