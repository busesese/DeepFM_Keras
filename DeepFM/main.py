from deepfm import KerasDeepFM
from datapreprocessing import preprocess, preprocess_data
import config
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 




if __name__ == "__main__":
	print("starting read data")
	data = pd.read_csv(config.TRAIN_FILE)
	# data = preprocess(data)
	print("starting preprocessing data")
	for col in config.CATEGORECIAL_COLS:
		lel = LabelEncoder()
		data[col] = lel.fit_transform(data[col])

	x_train,x_val,y_train,y_val = train_test_split(data[config.NUMERIC_COLS+config.CATEGORECIAL_COLS],data['target'],test_size=0.8,random_state=config.RANDOMSTATE)
	x_train = x_train.values.T
	x_train = [np.array(x_train[i,:]) for i in range(x_train.shape[0])]
	y_train = y_train.values

	x_val = x_val.values.T
	x_val = [np.array(x_val[i,:]) for i in range(x_val.shape[0])]
	y_val = y_val.values

	feat_dict = preprocess_data(data)
	print("train model")
	kfm = KerasDeepFM(8, feat_dict)
	kfm.fit(x_train, y_train, x_val, y_val)