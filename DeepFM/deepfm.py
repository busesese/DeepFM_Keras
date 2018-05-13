import numpy as np 
import pandas as pd 
from keras.layers import Input, Dense, Embedding, Add, Concatenate, RepeatVector,Multiply,Subtract,Lambda,Dropout,Reshape,Flatten
from keras.engine.topology import Layer 
from keras import backend as K 
import tensorflow as tf
from keras.models import Model
from keras.utils import plot_model
from mylayers import MyMeanPool,MySumLayer,MyFlatten
from keras.optimizers import Adam
import config
from keras.metrics import binary_accuracy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K

def gini(actual, pred, cmpcol = 0, sortcol = 1):
	assert( len(actual) == len(pred) )
	all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
	all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
	totalLosses = all[:,0].sum()
	giniSum = all[:,0].cumsum().sum() / totalLosses

	giniSum -= (len(actual) + 1) / 2.
	return giniSum / len(actual)
 
def gini_normalized(a, p):
	return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
	labels = dtrain.get_label()
	gini_score = gini_normalized(labels, preds)
	return 'gini', gini_score

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41108
def jacek_auc(y_true, y_pred):
	score, up_opt = tf.metrics.auc(y_true, y_pred)
	#score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)    
	K.get_session().run(tf.local_variables_initializer())
	with tf.control_dependencies([up_opt]):
		score = tf.identity(score)
	return score

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41015
# AUC for a binary classifier
def discussion41015_auc(y_true, y_pred):
	ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
	pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
	pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
	binSizes = -(pfas[1:]-pfas[:-1])
	s = ptas*binSizes
	return K.sum(s, axis=0)

#---------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
	y_pred = K.cast(y_pred >= threshold, 'float32')
	# N = total number of negative labels
	N = K.sum(1 - y_true)
	# FP = total number of false alerts, alerts from the negative class labels
	FP = K.sum(y_pred - y_pred * y_true)
	return FP/N

#----------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
	y_pred = K.cast(y_pred >= threshold, 'float32')
	# P = total number of positive labels
	P = K.sum(y_true)
	# TP = total number of correct alerts, alerts from the positive class labels
	TP = K.sum(y_pred * y_true)
	return TP/P

class KerasDeepFM(object):
	def __init__(self,k, feat_dict,):
		self.k = k 				#the number of embedding dim
		self.feat_dict = feat_dict			#the unique number of every features

	def fit(self, x_train, y_train,x_val,y_val):
		#numeric cols
		input_cols = []
		numeric_cols = []
		embed_col = []
		for col in config.NUMERIC_COLS:
			in_neu = Input(shape=(1,), name=col)			#None*1
			input_cols.append(in_neu)
			in_embed = RepeatVector(1)(Dense(self.k)(in_neu))	#None*1*k
			numeric_cols.append(in_neu)
			embed_col.append(in_embed)
		con_numeric = Concatenate(axis=1)(numeric_cols)		#None*len(config.NUMERIC_COLS)
		dense_numeric = RepeatVector(1)(Dense(1)(con_numeric))	#None*1*1

		#categorical cols
		categorical_cols = []
		for col in config.CATEGORECIAL_COLS:
			in_cate = Input(shape=(1,),name=col)			#None*1
			input_cols.append(in_cate)
			cate_embedding = Embedding(self.feat_dict[col], 1)(in_cate)	#None*1*1
			in_embed = Embedding(self.feat_dict[col], self.k)(in_cate)		#None*1*k
			embed_col.append(in_embed)
			categorical_cols.append(cate_embedding)
		con_cate = Concatenate(axis=1)(categorical_cols)		#None*len(config.CATEGORECIAL_COLS)*1

		#first order
		y_first_order = Concatenate(axis=1)([dense_numeric, con_cate]) 		#None*len*1
		y_first_order = MySumLayer(axis=1)(y_first_order)				#None*1	

		#second order
		emb = Concatenate(axis=1)(embed_col)						#None*s*k

		summed_feature_emb = MySumLayer(axis=1)(emb)				#None*k
		summed_feature_emb_squred = Multiply()([summed_feature_emb,summed_feature_emb])	#None*k

		squared_feature_emb = Multiply()([emb,emb])					#None*s*k
		squared_sum_feature_emb = MySumLayer(axis=1)(squared_feature_emb)	#None*k

		sub = Subtract()([summed_feature_emb_squred,squared_sum_feature_emb])	#None*k
		sub = Lambda(lambda x: x*0.5)(sub)						#None*k
		y_second_order = MySumLayer(axis=1)(sub)					#None*1

		#deep order
		y_deep = Flatten()(emb)								#None*(s*k)
		y_deep = Dropout(0.5)(Dense(32,activation='relu')(y_deep))			#None*32
		y_deep = Dropout(0.5)(Dense(32,activation='relu')(y_deep))			#None*32
		# y_deep = Dropout(0.5)(Dense(32,activation='relu')(y_deep))			#None*32
		y_deep = Dropout(0.5)(Dense(1,activation='relu')(y_deep))			#None*1


		#deep fm
		y = Concatenate()([y_first_order,y_second_order,y_deep])			#None*3
		y = Dense(1,activation='sigmoid')(y)						#None*1


		
		self.model = Model(inputs=input_cols, outputs=[y])
		# self.model.summary()
		# model.compile(optimizer='rmsprop', loss='binary_crossentropy')
		# model.fit(x_train, y_train, batch_size=64, nb_epoch=200,verbose=1, validation_data=(x_valid,y_valid))
		self.model.compile(optimizer=Adam(lr=0.001,decay=0.1), loss='binary_crossentropy',metrics=[jacek_auc])
		self.model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val,y_val))

	def predict(self,x):
		y_pred = self.model.predict(x)
		return y_pred

# plot_model(model, '../save/deepfm.png',show_shapes=True)
def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

def preprocess_data(df):
	feat_dict = {}
	for col in df.columns:
		if col in config.CATEGORECIAL_COLS:
			features_num = df[col].unique()
			feat_dict[col] = len(features_num)
	return feat_dict


print("starting read data")
data = pd.read_csv(config.TRAIN_FILE)
# data = preprocess(data)
print("starting preprocessing data")
for col in config.CATEGORECIAL_COLS:
	lel = LabelEncoder()
	data[col] = lel.fit_transform(data[col])

# cols = [c for c in data.columns if c not in ["id", "target"]]
# cols = [c for c in cols if (not c in config.IGNORE_COLS)]
# x = dfTrain[cols]
# y = dfTrain["target"]

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