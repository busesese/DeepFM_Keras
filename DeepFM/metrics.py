import tensorflow as tf
import keras.backend as K
import numpy as np 


# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41015
# AUC for a binary classifier
def auc(y_true, y_pred):
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

