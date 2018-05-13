from keras.engine.topology import Layer
from keras import backend as K 
import tensorflow as tf 
import numpy as np


class MyMeanPool(Layer):
	def __init__(self, axis, **kwargs):
		self.supports_masking = True
		self.axis = axis
		super(MyMeanPool, self).__init__(**kwargs)

	def compute_mask(self, input, input_mask=None):
		# need not to pass the mask to next layers
		return None

	def call(self, x, mask=None):
		if mask is not None:
			if K.ndim(x)!=K.ndim(mask):
				mask = K.repeat(mask, x.shape[-1])
				mask = tf.transpose(mask, [0,2,1])
			mask = K.cast(mask, K.floatx())
			x = x * mask
			return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
		else:
			return K.mean(x, axis=self.axis)

	def compute_output_shape(self, input_shape):
		output_shape = []
		for i in range(len(input_shape)):
			if i!=self.axis:
				output_shape.append(input_shape[i])
		return tuple(output_shape)

class MySumLayer(Layer):
	def __init__(self, axis, **kwargs):
		self.supports_masking = True
		self.axis = axis
		super(MySumLayer, self).__init__(**kwargs)

	def compute_mask(self, input, input_mask=None):
		# do not pass the mask to the next layers
		return None

	def call(self, x, mask=None):
		if mask is not None:
			# mask (batch, time)
			mask = K.cast(mask, K.floatx())
			if K.ndim(x)!=K.ndim(mask):
				mask = K.repeat(mask, x.shape[-1])
				mask = tf.transpose(mask, [0,2,1])
			x = x * mask
			if K.ndim(x)==2:
				x = K.expand_dims(x)
			return K.sum(x, axis=self.axis)
		else:
			if K.ndim(x)==2:
				x = K.expand_dims(x)
			return K.sum(x, axis=self.axis)

	def compute_output_shape(self, input_shape):
		output_shape = []
		for i in range(len(input_shape)):
			if i!=self.axis:
				output_shape.append(input_shape[i])
		if len(output_shape)==1:
			output_shape.append(1)
		return tuple(output_shape)

