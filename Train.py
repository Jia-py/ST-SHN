import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from DataHandler import DataHandler
import tensorflow as tf
import pickle
from HG_ST_labcode import Model

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	log('Start')
	handler = DataHandler()
	log('Load Data')
	args.load_model = None
	with tf.Session(config=config) as sess:
		model = Model(sess, handler)
		model.run()