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



logger.saveDefault = True
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

log('Start')
handler = DataHandler()
log('Load Data')

with open('Datasets/CHI_crime/pre.pkl','rb') as fs:
    pre = pickle.load(fs)

args.row, args.col, _, args.offNum = pre.shape
args.areaNum = args.row * args.col
rspFunc = (lambda tensor: np.reshape(tensor, [args.areaNum, -1, args.offNum]))
pre_tensor = rspFunc(pre)

with tf.Session(config=config) as sess:
    model = Model(sess, handler)
    preds = model.run_one(pre_tensor) # row*col, 30 days, offNum
    print(preds)
    np.savetxt('today\'s prediction.txt', preds, fmt='%.4f')