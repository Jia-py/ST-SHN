import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Bias, defineParam, defineRandomNameParam
from DataHandler import DataHandler
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

class Model:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		self.metrics = dict()
		mets = ['preLoss', 'microF1', 'macroF1']
		for i in range(args.offNum):
			mets.append('F1_%d' % i)
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainpreLoss']) * args.tstEpoch
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		bestRes = None
		# args.epoch = 10 default
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			# 返回loss和preloss的字典
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch(self.handler.tstT, np.concatenate([self.handler.trnT, self.handler.valT], axis=1))
				if bestRes is None or args.task == 'r' and bestRes['MAPE'] > reses['MAPE'] or args.task == 'c' and bestRes['macroF1'] > reses['macroF1']:
					bestRes = reses
			if ep % args.tstEpoch == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch(self.handler.tstT, np.concatenate([self.handler.trnT, self.handler.valT], axis=1))
		log(self.makePrint('Test', args.epoch, reses, True))
		if bestRes is None or args.task == 'r' and bestRes['MAPE'] > reses['MAPE'] or args.task == 'c' and bestRes['macroF1'] > reses['macroF1']:
			bestRes = reses
		log(self.makePrint('Best', args.epoch, bestRes, True))
		self.saveHistory()

	def run_one(self, predict_tensor):
		self.prepareModel()
		log('Model Prepared')
		self.loadModel()
		# input tensor col*row,temporal_range,offence_num
		# test_tensor = self.handler.tstT[:,:args.temporalRange,:]
		feats = self.handler.zScore(predict_tensor)
		# 计算预测值
		preds = self.sess.run(self.preds, feed_dict={self.feats: feats, self.dropRate: 0.0})
		return preds

	def spacialModeling(self, rows, cols, vals, embeds):
		# edge, time, offense, latdim
		'''rowEmbeds和colEmbeds具体是啥，比较难理解
		猜测rowEmbeds是出发点
		colEmbeds是结束点的embeddings
		'''
		rowEmbeds = tf.nn.embedding_lookup(embeds, rows)# rowEmbeds 2116,30,4,16 embeds 256*30*4*16 rows 2116,1
		colEmbeds = tf.nn.embedding_lookup(embeds, cols)
		Q = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		K = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		V = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		# head相当于有多少层
		# q 2116,30,4,1,4,4 k 2116,30,1,4,4,4 rowEmbeds*Q 2116,30,4,16
		q = tf.reshape(tf.einsum('etod,dl->etol', rowEmbeds, Q), [-1, args.temporalRange, args.offNum, 1, args.head, args.latdim//args.head])
		k = tf.reshape(tf.einsum('etod,dl->etol', colEmbeds, K), [-1, args.temporalRange, 1, args.offNum, args.head, args.latdim//args.head])
		v = tf.reshape(tf.einsum('etod,dl->etol', colEmbeds, V), [-1, args.temporalRange, 1, args.offNum, args.head, args.latdim//args.head])
		# 对应公式2和3
		# att 2116,30,4,4,4,1 q*k 2116,30,4,4,4,4
		att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keep_dims=True) / tf.sqrt(float(args.latdim//args.head)), axis=3)
		# attV 2116,30,4,16
		attV = tf.reshape(tf.reduce_sum(att * v, axis=3), [-1, args.temporalRange, args.offNum, args.latdim])
		# 按照节点求和，之前的边是有重复值的
		ret = tf.math.segment_sum(attV * tf.nn.dropout(vals, rate=self.dropRate), rows)
		return Activate(ret, 'leakyRelu') # area, time, offense, latdim

	def temporalModeling(self, rows, cols, vals, embeds):
		# tf.slice取切片，tf.slice(tensor,start_position,get_how_many_data)
		prevTEmbeds = tf.slice(embeds, [0, 0, 0, 0], [-1, args.temporalRange-1, -1, -1])
		nextTEmbeds = tf.slice(embeds, [0, 1, 0, 0], [-1, args.temporalRange-1, -1, -1])
		# 选取id为rows,cols的向量
		rowEmbeds = tf.nn.embedding_lookup(nextTEmbeds, rows)
		colEmbeds = tf.nn.embedding_lookup(prevTEmbeds, cols)
		Q = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		K = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		V = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		q = tf.reshape(tf.einsum('etod,dl->etol', rowEmbeds, Q), [-1, args.temporalRange-1, args.offNum, 1, args.head, args.latdim//args.head])
		k = tf.reshape(tf.einsum('etod,dl->etol', colEmbeds, K), [-1, args.temporalRange-1, 1, args.offNum, args.head, args.latdim//args.head])
		v = tf.reshape(tf.einsum('etod,dl->etol', colEmbeds, V), [-1, args.temporalRange-1, 1, args.offNum, args.head, args.latdim//args.head])
		att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keep_dims=True) / tf.sqrt(float(args.latdim//args.head)), axis=3)
		attV = tf.reshape(tf.reduce_sum(att * v, axis=3), [-1, args.temporalRange-1, args.offNum, args.latdim])
		ret = tf.math.segment_sum(attV * tf.nn.dropout(vals, rate=self.dropRate), rows)
		ret = tf.concat([tf.slice(embeds, [0, 0, 0, 0], [-1, 1, -1, -1]), ret], axis=1)
		return Activate(ret, 'leakyRelu') # area, time, offense, latdim

	def hyperGNN(self, adj, embeds):
		tpadj = tf.transpose(adj)
		# 矩阵相乘
		hyperEmbeds = Activate(tf.einsum('hn,ntod->htod', tf.nn.dropout(adj, rate=self.dropRate), embeds), 'leakyRelu')
		retEmbeds = Activate(tf.einsum('nh,htod->ntod', tf.nn.dropout(tpadj, rate=self.dropRate), hyperEmbeds), 'leakyRelu')
		return retEmbeds
	'''
	1. what are the rows and cols in handler?
	2. how to do the propagate in the temporal? what is the code.
	3. why not one kind of crime one channel
	4. temporal和spatial的edges的表示是一致的，但含义是不同的?
	'''
	def ours(self):
		offenseEmbeds = defineParam('offenseEmbeds', [1, 1, args.offNum, args.latdim], reg=False)
		initialEmbeds = offenseEmbeds * tf.expand_dims(self.feats, axis=-1) # area, time, offense, latdim
		areaEmbeds = defineParam('areaEmbeds', [args.areaNum, 1, 1, args.latdim], reg=False)
		embeds = [initialEmbeds]# + areaEmbeds]
		for l in range(args.spacialRange):
			embed = embeds[-1]
			embed = self.spacialModeling(self.rows, self.cols, self.vals, embed)
			embed = self.hyperGNN(self.hyperAdj, embed) + embed
			embeds.append(embed)
		# add_n输出所有空间步embeds之和，求平均
		embed = tf.add_n(embeds) / args.spacialRange
		embeds = [embed]
		# 跑完spatial后再跑temporal
		for l in range(args.temporalGnnRange):
			embeds.append(self.temporalModeling(self.rows, self.cols, self.vals, embeds[-1]))
		# 再次求平均
		embed = tf.add_n(embeds) / args.temporalGnnRange
		# 计算时间维度上的平均值
		'''为啥在时间维度上求均值呢，而不是用最后求出的信息'''
		embed = tf.reduce_mean(embed, axis=1) # area, offense, latdim
		W = defineParam('predEmbeds', [1, args.offNum, args.latdim], reg=False)
		# 两向量直接对应位置相乘
		if args.task == 'c':
			allPreds = Activate(tf.reduce_sum(embed * W, axis=-1), 'sigmoid') # area, offense
		elif args.task == 'r':
			allPreds = tf.reduce_sum(embed * W, axis=-1)
		# allPreds就是预测值，输出即可
		return allPreds, embed # allpreds 168*4

	def prepareModel(self):
		self.rows = tf.constant(self.handler.rows)
		self.cols = tf.constant(self.handler.cols)
		self.vals = tf.reshape(tf.constant(self.handler.vals, dtype=tf.float32), [-1, 1, 1, 1])
		self.hyperAdj = defineParam('hyperAdj', [args.hyperNum, args.areaNum], reg=True)
		self.feats = tf.placeholder(name='feats', dtype=tf.float32, shape=[args.areaNum, args.temporalRange, args.offNum])
		self.dropRate = tf.placeholder(name='dropRate', dtype=tf.float32, shape=[])

		self.labels = tf.placeholder(name='labels', dtype=tf.float32, shape=[args.areaNum, args.offNum])
		self.preds, embed = self.ours()

		if args.task == 'c':
			posInd = tf.cast(tf.greater(self.labels, 0), tf.float32)
			negInd = tf.cast(tf.less(self.labels, 0), tf.float32)
			posPred = tf.cast(tf.greater_equal(self.preds, args.border), tf.float32)
			negPred = tf.cast(tf.less(self.preds, args.border), tf.float32)
			NNs.addReg('embed', embed * tf.expand_dims(posInd + negInd, axis=-1))
			self.preLoss = tf.reduce_sum(-(posInd * tf.log(self.preds + 1e-8) + negInd * tf.log(1 - self.preds + 1e-8))) / (tf.reduce_sum(posInd) + tf.reduce_sum(negInd))
			self.truePos = tf.reduce_sum(posPred * posInd, axis=0)
			self.falseNeg = tf.reduce_sum(negPred * posInd, axis=0)
			self.trueNeg = tf.reduce_sum(negPred * negInd, axis=0)
			self.falsePos = tf.reduce_sum(posPred * negInd, axis=0)
		elif args.task == 'r':
			self.mask = tf.placeholder(name='mask', dtype=tf.float32, shape=[args.areaNum, args.offNum])
			self.preLoss = tf.reduce_sum(tf.square(self.preds - self.labels) * self.mask) / tf.reduce_sum(self.mask)
			self.sqLoss = tf.reduce_sum(tf.square(self.preds - self.labels) * self.mask, axis=0)
			self.absLoss = tf.reduce_sum(tf.abs(self.preds - self.labels) * self.mask, axis=0)
			self.tstNums = tf.reduce_sum(self.mask, axis=0)
			posMask = self.mask * tf.cast(tf.greater(self.labels, 0.5), tf.float32)
			self.apeLoss = tf.reduce_sum(tf.abs(self.preds - self.labels) / (self.labels + 1e-8) * posMask, axis=0)
			self.posNums = tf.reduce_sum(posMask, axis=0)
			NNs.addReg('embed', embed * tf.expand_dims(self.mask, axis=-1))
		'''
		这里计算loss为什么theta只有hyperAdj呢
		'''
		self.regLoss = args.reg * Regularize() + args.spreg * tf.reduce_sum(tf.abs(self.hyperAdj))
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds):
		idx = batIds[0]
		label = self.handler.trnT[:, idx, :]# area, offNum
		if args.task == 'c':
			negRate = args.negRate#np.random.randint(1, args.negRate*2)
		elif args.task == 'r':
			negRate = 0
		posNums = np.sum(label != 0, axis=0) * negRate
		retLabels = (label != 0) * 1
		if args.task == 'r':
			mask = retLabels
			retLabels = label
		for i in range(args.offNum):
			temMap = label[:, i]
			negPos = np.reshape(np.argwhere(temMap==0), [-1])
			sampedNegPos = np.random.permutation(negPos)[:posNums[i]]
			# sampedNegPos = negPos
			if args.task == 'c':
				retLabels[sampedNegPos, i] = -1
			elif args.task == 'r':
				mask[sampedNegPos, i] = 1
		# 取出由idx-temporalRange到idx的数据切片，当作一条数据
		feat = self.handler.trnT[:, idx-args.temporalRange: idx, :]
		if args.task == 'c':
			return self.handler.zScore(feat), retLabels
		elif args.task == 'r':
			return self.handler.zScore(feat), retLabels, mask

	def trainEpoch(self):
		# 打乱的天数的索引，从30(temporalRange)到总天数(数据的第三维长度)
		ids = np.random.permutation(list(range(args.temporalRange, args.trnDays)))
		epochLoss, epochPreLoss, epochAcc = [0] * 3
		num = len(ids)
		# 一个epoch内有几个batch
		steps = int(np.ceil(num / args.batch))
		for i in range(steps):
			# 当前batch的start和end的下标
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = ids[st: ed]
			# batIds是一个list，list中是一个batch的数据，如[66,32,55]，表示的是一条数据的结束时间索引
			# 生成数据
			tem = self.sampleTrainBatch(batIds)
			if args.task == 'c':
				feats, labels = tem
			elif args.task == 'r':
				feats, labels, mask = tem

			targets = [self.optimizer, self.preLoss, self.loss]
			feeddict = {self.feats: feats, self.labels: labels, self.dropRate: args.dropRate}
			if args.task == 'r':
				feeddict[self.mask] = mask
			res = self.sess.run(targets, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			# 输出训练数据
			log('Step %d/%d: preLoss = %.4f         ' % (i, steps, preLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def sampTestBatch(self, batIds, tstTensor, inpTensor):
		idx = batIds[0]
		# labels是准确值
		label = tstTensor[:, idx, :]# area, offNum
		if args.task == 'c':
			retLabels = ((label > 0) * 1 + (label == 0) * (-1)) * self.handler.tstLocs
		elif args.task == 'r':
			retLabels = label
			mask = self.handler.tstLocs * (label > 0)
		# 如果当前选的idx是测试集较前面的，得取一部分训练集的数据
		if idx - args.temporalRange < 0:
			temT = inpTensor[:, idx-args.temporalRange:, :]
			temT2 = tstTensor[:, :idx, :]
			feats = np.concatenate([temT, temT2], axis=1)
		else:
			feats = tstTensor[:, idx-args.temporalRange: idx, :]
		if args.task == 'c':
			return self.handler.zScore(feats), retLabels
		elif args.task == 'r':
			return self.handler.zScore(feats), retLabels, mask

	def testEpoch(self, tstTensor, inpTensor):
		ids = np.random.permutation(list(range(tstTensor.shape[1])))
		epochLoss, epochPreLoss,  = [0] * 2
		if args.task == 'c':
			epochTp, epochFp, epochTn, epochFn = [np.zeros(4) for i in range(4)]
		elif args.task == 'r':
			epochSqLoss, epochAbsLoss, epochTstNum, epochApeLoss, epochPosNums = [np.zeros(4) for i in range(5)]
		num = len(ids)

		steps = int(np.ceil(num / args.batch))
		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = ids[st: ed]

			tem = self.sampTestBatch(batIds, tstTensor, inpTensor)
			if args.task == 'c':
				feats, labels = tem
			elif args.task == 'r':
				feats, labels, mask = tem

			if args.task == 'c':
				targets = [self.preLoss, self.regLoss, self.loss, self.truePos, self.falsePos, self.trueNeg, self.falseNeg]
				feeddict = {self.feats: feats, self.labels: labels, self.dropRate: 0.0}
			elif args.task == 'r':
				targets = [self.preds, self.preLoss, self.regLoss, self.loss, self.sqLoss, self.absLoss, self.tstNums, self.apeLoss, self.posNums]
				feeddict = {self.feats: feats, self.labels: labels, self.dropRate: 0.0, self.mask: mask}
			res = self.sess.run(targets, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			if args.task == 'c':
				preLoss, regLoss, loss, tp, fp, tn, fn = res
				epochTp += tp
				epochFp += fp
				epochTn += tn
				epochFn += fn
			elif args.task == 'r':
				# 跑出来的结果
				preds, preLoss, regLoss, loss, sqLoss, absLoss, tstNums, apeLoss, posNums = res
				epochSqLoss += sqLoss
				epochAbsLoss += absLoss
				epochTstNum += tstNums
				epochApeLoss += apeLoss
				epochPosNums += posNums
			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['preLoss'] = epochPreLoss / steps
		if args.task == 'c':
			temSum = 0
			for i in range(args.offNum):
				ret['F1_%d' % i] = epochTp[i] * 2 / (epochTp[i] * 2 + epochFp[i] + epochFn[i])
				temSum += ret['F1_%d' % i]
			ret['microF1'] = temSum / args.offNum
			ret['macroF1'] = np.sum(epochTp) * 2 / (np.sum(epochTp) * 2 + np.sum(epochFp) + np.sum(epochFn))
		elif args.task == 'r':
			for i in range(args.offNum):
				ret['RMSE_%d' % i] = np.sqrt(epochSqLoss[i] / epochTstNum[i])
				ret['MAE_%d' % i] = epochAbsLoss[i] / epochTstNum[i]
				ret['MAPE_%d' % i] = epochApeLoss[i] / epochPosNums[i]
			ret['RMSE'] = np.sqrt(np.sum(epochSqLoss) / np.sum(epochTstNum))
			ret['MAE'] = np.sum(epochAbsLoss) / np.sum(epochTstNum)
			ret['MAPE'] = np.sum(epochApeLoss) / np.sum(epochPosNums)
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
		return hit, ndcg
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(self.sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')

# if __name__ == '__main__':
# 	logger.saveDefault = True
# 	config = tf.ConfigProto()
# 	config.gpu_options.allow_growth = True

# 	log('Start')
# 	handler = DataHandler()
# 	log('Load Data')
# 	with tf.Session(config=config) as sess:
# 		model = Model(sess, handler)
# 		# model.run()
# 		with open('Datasets/CHI_crime/tst.pkl','rb') as fs: pre = pickle.load(fs)
# 		args.row, args.col, _, args.offNum = pre.shape
# 		args.areaNum = args.row * args.col
# 		rspFunc = (lambda tensor: np.reshape(tensor, [args.areaNum, -1, args.offNum]))
# 		pre_tensor = rspFunc(pre)
# 		model.run_one(pre_tensor[:,:30,:])
# 		# model.run_one()