import numpy as np 
import operator
from collections import namedtuple

cut_off = namedtuple('cut_off', 'dim threshold infix_opt')

def demo_set():
	dataSet = np.array([[1.0, 2.1],
						[2.0, 1.1],
						[1.3, 1.0],
						[1.0, 1.0],
						[2.0, 1.0]])
	labels = np.array([1, 1, -1, -1, 1]).reshape(-1, 1)
	return dataSet, labels



class DecisionStump():
	def __init__(self, dim, threshold, infix_opt='lt'):
		"""
		Parameters:
			"threshold": int or float
			"infix_opt": str or callable(default 'lq')
			if callable: call infix_opt(x, threshold)
			if <str>: if infix_opt is a legal infix operation like 'lt','le'
		"""
		if callable(infix_opt):
			self._infix_opt = infix_opt

		elif hasattr(operator,infix_opt):
				
			self._infix_opt = getattr(operator, infix_opt)
			
		else:
			raise TypeError("'infix_opt' must be str or callable(default 'lq')")

		if type(threshold) != str :
			self._threshold = threshold
		else:
			raise TypeError("'threshold': int or float, input type:'{}'"
				.format(type(threshold).__name__))

		self._dim = dim

	def stumpInfo(self):
		info = cut_off(dim=self._dim, threshold=self._threshold,infix_opt=self._infix_opt)
		return info

	def predict(self, dataSet):
		"""
		if satisfy infix_opt(x, threshold) then set x to -1(default 1)
		"""
		dim, threshold, infix_opt = self.stumpInfo()
		result = np.ones((dataSet.shape[0], 1))
		bool_idx = infix_opt(dataSet[:, dim], threshold)
		result[bool_idx] = -1
		return result


	def __hash__(self):
		return hash(self.stumpInfo())

class Adaboost():

	@staticmethod
	def _weighted_01error_scaler(true_labels, preds, weight):
		if true_labels.shape != preds.shape:
			raise ValueError("shape {} and {} not matches, check input"
							.format(true_labels.shape, preds.shape))
		num_incorrect = np.dot(weight, (true_labels != preds))
		error_rate = num_incorrect / np.sum(weight)
		with np.errstate(divide='ignore'):
			scaler = np.sqrt( (1 - error_rate) / error_rate )
			return float(scaler)


	@staticmethod
	def _getStump(dataSet, labels, weight):
		m, n = dataSet.shape
		max_scaler = float('-Inf')
		bestStump = None
		bestPreds = None
		for feat_ind in range(n):
			for val in set(dataSet[:, feat_ind]):
				for infix_opt in ['lt','ge']:
					stump = DecisionStump(dim=feat_ind, threshold=val, infix_opt=infix_opt)
					preds = stump.predict(dataSet)
					scaler = Adaboost._weighted_01error_scaler(labels,preds,weight)
					if scaler > max_scaler:
						max_scaler = scaler
						bestStump = stump
						bestPreds = preds
		return bestStump, max_scaler, bestPreds
	def __init__(self, n_stumps):
		self._weight = None
		self._Stumps = {}
		self.n_stumps = n_stumps

	def fit(self, dataSet, labels):
		m, n = dataSet.shape
		self._weight = 1/m * np.ones((1, m))
		for i in range(self.n_stumps):
			bestStump, max_scaler, bestPreds = Adaboost._getStump(dataSet, 														labels, self._weight)
			stump_weight = np.log(max_scaler)
			self._Stumps[bestStump] = stump_weight
			incorrect_idx = (labels != bestPreds).reshape(1, m)
			correct_idx = True ^ incorrect_idx
			# scale up incorrect, scale down correct
			self._weight[incorrect_idx] *= max_scaler
			self._weight[correct_idx] /= max_scaler

	def predict(self, dataSet):
		if type(dataSet[0]).__name__ != 'ndarray':
			raise TypeError("input only accept 2-dimension array")
		preds = []

		agg_score = 0.0
		for stump, stump_weight in self._Stumps.items():
			pred = stump.predict(dataSet)
			agg_score += pred * stump_weight
			#print(agg_score)
		preds.append(np.sign(agg_score))
		return np.array(preds).reshape(-1, 1)




if __name__ == '__main__':
	ab_clf = Adaboost(n_stumps=3)

	# ds = DwaecisionStump(threshold=1.5)
	dataSet, labels = demo_set()
	# res = ds.predict(dataSet, dim=0)
	# print(res)
	ab_clf.fit(dataSet, labels)
	for stump, weight in ab_clf._Stumps.items():
		print(stump.stumpInfo(), weight)
	x = np.array([[0, 0], [5,5]])
	preds = ab_clf.predict(x)
	print("Predictions:")
	for examp, pred in zip(x, preds):
		print('\t{} --> {}'.format(examp, pred))