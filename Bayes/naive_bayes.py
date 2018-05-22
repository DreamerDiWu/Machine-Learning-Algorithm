from collections import Counter, namedtuple, defaultdict
from math import log

ID = namedtuple('ID', 'feat_ind val label')

def label_split(dataSet):
	"""
	accept dataSet of 2d list
	"""
	if type(dataSet[0]).__name__ != 'list':
		raise TypeError("Only accept dataSet of 2D-list")
	labels = [example.pop() for example in dataSet]
	return dataSet, labels

class Naive_Bayes:

	@staticmethod
	def _PriorProb(labels, smoother):
		"""
		return a dictionary store P(Y=c_k)
		"""
		PriorProbDic = Counter(labels)
		for key in PriorProbDic.keys():
			PriorProbDic[key] = (PriorProbDic[key]+smoother)\
								/(len(labels) + len(PriorProbDic)*smoother)
		return PriorProbDic

	@staticmethod
	def _ConditionalProb(dataSet, labels, smoother):
		"""
		return a dictionary store conditional probability
		"""
		label_counter = Counter(labels)
		ConditionalProbDic = defaultdict(lambda: 0)
		uniqVal = defaultdict(set)

		for feat_ind in range(len(dataSet[0])):
			for label in label_counter.keys():
				for example, ex_label in zip(dataSet, labels):
					val = example[feat_ind]
					uniqVal[feat_ind].add(val)
					id_ = ID(feat_ind, val, label)
					if ex_label == label:
						ConditionalProbDic[id_] += 1
					else:
						ConditionalProbDic[id_] += 0

		for key in ConditionalProbDic.keys():
			
			num_val = len(uniqVal[key.feat_ind])

			ConditionalProbDic[key] = (ConditionalProbDic[key] + smoother) \
									/ (label_counter[key.label] + num_val*smoother)

		return ConditionalProbDic

	def __init__(self, smoother=0):
		self._PriorProbDic = None
		self._ConditionalProbDic = None
		self.smoother = smoother


	def demo_set(self):
		data = [[1, 'S', -1],
				[1, 'M', -1],
				[1, 'M',  1],
				[1, 'S',  1],
				[1, 'S', -1],
				[2, 'S', -1],
				[2, 'M', -1],
				[2, 'M',  1],
				[2, 'L',  1] ,
				[2, 'L',  1],
				[3, 'L',  1],
				[3, 'M',  1],
				[3, 'M',  1],
				[3, 'L',  1],
				[3, 'L', -1]]
		return data

	def fit(self, dataSet, labels):
		self._PriorProbDic = Naive_Bayes._PriorProb(labels, self.smoother)
		self._ConditionalProbDic = Naive_Bayes._ConditionalProb\
									(dataSet, labels,self.smoother)

	def predict(self, new_X):
		if not new_X or type(new_X[0]) != list:
			raise ValueError("illegal input '{}'!, input must be a 2D list".format(new_X))
		if not self._PriorProbDic or not self._ConditionalProbDic:
			raise ValueError("not fit yet, please fit first!")
		predictions = []
		for x in new_X:
			maxProb = float('-Inf')
			pred = -999
			for label in self._PriorProbDic.keys():
				prob = log(self._PriorProbDic[label])
				for feat_ind, val in enumerate(x):
					cur_id = ID(feat_ind, val, label)
					if self._ConditionalProbDic[cur_id]:
						prob += log(self._ConditionalProbDic[cur_id])
				print(prob)
				if prob > maxProb:
					maxProb = prob
					pred = label
			predictions.append(pred)
			
		return predictions


if __name__ == '__main__':

	nb_clf = Naive_Bayes()
	data = nb_clf.demo_set()
	dataSet, label = label_split(data)
	nb_clf.fit(dataSet, label)
	new_x = [[2, 'S'],[1, 'L']]
	pred = nb_clf.predict(new_x)
	print(new_x, '---->', pred)
	nb_clf.smoother = 1
	nb_clf.fit(dataSet, label)
	pred = nb_clf.predict(new_x)
	print(new_x, '---->', pred, '(with smoother %d)'% nb_clf.smoother)
	print(nb_clf._ConditionalProbDic)

	



