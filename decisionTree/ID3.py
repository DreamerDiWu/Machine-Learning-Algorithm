import operator, collections
from math import log

def loadDataSet(filename):
	"""
	input: filename: the path of data file.
	output: a numpy array.
	"""
	with open(filename) as fr:
		dataSet = []
		lines = fr.readlines()
		for line in lines:
			try:
				dataSet.append(list(map(float, line.strip().split())))
			except ValueError:
				dataSet.append(line.strip().split('\t'))
	return dataSet

class TreeNode():
	def __init__(self):
		self.directions = {}
		self.feat = None
		self.featname = None

class ID3():
	@staticmethod
	def _calEntropy(dataSet):
		"""
		input: a dataSet include label columns --> (X, y).
		output: the entropy of whole dataSet.

		"""

		prob_dic = collections.Counter((example[-1] for example in dataSet))
		num_labels = len(dataSet)
		for key in prob_dic.keys():
			prob_dic[key] /= num_labels
		entropy = sum(map(lambda x: -x*log(x,2), prob_dic.values()))
		return entropy

	@staticmethod
	def _splitDataSet(dataSet, feat_ind, value):
		"""
		input:  dataSet ... 
				feat_ind is the index of feature to split
				value is how to split dataset with feature been choosen
		output: return dataSet which 'feat_ind'th feature == value

		"""
		retSet = []
		for sample in dataSet:
			if sample[feat_ind] == value:
				vec = sample[:]
				del vec[feat_ind]
				retSet.append(vec)
		return retSet

	def demo_set(self):
		"""
		input: None.
		output: a Dataset created manually with feature name.

		"""
		dataSet = [[1, 1, "yes"],
					[1, 1, "yes"],
					[1, 0, 'no'],
					[0, 1, 'no'],
					[0, 1, 'no']]
		featNames = ["no surfacing", "flippers"]
		return dataSet, featNames


	@staticmethod
	def _bestFeat2split(dataSet):
		"""
		input: dataSet..
		output: bestFeat: return the best feature to split dataSet that has maxInfoGain

		"""

		baseEntropy = ID3._calEntropy(dataSet)
		featNum = len(dataSet[0]) - 1
		maxInfoGain = 0.0
		for feat_ind in range(featNum):
			uniqVal = set(sample[feat_ind] for sample in dataSet)
			newEntropy = 0.0
			for value in uniqVal:
				subSet = ID3._splitDataSet(dataSet, feat_ind, value)
				newEntropy += ID3._calEntropy(subSet) * (len(subSet) / len(dataSet))
			curInfoGain = baseEntropy - newEntropy
			if curInfoGain > maxInfoGain:
				maxInfoGain = curInfoGain
				bestFeat = feat_ind
		return bestFeat

	@staticmethod
	def _createTree( dataSet, featNames=None ):
		"""
		input:  dataSet: ...
			featNames: list of feature names in dataSet(default None)
		output: a Tree with subTrees of subSet been splited into by best feature
				with loss and number of leaves

		"""
		if featNames is None:
			featNames = ['X'+str(i) for i in range(len(dataSet[0])-1)]

		num_leaves = 0
		loss = 0

		def createTree(dataSet, featNames=None):
			nonlocal num_leaves, loss
			root = TreeNode()


			labels = [example[-1] for example in dataSet]
			label_counter = collections.Counter(labels)

			# all examples have same label or only one feature left
			if len(label_counter) == 1 or len(featNames) == 0:
				num_leaves += 1
				loss += len(dataSet)*ID3._calEntropy(dataSet)

				return label_counter.most_common()[0][0]

			bestFeat = ID3._bestFeat2split(dataSet)
			uniq_val = {example[bestFeat] for example in dataSet}
			root.feat = bestFeat
			root.featname = featNames[bestFeat]
			for val in uniq_val:
				subSet = ID3._splitDataSet(dataSet, bestFeat, val)
				featNames_copy = featNames[:]
				del featNames_copy[bestFeat]
				root.directions[str(val)] = createTree( subSet, featNames_copy )

			return root

		root = createTree(dataSet, featNames)
		return root, loss, num_leaves



	def __init__(self, alpha=0):
		"""
		'alpha': positive int or float(default 0), 
			parameter to control complexity of tree
		"""
		self.alpha = alpha
		self._root = None
		self._loss = None
		self._num_leaves = None



	def fit(self, dataSet, featNames=None):
		"""
		'dataSet': dataSet with labels
		'featNames': feature names (default None)
		"""
		if featNames is not None and len(featNames) != len(dataSet[0])-1:
			raise ValueError("length of 'featNames' not matches.")
		self._root, self._loss, self._num_leaves = ID3._createTree(dataSet, featNames)

		if self.alpha > 0:
			print("Pruning..")
			self._prune(dataSet)
		print("Decision Tree generated.")

	def predict(self, dataSet, decision_visual=True):
		"""
		'decision_visual': print decision process message(default True)
						if data set is large, recommend setting it 'False'.
		"""
		if type(dataSet[0]) != list:
			raise TypeError('dataSet only accept 2-D list type.')
		predictions = []
		for example in dataSet: 
			node = self._root
			while hasattr(node, 'feat'):
				val = example[node.feat]
				if decision_visual:
					print('{}: {}'.format(node.featname, val))
					print('     |   ')
				node = node.directions[str(val)]
			if decision_visual:
				print('leaf: {}\n\n'.format(node))
			predictions.append(node)
		return predictions

	def _prune(self, dataSet):
		loss = self._loss
		alpha = self.alpha
		leaves = self._num_leaves

		def prune(root, dataSet):
			nonlocal loss, alpha, leaves
			root_leaves = 0
			root_loss = 0
			if not hasattr(root, 'feat'):
				return root, 1, len(dataSet)*ID3._calEntropy(dataSet)
			uniqVal = {example[root.feat] for example in dataSet}
			for val in uniqVal:
				subSet = ID3._splitDataSet(dataSet, root.feat, val) 
				root.directions[str(val)] ,child_leaves, child_loss = \
								prune(root.directions[str(val)], subSet)
				root_leaves += child_leaves
				root_loss += child_loss
			leaves_decrease = root_leaves - 1
			impurity_increase = len(dataSet)*ID3._calEntropy(dataSet) - root_loss
			# print('impurity_increase',impurity_increase)
			# print('leaves_decrease*alpha', leaves_decrease*alpha)
			if impurity_increase < leaves_decrease*alpha:
				# need to prune
				label_counter = collections.Counter((example[-1] for example in dataSet))
				root = label_counter.most_common()[0][0]
				leaves -= leaves_decrease
				loss += impurity_increase
				# update the leaves and loss, make node to leaf
				return root, 1, len(dataSet)*ID3._calEntropy(dataSet)
			else:
				return root, root_leaves, root_loss

		# start pruning
		self._root, pruned_leaves, pruned_loss = prune(self._root, dataSet)

		print('Pruning Done! \nleaves from {} down to {}'
				.format(self._num_leaves,pruned_leaves)) 
		total_loss = self._loss + self.alpha*self._num_leaves
		pruned_total_loss = pruned_loss + self.alpha*pruned_leaves
		print("total loss from {} down to {}".format(total_loss, pruned_total_loss))
		self._loss = pruned_loss
		self._num_leaves = pruned_leaves


			
	def __repr__(self):
		def Traversal(root):			
			if hasattr(root, 'feat'):
				dic = {}
				dic[root.featname] = {}
				for val, node in root.directions.items():
					dic[root.featname][val] = Traversal(node)
				return dic
			return root
		tree_dic = Traversal(self._root)
		return '{}'.format(tree_dic)

if __name__ == '__main__':
    clf = ID3(alpha=5)
    dataSet = loadDataSet('lenses.txt')
    num_test = 5
    testSet = dataSet[:num_test]
    trainSet = dataSet[num_test:]
    featNames = 'age prescript astigmatic tearRate'.split(' ')
    clf.fit(trainSet, featNames)
    print(clf)
    # testSet = [[1, 1],[0, 1 ]]
    preds = clf.predict(testSet)
    print(preds)

    

