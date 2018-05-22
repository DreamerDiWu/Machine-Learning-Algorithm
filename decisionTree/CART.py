import numpy as np
from collections import namedtuple, Counter
from copy import deepcopy

cut_off = namedtuple('cut_off', ['feat_index', 'value'])

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
	return np.array(dataSet)

def subsets(lst):
	"""
	accept a list, return all the subset of input list
	"""
	subSet = [[]]
	for element in lst:
		for s in subSet[:]:
			subSet.append(s.copy())
			s.append(element)
	return subSet

def train_valid_split(dataSet, prob=0.8, seed=1):
	np.random.seed(seed)
	random_ind = np.random.permutation(len(dataSet))
	random_set = dataSet[random_ind, :]
	brk = int(len(dataSet)*prob)
	return random_set[:brk, :], random_set[brk:, :]

class TreeNode():
	def __init__(self, value=None, left=None, right=None):
		self.value = value
		self.left = left
		self.right = right
		self.cut_off = None

	def leaves(self):
		def count_leaves(root):
			if root.cut_off is None:
				return 1
			num_leaves = count_leaves(root.left) + count_leaves(root.right)
			return num_leaves
		num_leaves = count_leaves(self.right) + count_leaves(self.left)
		return num_leaves

	@staticmethod
	def findLeafVal(root, feat_vec):
		if root.cut_off is None:
			return (root.value)
		feat_ind, val = root.cut_off
		if type(val).__name__ == 'set':
			if feat_vec[feat_ind] in val:
				ret = TreeNode.findLeafVal(root.left, feat_vec)
			else:
				ret = TreeNode.findLeafVal(root.right, feat_vec)
		else:
			if feat_vec[feat_ind] > val:
				ret = TreeNode.findLeafVal(root.left, feat_vec)
			else:
				ret = TreeNode.findLeafVal(root.right, feat_vec)
		return ret

class DecisionTree():
	@staticmethod
	def _MSE(dataSet):
		"""
		input: accept a numpy array
		output: mean squared error of labels
		"""
		return np.var( dataSet[:, -1] )

	@staticmethod
	def _Gini(dataSet):
		gini_score = 1
		counter = Counter(dataSet[:,-1])
		for value in counter.values():
			gini_score -= np.square(value / len(dataSet))
		return gini_score

	@staticmethod
	def _make_leaf(dataSet, impurity_crit):
		if impurity_crit == DecisionTree._MSE:
			return np.mean( dataSet[:, -1] )
		elif impurity_crit == DecisionTree._Gini:
			counter = Counter(dataSet[:,-1])
			return counter.most_common(1)[0][0]

	@staticmethod
	def _createTree(dataSet, impurity_crit, min_impurity_decrease, min_samples_split):
		"""
		according the feature and value choosen to split, create 
		left tree, right tree recursively

		Argument:
		<impurity_crit>: function (default=MSE)
			The function to measure the impurity of a data set.

		<min_impurity_decrease> : float, optional (default=0.)

			a node will be split if this split induces a decrease
			of the impurity greater than or equal to this value.

		<min_samples_split> : int, optional (default=2)

			The minimum number of samples required to split an internal node

		return TreeNode object or constant 
		"""
		if type(dataSet).__name__ != 'ndarray':
			raise TypeError('input must be a numpy array.')

		treenode = TreeNode()
		feat_ind, val = DecisionTree._bestFeat2split(dataSet, impurity_crit, 
						min_impurity_decrease, min_samples_split)
		if feat_ind is None:
			treenode.value = val
			return treenode
		treenode.cut_off = cut_off(feat_ind, val)
		
		D1, D2 = DecisionTree._binarySplit(dataSet, *treenode.cut_off)

		treenode.left = DecisionTree._createTree(D1, impurity_crit, 
						min_impurity_decrease, min_samples_split)
		treenode.right = DecisionTree._createTree(D2, impurity_crit, 
						min_impurity_decrease, min_samples_split)
		return treenode

	@staticmethod
	def _bestFeat2split(dataSet, impurity_crit, min_impurity_decrease, min_samples_split):
		"""
		choose best feature and value to split data set such that impurity decrease most

		Argument:

		<impurity_crit>: function (default=MSE)
			The function to measure the impurity of a data set.

		<min_impurity_decrease> : float, optional (default=0.)

			a node will be split if this split induces a decrease
			of the impurity greater than or equal to this value.

		<min_samples_split> : int, optional (default=2)

			The minimum number of samples required to split an internal node

		return 
		if cannot be splited: None and leaf_value 
		else return best feature and value

		"""
		m, n = dataSet.shape
		bestFeatInd, bestVal = None, DecisionTree._make_leaf(dataSet, impurity_crit)

		if m < min_samples_split or len(set(dataSet[:,-1])) == 1:
			return bestFeatInd, bestVal

		impurity = m * impurity_crit(dataSet)
		min_impurity = np.inf
		

		for feat_ind in range(n-1):
			if type(dataSet[:, feat_ind][0]) != str:
				uniqVal = set(dataSet[:, feat_ind])
			else:
				uniqVal = map(set, subsets(list(dataSet[:, feat_ind])))
			for val in uniqVal:
				D1, D2 = DecisionTree._binarySplit(dataSet, feat_ind, val)
				if len(D1) < min_samples_split or len(D2) < min_samples_split:
					continue
				new_impurity = len(D1)*impurity_crit(D1) + len(D2)*impurity_crit(D2)
				if impurity - new_impurity < min_impurity_decrease:
					continue
				if new_impurity < min_impurity:
					min_impurity = new_impurity
					bestFeatInd = feat_ind; bestVal = val
		return bestFeatInd, bestVal

	@staticmethod
	def _binarySplit(dataSet, feat_ind, val):
		"""
		through feature index and value, binary split data set 
		"""
		if type(val).__name__ == 'set':
			D1_row_ind = np.array([value in val for value in dataSet[:, feat_ind]])
		else:
			D1_row_ind = dataSet[:, feat_ind] > val
		D2_row_ind = True ^ D1_row_ind
		D1, D2 = dataSet[D1_row_ind, :], dataSet[D2_row_ind, :]
		return D1, D2

	@staticmethod
	def _prune( tree, impurity_crit, dataSet, treeSeq ):
		"""
		get sequence of subtree pruned

		<treeSeq>: a dictionary to store subtree 
					"
					default: treeSeq={'tree':'fully-grown tree', 
						'alpha':0, 'num_leaves':'fully-grown tree'.leaves()}
					"
		"""

		saved = {}

		total_leaf_impurity, num_leaves = DecisionTree._fetch(tree, impurity_crit, dataSet, saved)

		nodes, sets, G = saved['node'], saved['set'], saved['G']

		# choose TreeNode such that g is minimum to prune
		min_g_ind = np.argmin(G)
		node2Prune = nodes[min_g_ind]
		node2Prune.value = DecisionTree._make_leaf(sets[min_g_ind], impurity_crit)
		node2Prune.cut_off = None

		# get a new tree pruned
		treeSeq['alpha'].append(G[min_g_ind])
		treeSeq['tree'].append(tree)
		treeSeq['num_leaves'].append(num_leaves-node2Prune.leaves()+1)

		if not (tree.left.cut_off is None and tree.right.cut_off is None):

			DecisionTree._prune(deepcopy(tree), impurity_crit, dataSet, treeSeq )
		else:
			return

	@staticmethod	
	def _fetch(tree, impurity_crit, dataSet, saved):
		"""
		'saved' accept an empty dictionary to fetch nodes, sets and gt
		"""
		if tree.cut_off is None:
			return len(dataSet)*impurity_crit(dataSet), 1

		else:
			D1, D2 = DecisionTree._binarySplit(dataSet, *tree.cut_off)
			left_impurity, left_leaves = DecisionTree._fetch(tree.left, impurity_crit, D1, saved)
			right_impurity, right_leaves = DecisionTree._fetch(tree.right, impurity_crit, D2, saved)

			# find node and set
			saved.setdefault('node',[]).append(tree)
			saved.setdefault('set', []).append(dataSet)
			# calculate g(t) for current TreeNode
			g = (len(dataSet)*impurity_crit(dataSet)-left_impurity-right_impurity) / \
				(left_leaves + right_leaves - 1)
			saved.setdefault('G',[]).append(g)
			
		return left_impurity+right_impurity, left_leaves+right_leaves


	@staticmethod
	def _bestSubtree(treeSeq, impurity_crit, validSet):
		"""
		<treeSeq>: dictionary contains subtree, alpha and number of leaves

		return best subtree performs on validation set best and its error score

		"""
		def validError(treenode, impurity_crit, validSet):
			"""
			return total error of a subtree in validation set
			"""
			if treenode.cut_off is None:
				if len(validSet):
					if impurity_crit == DecisionTree._MSE:
						return np.sum( np.square( validSet[:, -1] - treenode.value ) )
					else:
						return np.sum( validSet[:,-1] != treenode.value ) / len(validSet)
				else:
					return 0.0

			D1, D2 = DecisionTree._binarySplit(validSet, *treenode.cut_off)
			left_err = validError(treenode.left, impurity_crit, D1)
			right_err = validError(treenode.right, impurity_crit, D2)

			return left_err + right_err

		min_error_score = np.inf
		bestSubtree = None
		count = 0
		for tree, alpha, num_leaves in zip(treeSeq['tree'], treeSeq['alpha'], treeSeq['num_leaves']):
			pred_error = validError(tree, impurity_crit, validSet)
			error_score = pred_error + alpha * num_leaves
			if count % int(0.2*len(treeSeq['tree'])+1) == 0:
				print('subTree {} error score: {}'.format( count,error_score) )
				print('...')
			count += 1
			if error_score < min_error_score:
				min_error_score = error_score
				bestSubtree = tree
		return bestSubtree, min_error_score


	def __init__(self, treeType='reg', min_impurity_decrease=0, min_samples_split=2):
		"""
		Argument:
		<treeType>: accept 'clf' or 'reg' only.
			default('reg') generate regression tree

		<min_impurity_decrease> : float, optional (default=0.)

			a node will be split if this split induces a decrease
			of the impurity greater than or equal to this value.

		<min_samples_split> : int, optional (default=2)

			The minimum number of samples required to split an internal node
		"""
		self._treeType = treeType
		self._model_complexity_args = {'min_impurity_decrease':min_impurity_decrease,
										'min_samples_split':min_samples_split}
		self._root = None
	
	def set_params(self, **kwargs):
		"""
		accpet a dictionary include keywords to update parameters 
		"""
		self._treeType = kwargs.get('treeType', self._treeType)
		for key, value in kwargs.items():
			if key in self._model_complexity_args:
				self._model_complexity_args[key] = value

	def fit(self, dataSet, prune=False, validSet=None):
		"""
		generate decision tree from data set.
		"""
		
		model_args = self._model_complexity_args.copy()
		if prune:
			if type(validSet).__name__ != 'ndarray':
				raise AttributeError("To make pruning, validation set accept 'ndarray'\
					, cannot be {}!".format(type(validSet).__name__))
			# get a fully-grown tree
			model_args['min_impurity_decrease'] = 0
			model_args['min_samples_split'] = 2
		
		if self._treeType == 'reg':
			impurity_crit = DecisionTree._MSE
		elif self._treeType == 'clf':
			impurity_crit = DecisionTree._Gini


		else:
			raise ValueError("Argument 'treeType' accept 'clf' or 'reg' only")
		self._root = DecisionTree._createTree(dataSet, impurity_crit=impurity_crit,
											**model_args)

		print("Decision Tree Generated!")

		if prune:
			print("Pruning...")
			treeSeq = {'tree':[self._root], 'alpha':[0], 'num_leaves': [self._root.leaves()]} 
			pruned_tree = DecisionTree._prune(deepcopy(self._root), impurity_crit, dataSet, treeSeq)
			print('Pruning Done: %d pruned sub tree got' % len(treeSeq['tree']))
			print('choosing best subtree through validation set...')
			bestSubtree, error_score = DecisionTree._bestSubtree(treeSeq, impurity_crit, validSet)
			print('best subtree selected with error score: {}'.format(error_score))

			self._root = bestSubtree

	def predict(self, testSet):
		preds = []	
		for sample in testSet:
			preds.append(TreeNode.findLeafVal(self._root, sample))

		return preds

	def __repr__(self):
		"""
		print decision tree in a dictionary
		"""
		def traversal(root):
			tree = {}
			if root.cut_off is None:
				return (root.value)
			tree['cut_off'] = root.cut_off
			tree['left'] = traversal(root.left)
			tree['right'] = traversal(root.right)
			return tree
		tree = traversal(self._root)

		message = "<Decision Tree> --> {}\n<Number of Leaves>: {}".format(tree, self._root.leaves())
		return message


if __name__ == '__main__':

	dataSet = loadDataSet('lenses.txt')
	trainSet, validSet = train_valid_split(dataSet)
	reg_tree = DecisionTree(treeType='clf')


	reg_tree.fit(trainSet, prune=True, validSet=validSet)
	print(reg_tree)
	preds = reg_tree.predict(validSet)
	for sample, pred in zip(validSet[:,:-1], preds):
		print('{:-<50}> {:<10}'.format(', '.join(sample), pred) )





	
