import numpy as np 
import collections

def loadDataSet(filename):
	"""
	input: filename: the path of data file.
	output: a numpy array.
	"""

	with open(filename) as fr:
		dataSet = []
		lines = fr.readlines()
		for line in lines:
			dataSet.append(list(map(float, line.strip().split())))
	return np.array(dataSet)

def binarySplit(dataSet, feat_ind, value):
	"""
	input:  dataSet: numpy array;
			feat_ind: index of feature to split
			value: divide dataSet into two subSet. 
			 --> { x | x > value }, { x | x <= value }
	output: two numpy array
	"""
	subSet0 = dataSet[ dataSet[:, feat_ind] > value, :]
	subSet1 = dataSet[ dataSet[:, feat_ind] <= value, :]
	return subSet0, subSet1

def regErr(dataSet):
	return np.var(dataSet[:, -1]) * dataSet.shape[0]

def make_leaf(dataSet):
	return np.mean(dataSet[:, -1])

def createTree(dataSet, criterion=regErr, 
				min_impurity_decrease=0, min_samples_split=2 ):
	"""
	<criterion>: function (default=regErr)
		The function to measure the quality of a split.

	<min_impurity_decrease> : float, optional (default=0.)

		a node will be split if this split induces a decrease
		of the impurity greater than or equal to this value.

	<min_samples_split> : int, optional (default=2)

		The minimum number of samples required to split an internal node
	"""
	feat_ind, val = bestFeat2Split(dataSet, criterion, min_impurity_decrease, min_samples_split)
	if feat_ind is None:
		return val
	regTree = {'splitInd': feat_ind, 'splitVal': val}
	left, right = binarySplit(dataSet, feat_ind, val)
	regTree['left'] = createTree(left, criterion, min_impurity_decrease, min_samples_split)
	regTree['right'] = createTree(right, criterion, min_impurity_decrease, min_samples_split)
	return regTree



def bestFeat2Split(dataSet, criterion, min_impurity_decrease, min_samples_split):
	if len(set(dataSet[:,-1])) == 1:
		return None, dataSet[:, 0]

	m, n = dataSet.shape
	score = criterion(dataSet)
	minScore = np.inf;  
	bestFeatInd = None
	bestVal = make_leaf(dataSet)

	for feat_ind in range(n-1):
		for val in set(dataSet[:, feat_ind]):
			left, right = binarySplit(dataSet, feat_ind, val)
			if left.shape[0] < min_samples_split or right.shape[0] < min_samples_split:
				continue
			curScore = criterion(left) + criterion(right)
			if curScore < minScore:
				minScore = curScore
				bestFeatInd = feat_ind
				bestVal = val
	if score - minScore < min_impurity_decrease:
		return None, make_leaf(dataSet)
	return bestFeatInd, bestVal

def isTree(obj):
	return (type(obj).__name__ == 'dict')

def getMean(tree):
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return ( tree['right'] + tree['left'] ) / 2


	
if __name__ == '__main__':
	dataSet = loadDataSet('ex0.txt')
	regTree = createTree(dataSet, min_samples_split=4,min_impurity_decrease=0)
	print( (regTree) )