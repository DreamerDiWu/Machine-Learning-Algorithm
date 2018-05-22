import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import bisect
import warnings

warnings.filterwarnings('ignore')

cut_off = namedtuple('cut_off', 'axis value label')

def loadDataSet(filename):
    dataSet = []; labels = []
    with open(filename) as fr:
        for line in fr.readlines():
            string = line.strip().split('\t')
            dataSet.append([float(s) for s in string[:-1] ])
            labels.append(string[-1])
    return np.array(dataSet), labels

def train_valid_split(dataSet, labels, prob=0.8, seed=1):
    np.random.seed(seed)
    random_ind = np.random.permutation(len(dataSet))
    brk = int(len(dataSet)*prob)
    random_set = dataSet[random_ind, :]
    random_labels = [labels[i] for i in random_ind]
    return dataSet[:brk, :], labels[:brk], dataSet[brk:, :], labels[brk:]

def Euclid_dist(item1, item2):
    """
    input: item1 :vector, item2: vector/ matrix

    output: Euclid distance of 'item1' and 'item2'
    """
    if type(item1).__name__ != 'ndarray':
        item1 = np.array(item1)
    if type(item2).__name__ != 'ndarray':
        item2 = np.array(item2)

    return float(np.sqrt(np.sum( ( item1 - item2 ) ** 2)))

class TreeNode():
    def __init__(self, left=None, right=None, value=None):
        self.left = left
        self.right = right
        self.cut_off = None

    def num_nodes(self):
        if not self:
            return 0
        if self.left is None and self.right is None:
            return 1
        num_nodes = 1
        if self.left:
            num_nodes += self.left.num_nodes()
        if self.right:
            num_nodes += self.right.num_nodes()
        return num_nodes


class OrderedList():
    """
    a queue that always keep the order and if it reaches its max length,
    insert and then pop the last element(which is also the max element)
    """
    def __init__(self, maxlen):
        self._maxlen = maxlen
        self._orderedList = []

    def append(self, item):
        bisect.insort(self._orderedList, item)
        if len(self) > self._maxlen:
            self._orderedList.pop()

    def size(self):
        return self._maxlen

    def __len__(self):
        return len(self._orderedList)

    def items(self):
        return self._orderedList

    def max(self):
        return self._orderedList[-1]


class KNN:

    def _binarySplit(dataSet, axis, value):
        """
        return 2 subsets splited by axis and value exclude value itself
        """
        left_row_ind = dataSet[:,axis] < value
        right_row_ind = dataSet[:, axis] > value
        return dataSet[left_row_ind, :], dataSet[right_row_ind, :], 

    @staticmethod
    def _generateKdTree(dataSet, depth):
        """
        generate a kd tree recursively
        axis to split = depth mod dims 
        """
        # there is no data
        m = len(dataSet)
        if m == 0:
            return None
        treenode = TreeNode()

        axis = depth % (dataSet.shape[1] - 1)
        median_on_axis = sorted(dataSet[:, axis])[m // 2]
        value_idx = dataSet[:,axis]==median_on_axis
        treenode.cut_off = cut_off(axis, list(dataSet[value_idx, :-1][0]), 
                                    int(dataSet[value_idx, -1][0]))

        left, right = KNN._binarySplit(dataSet, axis, median_on_axis)
        treenode.left = KNN._generateKdTree(left, depth+1)
        treenode.right = KNN._generateKdTree(right, depth+1)

        return treenode

    @staticmethod

    # OrderedList to store k best point

    # if abs(target[axis] - value[cut_off[axis]]) < min_dist:
        # find another child
    def _findBestK(kdtree, target, dist_crit, bestK):
        """
        find k nearest examples of target, and store the examples in 'bestK' .
        <dist_crit> : function to calculate distance between two examples.
        <bestK>: a OrderedList to store k nearest examples.
        """
        if kdtree is None:
            return
        axis, value, label = kdtree.cut_off

        if kdtree.left is None and kdtree.right is None:
            # reach the leaf return
            dist = dist_crit(value, target)
            bestK.append((dist, value, label))
            return
     
        if target[axis] < value[axis]:
            KNN._findBestK(kdtree.left, target, dist_crit, bestK)
        else:
            KNN._findBestK(kdtree.right, target, dist_crit, bestK)

        # back to the parent 
        dist = dist_crit(value, target)
        bestK.append((dist, value, label))
        
        if abs(target[axis] - value[axis]) < bestK.max()[0] or \
            len(bestK) < bestK.size():
            if target[axis] < value[axis]:
                KNN._findBestK(kdtree.right, target, dist_crit, bestK)
            else:
                KNN._findBestK(kdtree.left, target, dist_crit, bestK)  


    def __init__(self, dist_crit=Euclid_dist, scaler=None):
        self._dist_crit = dist_crit
        self._kdTree = None
        self._encoder = None
        self.scaler = None

    def kdTreePrint(self):
        
        def traversal(root):
            kdTree = {}
            if root is None:
                return None
            kdTree['root'] = root.cut_off
            if root.left:
               kdTree['left'] = traversal(root.left)
            if root.right:
               kdTree['right'] = traversal(root.right)
            return kdTree
        print(traversal(self._kdTree))
        
    def demo_set(self):
        """
        generate a demo set
        """
        dataSet = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ["A", "A", "B", 'B']
        return dataSet, labels

    def fit(self, dataSet, labels):
        if type(dataSet).__name__ != 'ndarray':
                raise TypeError('dataSet must be a numpy array.')
        if self.scaler:           
            self.scaler.fit(dataSet)
            dataSet = self.scaler.transform(dataSet)

        if type(labels[0]) == str:
            self._encoder = LabelEncoder()
            self._encoder.fit(labels)
            encoded_labels = self._encoder.transform(labels)
            dataSet_with_label = np.c_[dataSet, encoded_labels]
        else:
            dataSet_with_label = np.c_[dataSet, np.array(labels)]

        self._kdTree = KNN._generateKdTree(dataSet_with_label, 0)
        print("kd Tree generated!")

    def predict(self, testSet, k):
        if self._kdTree is None:
            raise TypeError("kd-Tree cannot be 'NoneType', use '.fit()' first.")
        if type(testSet).__name__ != 'ndarray':
            raise TypeError('testSet must be a numpy array.')
        if k > self._kdTree.num_nodes():
            print("warning: value of 'k' is greater than the total nodes in kd-Tree, \
                \rset 'k' to maximum nodes {}".format(self._kdTree.num_nodes()))
            k = self._kdTree.num_nodes()

        if self.scaler:
            testSet = self.scaler.transform(testSet)
        results = []
        for example in testSet:
            bestK = OrderedList(k)
            KNN._findBestK(self._kdTree, example, self._dist_crit, bestK)
            labels = [point[-1] for point in bestK.items()]
            results.append(Counter(labels).most_common(1)[0][0])
        return self._encoder.inverse_transform(results)


if __name__ == '__main__':
    dataSet, labels = loadDataSet('datingTestSet.txt')
    train_X, train_y, valid_X, valid_y = train_valid_split(dataSet, labels, 0.8, seed=1)

    knn_clf = KNN(scaler=MinMaxScaler())
    knn_clf.fit(train_X, train_y)

    preds = knn_clf.predict(valid_X, 2)
    compare = [pred != true_label for pred, true_label in zip(preds, valid_y)]
    error_rate = sum(compare) / len(compare)
    print('error_rate: {}, total: {}'.format(error_rate, len(compare)))


 
    
