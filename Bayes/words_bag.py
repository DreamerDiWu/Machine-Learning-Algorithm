import re
from functools import reduce
import numpy as np 

class words_bag():

	@staticmethod
	def _punc_filter(words):

		"""
		remove punctuation from words
		"""

		ret = []
		for word in words:
			words_punc_remv = re.sub('[,.!?;:]', '', word[0]) 
			ret.append(words_punc_remv.split())
		return ret
	
	@staticmethod
	def _vocabulary(words):
		"""
		generate vocabulary 
		"""
		def extd(lst1, lst2):
			lst1.extend(lst2)
			return lst1
		all_words = reduce(extd, words)
		return list(set(all_words))

	def __init__(self):
		self._vocabulary = None

	def demo_set(self):
		words = [['my dog has flea problems, help please!'],
				['maybe not to take him to park stupid!'],
				['my dalmation is so cute, I love him.'],
				['stop posting stupid worthless garbage!'],
				['mr licks ate my steak, how to stop him?'],
				['quit buying worthless dog food stupid']]
		labels = [0, 1, 0, 1, 0, 1]
		return words, labels

	def fit(self, words):
		words_punc_remv = words_bag._punc_filter(words)
		self._vocabulary = words_bag._vocabulary(words_punc_remv)

	def transform(self, words):
		#if only one sentence
		if type(words[0]) != list:
			raise ValueError("input must be 2D")
		retVec = []
		words_punc_remv = words_bag._punc_filter(words)
		for words in words_punc_remv:
			wordVec = [0] * len(self._vocabulary)
			for word in words:
				try:
					idx = self._vocabulary.index(word)
					wordVec[idx] = 1
				except ValueError:
					continue
			retVec.append(wordVec)
		return retVec

	# def fit_transform(self, words):
	# 	self.fit(words)
	# 	return self.transform(words)


if __name__ == '__main__':
	wbs = words_bag()
	words, labels = wbs.demo_set()
	wbs.fit(words)
	vec = wbs.transform(words)
	print(np.array(vec))