from naive_bayes import *
from words_bag import *
import numpy as np
if __name__ == '__main__':

	wb = words_bag()
	words,labels = wb.demo_set()
	wb.fit(words)
	wordsVec = wb.transform(words)
	nb_clf = Naive_Bayes(smoother = 1)
	nb_clf.fit(wordsVec, labels)
	print(np.array(wordsVec))
	new_words = [['love love love reallyyyy'], ['stupid dogs kill them!']]
	words2pred = wb.transform(new_words)
	print(wb._vocabulary)
	print(words2pred)
	pred = nb_clf.predict(words2pred)
	#print(nb_clf._PriorProbDic)
	print(nb_clf._ConditionalProbDic)
	print(new_words, '------->', pred)