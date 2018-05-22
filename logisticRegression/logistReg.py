import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['axes.facecolor'] = '0.85'
plt.rcParams['axes.grid'] = True

def loadDataSet(filename):
	dataSet, labels = [], []
	with open(filename) as fr:
		for line in fr.readlines():
			example = list(map(float, line.strip().split()))
			dataSet.append(example[:-1])
			labels.append(example[-1])
	return np.array(dataSet), np.array(labels).reshape(-1,1)

class LogisticRegression():
	@staticmethod
	def sigmoid(x):
		return 1.0 / (1+np.exp(-x))

	def __init__(self, learning_rate, iterations, 
					optimazition, learning_curve=True):
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.optimazition = optimazition
		self.learning_curve = learning_curve
		self._weight = None
		

	def weights(self):
		return self._weight

	def _gradDescent(self, dataSet, labels):
		m, n = dataSet.shape
		X = np.c_[np.ones((m, 1)), dataSet]
		y = np.array(labels).reshape(m, 1)
		self._weight = np.ones((n+1, 1))

		error_records = []
		for i in range(self.iterations):
			theta = LogisticRegression.sigmoid(np.dot(X, self._weight))
			error = theta - y
			gradient = np.dot(X.T, error) 
			self._weight -= self.learning_rate * gradient
			if i % (self.iterations*0.05) == 0:
				mse_score = float(np.mean(np.square(error)))
				print("error score:", mse_score)
				error_records.append(mse_score)
		if self.learning_curve:
			plt.figure()
			x = np.arange(0, self.iterations, self.iterations // len(error_records))
			plt.plot(x, error_records, label='learning curve')
			plt.xlabel("iterations")
			plt.ylabel("mse_score")
			plt.legend()
			plt.show()

	def _stocGradDescent(self, dataSet, labels):
		m, n = dataSet.shape
		X = np.c_[np.ones((m, 1)), dataSet]
		y = np.array(labels).reshape(m, 1)
		self._weight = np.ones((n+1, 1))

		error_records = []
		for i in range(self.iterations):
			batch_error = 0
			for randInd in np.random.permutation(m):
				rand_sample = X[randInd, :].reshape(1, -1)
				theta = LogisticRegression.sigmoid(np.dot(rand_sample, self._weight))
				error = theta - y[randInd]
				gradient = np.dot(rand_sample.T, error) 
				self._weight -= self.learning_rate * gradient
				batch_error += error
			if i % (self.iterations*0.05) == 0:
				mse_score = float(np.square(batch_error) / m)
				print("error score:", mse_score)
				error_records.append(mse_score)
		if self.learning_curve:
			plt.figure()
			x = np.arange(0, self.iterations, self.iterations // len(error_records))
			plt.plot(x, error_records, label='learning curve')
			plt.xlabel("iterations")
			plt.ylabel("mse_score")
			plt.legend()
			plt.show()

	def decision_boundary(self, dataSet, labels):
		weights = self.weights()
		if weights is None or weights.shape[0] != 3:
			raise ValueError("cannot plot if it's not 2D")
		plt.figure()
		x = dataSet[:, 0]
		y = dataSet[:, 1]
		plt.scatter(x, y, c=labels.squeeze())
		x_lo, x_hi = np.min(x)*0.9, np.max(x)*1.1
		y_lo, y_hi = np.min(y)*0.9, np.max(y)*1.1
		plt.xlim(x_lo, x_hi)
		plt.ylim(y_lo, y_hi)
		boundary_x = np.arange(x_lo, x_hi, (x_hi-x_lo)/500)
		boundary_y = -(weights[0] + weights[1]*boundary_x) / weights[2]
		plt.plot(boundary_x, boundary_y, color='blue', label='decision_boundary') 
		plt.legend()
		plt.show()


	def fit(self, dataSet, labels, optimazition='GD'):
		if self.optimazition == 'GD':
			self._gradDescent(dataSet, labels)
		elif self.optimazition == 'SGD':
			self._stocGradDescent(dataSet, labels)
		else:
			raise AttributeError("optimazition only accept \
								['GD','SGD']")

	def predict(self, dataSet):
		if type(dataSet[0]).__name__ != 'ndarray':
			raise TypeError("input only accept 2d-numpy.ndarray")
		weights = self.weights()
		new_X = np.c_[np.ones((dataSet.shape[0],1)), dataSet]
		score = LogisticRegression.sigmoid(np.dot(new_X, weights))
		positiveClass_idx = score >= 0.5
		preds = np.zeros(score.shape)
		preds[positiveClass_idx] = 1
		return preds


	
if __name__ == '__main__':
	dataSet, labels = loadDataSet('testSet.txt')
	logistReg = LogisticRegression(learning_rate = 0.1, iterations=100, 
									optimazition='SGD')
	logistReg.fit(dataSet, labels)
	print(logistReg.weights())

	logistReg.decision_boundary(dataSet, labels)
