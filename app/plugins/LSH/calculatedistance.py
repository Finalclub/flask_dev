import numpy as np

class CalculateDistance(object):
	"""docstring for CalculateDistance
	定义不同的距离计算公式:
	funHammingDistance - 海明距离
	funEuclideanDistance - 欧几里德距离
	funSquaredEuclideanDistance - SED距离
	funEuclideanDistanceCentred - 待研究正确称呼
	funManhattanDistance - norm-l1距离
	funCosineSimilarity - 余弦相似性
	"""
	def __init__(self, inputx,inputy):
		super(CalculateDistance, self).__init__()
		self.inputx = inputx
		self.inputy = inputy
		self.listOfInput = [self.inputx,self.inputy]

	def funHammingDistance(self):
		'''
		计算两个输入之间的hamming距离
		要求输入为0～255的整数,即8bits
		Hamming距离的计算不能采用向量或着矩阵，必须是单个数字
		如果要对上述数据结构计算Hamming距离，就要反复调用函数迭代
		'''
		try:
			for i in range(len(self.listOfInput)):	
				self.listOfInput[i] = format(self.listOfInput[i],'b')
				if len(self.listOfInput[i]) != 8:
					self.listOfInput[i] = '0'*(8 - len(self.listOfInput[i])) + self.listOfInput[i]			
			return str(int(self.listOfInput[0]) ^ int(self.listOfInput[1])).count('1')
		
		except ValueError:
			raise ValueError("Shape of inputs are not match!") 


	def funEuclideanDistance(self):
		'''
		计算两个输入之间的欧式距离
		输入可以为向量
		'''
		try:
			# minusResult = self.listOfInput[0] - self.listOfInput[1]
			return np.linalg.norm(self.listOfInput[0] - self.listOfInput[1])
		except ValueError:
			return "Shape of inputs are not match!"

	def funSquaredEuclideanDistance(self):
		'''
		计算欧式距离的平方
		'''
		try:
			minusResult = self.listOfInput[0] - self.listOfInput[1]
			return np.sum(minusResult)
		except ValueError:
			return "Shape of inputs are not match!"

	def funEuclideanDistanceCentred(self):
		'''
		计算一个类似均匀分布的距离度量方式
		'''
		try:
			minusResult = np.mean(self.listOfInput[0]) - np.mean(self.listOfInput[1])
			return minusResult * minusResult
		except ValueError:
			return "Shape of inputs are not match!"

	def funManhattanDistance(self):
		'''
		计算L1距离
		'''
		try:
			return np.sum(np.abs(self.listOfInput[0] - self.listOfInput[1]))
		except ValueError:
			return "Shape of inputs are not match!"
			
	def funCosineSimilarity(self):
		'''
		计算余弦相似性
		'''
		try:
			return np.sum(np.multiply(self.listOfInput[0],self.listOfInput[1])) / np.sqrt(np.multiply(np.sum(np.power(self.listOfInput[0],2)),np.sum(np.power(self.listOfInput[1],2))))
		except ValueError:
			return "Shape of inputs are not match!"
		
		



