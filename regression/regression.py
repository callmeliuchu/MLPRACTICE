import numpy as np
import matplotlib.pyplot as plt



def loadDataSet(fileName):
	dataMat = []
	labelMat = []
	with open(fileName) as f:
		lines = f.readlines()
		for line in lines:
			tmpArr = line.strip().split('\t')
			tmpArr = list(map(lambda x:float(x),tmpArr))
			dataMat.append(tmpArr[:-1])
			labelMat.append(tmpArr[-1])
	return dataMat,labelMat

xArr,yArr = loadDataSet('ex0.txt')

def standRegres(xMat,yMat):
	xTx = xMat.T*xMat
	if np.linalg.det(xTx) == 0:
		print('this matrix is singular cannot do inverse')
	ws = xTx.I*(xMat.T*yMat.T)
	return ws


def draw(xMat,yMat):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
	plt.show()


xMat = np.mat(xArr)
yMat = np.mat(yArr)
ws = standRegres(xMat,yMat)
# draw(xMat,yMat)
# print(len(xMat[:,1]))
# print(len(yMat.T[:,0]))
# arr = []
# for i in range(len(xMat[:,1])):
# 	x = float(xMat[:,1][i])
# 	y = float(yMat.T[:,0][i])
# 	arr.append([x,y])
# print(arr)
# print(xMat[:,1].flatten().A[0])
# print(len(yMat.T[:,0].flatten().A[0]))
print(ws.T)