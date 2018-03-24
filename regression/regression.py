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



def standRegres(xMat,yMat):
	xTx = xMat.T*xMat
	if np.linalg.det(xTx) == 0:
		print('this matrix is singular cannot do inverse')
	ws = xTx.I*(xMat.T*yMat.T)
	return ws

def draw(xMat,yMat,yHat):
	srtInd = xMat[:,1].argsort(0)
	xSort = xMat[srtInd][:,0,:]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xSort[:,1],yHat[srtInd].flatten().A.T)
	ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=2,c='red')
	plt.show()

def genMat(n):
	ret = []
	for i in range(n):
		tmp = []
		for j in range(n):
			tmp.append(10*np.random.rand()+1)
		ret.append(tmp)
	return ret

def lwlr(testPoint,xArr,yArr,k=1.0):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	m,n = np.shape(xMat)
	weights = np.mat(np.eye((m)))
	for j in range(m):
		diffMat = testPoint - xMat[j,:]
		weights[j,j] = np.exp(diffMat*diffMat.T/(-2*k**2))
	xTx = xMat.T*(weights*xMat)
	if np.linalg.det(xTx) == 0.0:
		print('this matrix is singular,cannot do inverse')
		return
	ws = xTx.I*(xMat.T*(weights*yMat))
	return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
	m,n = np.shape(testArr)
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i],xArr,yArr,k)
	return yHat

xArr,yArr = loadDataSet('ex0.txt')
yHat = lwlrTest(xArr,xArr,yArr,0.003)
draw(np.mat(xArr),np.mat(yArr),np.mat(yHat).T)