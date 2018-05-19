from numpy import *


#loading data
def loaddata(filename):
    num = len(open(filename).readline().split('\t'))-1
    #print(num)
    xmat = []
    ymat = []

    #reading txt file, formatting data into arrays
    f = open(filename)
    for line in f.readlines():
        arr = []
        currline = line.strip().split('\t')
        for i in range(num):
            arr.append(float(currline[i]))
            
        xmat.append(arr)
        ymat.append(float(currline[-1]))
    return xmat, ymat



# implementation of linear regression from scratch
def regres(x,y):
    xmat = mat(x)
    ymat = mat(y).T
    xtx = xmat.T * xmat

    if linalg.det(xtx) == 0.0:
        print('Invertibele')
        return
    ws = xtx.I * (xmat.T * ymat)
    return ws

x, y = loaddata('abalone.txt')
ws = regres(x,y)
xmat = mat(x); ymat = mat(y)
ytemp = xmat*ws

#just for visualisation purposes, to see if any relationship exists between the attributes
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(xmat[:,1].flatten().A[0], ymat.T[:,0].flatten().A[0])
xcopy = xmat.copy()
xcopy.sort(0)
yhat = xcopy*ws
#print(xcopy[:,1])
ax.plot(xcopy[:,1], yhat)
plt.show()
yhat = xmat*ws
arr = corrcoef(yhat.T, y)
print(arr)