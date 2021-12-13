import cv2
import numpy as np
import math


def GTM(points1, points2, K):
    distp1 = ComputeDistanceMatrix(points1)
    distp2 = ComputeDistanceMatrix(points2)
    medianp1 = computeMedian(distp1)
    medianp2 = computeMedian(distp2)
    # K = 2
    Q1 = points1.copy()
    Q2 = points2.copy()
    AP1 = BuildMedianKNNGraph3(distp1, K, medianp1)
    AP2 = BuildMedianKNNGraph3(distp2, K, medianp2)

    numIteration = 0
    while (AP1 == AP2).all() == False:
        numIteration = numIteration + 1
        jOut = FindOutlier(AP1, AP2)
        print('count：',numIteration,'delete：',jOut)
        if jOut != 0:
            Q1 = np.delete(Q1, jOut, 0)
            Q2 = np.delete(Q2, jOut, 0)
            distp1 = np.delete(distp1, jOut, 0)
            distp1 = np.delete(distp1, jOut, 1)
            distp2 = np.delete(distp2, jOut, 0)
            distp2 = np.delete(distp2, jOut, 1)
            print('remain pts:',np.shape(Q1)[0])
            medianp1 = computeMedian(distp1)
            medianp2 = computeMedian(distp2)

            AP1 = BuildMedianKNNGraph3(distp1, K, medianp1)
            AP2 = BuildMedianKNNGraph3(distp2, K, medianp2)
    # print('Iteration count:', numIteration)

    conVertices = GetConnectedVertices(AP1)
    newQ1 = []
    newQ2 = []

    for floati in conVertices:
        i = int(floati)-1
        if i>=0:
            newQ1.append([Q1[i][0],Q1[i][1]])
            newQ2.append([Q2[i][0], Q2[i][1]])
    Nodes1 = np.array(newQ1)
    Nodes2 = np.array(newQ2)

    return [Nodes1,Nodes2]

def ComputeDistanceMatrix(points):
    pointsCount = np.shape(points)[0]
    distMatrix = np.zeros((pointsCount, pointsCount))

    for elemID in range(0, pointsCount):
        for elem2 in range(0, pointsCount):
            distMatrix[elemID, elem2] = math.sqrt(
                math.pow((points[elemID, 0] - points[elem2, 0]), 2) + math.pow((points[elemID, 1] - points[elem2, 1]),
                                                                               2))
    return distMatrix


def computeMedian(distMatrix):
    nRows = np.shape(distMatrix)[0]
    temp = distMatrix.reshape(1, -1)
    temp = temp.tolist()[0]
    temp.sort()
    n = len(temp)
    n = (n - n % 2) // 2
    med = temp[n + 1]
    return med


def BuildMedianKNNGraph3(distMatrix, K, medianValue):
    pointsCount = np.shape(distMatrix)[0]
    KNNGraph = np.zeros((pointsCount, pointsCount))
    bigNumber = 10e06

    tmpDistanceMatrix = distMatrix.copy()
    tmpDistanceMatrix2 = distMatrix.copy().tolist()
    for i in range(0, pointsCount):
        for j in range(0, pointsCount):
            if tmpDistanceMatrix[i][j] > medianValue or tmpDistanceMatrix[i][j] == 0:
                tmpDistanceMatrix[i][j] = bigNumber

    tmpDistanceMatrix = tmpDistanceMatrix.tolist()
    for temprow in tmpDistanceMatrix:
        temprow.sort()

    for i in range(0, pointsCount):
        if tmpDistanceMatrix[i][K - 1] != bigNumber:
            for j in range(0, K):
                item = tmpDistanceMatrix[i][j]
                for m in range(0, pointsCount):
                    if item == distMatrix[i][m]:
                        KNNGraph[i][m] = 1
                        KNNGraph[m][i] = 1

    a = 1
    return KNNGraph


def FindOutlier(matrixAP1, matrixAP2):
    R = abs(matrixAP1 - matrixAP2)
    tmp = sum(R)
    jOutlier = 0
    if sum(tmp) != 0:
        nR = np.shape(tmp)[0]
        maxNum = max(tmp)
        for kMax in range(0, nR):
            if tmp[kMax] == maxNum:
                jOutlier = kMax
                break
    return jOutlier


def GetConnectedVertices(matrixAdj):
    tmp = sum(matrixAdj)
    n = 0
    sz = np.shape(tmp)[0]
    tmp1 = np.zeros((1,sz))
    for i in range(0, sz):
        if tmp[i] != 0:
            tmp1[0][n] = i+1
            n = n + 1
    resNodesNumber = tmp1.tolist()[0]
    return resNodesNumber


if __name__ == '__main__':
    points1 = np.array([[10, 10], [11, 50], [25, 25], [32, 8], [43, 40]])
    points2 = np.array([[10, 14], [12, 48], [25, 28], [32, 7], [48, 38]])
    [after1,after2]=GTM(points1, points2, 2)
    print(after1,after2)
