import numpy as np 
import matplotlib.pyplot as plt 

def Distance(x):
	MinDist = 1e20
	MaxDist = 0
	Radius = np.zeros(32)
	for i in range(x.shape[0]-1,0,-1):
		Distances = DistVectPoint(x[0:i,:],x[i,:])
		#print i
		DistanceNoZero = ElimZero(Distances, 1e-10)
		minval = min(DistanceNoZero)
		maxval = max(Distances)
		if MinDist > minval :
			MinDist = minval
		if MaxDist < maxval :
			MaxDist = maxval
	for k in range(32):
		Radius[k] = np.exp(np.log(MinDist)+(k+1)*(np.log(MaxDist)-np.log(MinDist))/32)
	return Radius
	
def ElimZero(Distances,Tolerance):
	SigDist = Distances-Tolerance
	SigDist = ((np.sign(np.sign(SigDist*-1)-0.5))+1)*1e20
	DistanceNoZero = Distances + SigDist
	return DistanceNoZero

def BinFilling(x,Radius):
	NoPoints = x.shape[0]
	BinCount = np.zeros(32)
	for i in range(x.shape[0]-1,0,-1):
		Distances = DistVectPoint(x[0:i,:],x[i,:])
		for j in range(32):
			BinCount[j] = BinCount[j] + CountPoints(Distances,Radius[j])
	BinCount = BinCount/((NoPoints)*(NoPoints-1)/2)
	return BinCount

def DistVectPoint(data,point):
	Diffe = np.zeros((data.shape[0],data.shape[1]))
	for i in range(data.shape[1]):
		Diffe[:,i] = data[:,i] - point[i]
	Diffe = Diffe**2
	Distances = np.sum(Diffe,1)
	Distances = np.sqrt(Distances)
	return Distances

def CountPoints(Distances, Threshold):
	NumofPoints = np.size(Distances)
	ThresholdMatr = np.ones(NumofPoints)*Threshold
	CountVect = np.sum(Distances<ThresholdMatr)
	return CountVect

def Slope(Radius,BinCount,centre,high):
	lnRadius = np.log(Radius)
	lnBinCount = np.log(BinCount)
	Max = 0
	Min = lnBinCount[0]
	IntervalHigh = (Max-Min)*high 
	Top = -((Max-Min)*(1-centre)) + (IntervalHigh/2)
	Base = -((Max-Min)*(1-centre)) - (IntervalHigh/2)
	RelDataX = []
	RelDataY = []
 
	for i in range(32):
		if ((lnBinCount[i] >= Base) and (lnBinCount[i]<=Top)):
			RelDataX.append(lnRadius[i])
			RelDataY.append(lnBinCount[i])
			
	RelDataX = np.array(RelDataX)
	RelDataY = np.array(RelDataY)
	P = np.polyfit(RelDataX,RelDataY,1)
	Slope = P[0]
	return Slope


def correlation_dim(d_data):
	x = d_data
	Radius = Distance(x)
	BinCount = BinFilling(x,Radius)
	RadiusNormal = Radius/Radius[31]
	plt.loglog(RadiusNormal,BinCount,basex=np.e,basey=np.e)
	Slope = Slope(Radius,BinCount,0.6,0.125)
	plt.show()	
	return Slope