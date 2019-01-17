import numpy as np 
from sklearn import manifold 
from RBF import Rbf
import intrinsic_dim as dm
from matplotlib import pyplot as plt
import scipy.io
from nurbs import Surface as ns

def affine(input_data):
	A = 5*np.random.uniform(low = -1, high =1, size = (3,3))
	transformed = np.zeros((1,np.size(input_data)))
	for i in range(np.size(input_data)/3):
		transformed[0,3*i:3*(i+1)] = np.dot(input_data[0,3*i:3*(i+1)],A)

	return transformed
#-----------------------------------------------------------------------------------------------
# Data creation:
surface = scipy.io.loadmat('F:\Semester VIII\IMT\Manifold_code\duck\surface1.mat')
surface = surface['surface1']
surface = surface[0,0]
input_data = surface['points']
sizex = input_data.shape[0]
sizey = input_data.shape[1]
sizez = input_data.shape[2]

input_data = input_data.reshape(1,sizex*sizey*sizez)
cardinality = 1000
D_data = np.zeros((cardinality,np.size(input_data)))
for i in range(cardinality):
    D_data[i,:] = affine(input_data)
    
#D_data[cardinality-1,:] = input_data

D = np.size(input_data)    
#-----------------------------------------------------------------------------------------------
# data stored in NXD matrix
#D = 30
#cardinality = 1000
#D_data = np.random.randn(cardinality,D)
# estimate intrinsic dimension of D_data
#-------------------------------------------------------------------------------------------------
print("D_data prep done\n")

dmin = 1
dmax = 10
x = np.zeros(dmax - dmin + 1)
y = np.zeros(dmax - dmin + 1)
for d in range(dmin,dmax+1):
    d_data = np.random.uniform(low = -1, high = 1, size = (cardinality,d))
    corr_dim = dm.correlation_dim(d_data)
    y[d-dmin] = d 
    x[d-dmin] = corr_dim
# calculate polynomial
z = np.polyfit(x, y, 2)
lookup = np.poly1d(z)

print("lookup creation done\n")
#interp=plt.plot(x,lookup(x),label='curve-fitted')
#actual=plt.plot(x,y,label='actual')
#plt.legend()
#plt.show()
#--------------------------------------------------------------------------------------------------
d = dm.intrinsic_est(lookup,D_data)
#d = int(round(d))
#print("d estimation done\n")
d = 40#9
# LLE over input data(take 'd' as input)
clf = manifold.LocallyLinearEmbedding(n_neighbors=6, n_components=d, method='standard')
d_data = clf.fit_transform(D_data)

print("d_data prep done\n")
# interpolation coeffecients estimation
test_num = 100
A = np.zeros((cardinality-test_num,D))
for i in range(D):
    # print d_data[0:990,:]
    rbfi = Rbf(d_data[0:cardinality-test_num,:],D_data[0:cardinality-test_num,i],epsilon=0.5)
    A[:,i] = rbfi.nodes

print("A prep done\n")
D_out = np.zeros((test_num,D))
# leave-one-out reconstruction from inverse mapping
# d_dim_point: txd
#d_dim_point = d_data[cardinality-test_num:cardinality,:]
#d_dim_point = np.random.randn(t,d)
for i in range(D):
    rbfi.nodes = A[:,i]
    D_out[:,i] = rbfi(d_data[cardinality-test_num:cardinality,:])

print("D_out prep done\n")
print("-------------------------------------------------------------------\n")
#---------------------------------------------------------------------------------------------------------------
surface = scipy.io.loadmat('F:\Semester VIII\IMT\Manifold_code\duck\surface1.mat')
surface = surface['surface1']
surface = surface[0,0]
weights = surface['weights']
measure = np.zeros(test_num)
full_vect = {}
for t in range(test_num):
    print("creating estim surface--%d\n"% t)
    points = D_out[t,:].reshape(sizex,sizey,sizez)
    twoD_list = []
    oneD_list = []
    for i in range(points.shape[1]):
        temp = []
        for j in range(points.shape[0]):
            temp.append(weights[j][i]*[float(points[j][i][0]),float(points[j][i][1]),float(points[j][i][2])])
            oneD_list.append(weights[j][i]*[float(points[j][i][0]),float(points[j][i][1]),float(points[j][i][2])])
        twoD_list.append(temp)
    surf = ns.Surface()
    surf._reset_ctrlpts()
    surf._reset_surface()
    surf._mCtrlPts_sizeU = len(twoD_list)
    surf._mCtrlPts_sizeV = len(twoD_list[0])
    surf._mCtrlPts2D = twoD_list
    surf._mCtrlPts = oneD_list
    surf._mWeights = [1.0] * surf._mCtrlPts_sizeU * surf._mCtrlPts_sizeV
    surf.degree_u = surface['udegree'].reshape(np.size(surface['udegree']))#3
    surf.degree_v = surface['vdegree'].reshape(np.size(surface['vdegree']))#3
    surf.knotvector_u = surface['uknot'].reshape(np.size(surface['uknot']))# [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    surf.knotvector_v = surface['vknot'].reshape(np.size(surface['vknot']))#[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
	# Calculate surface points
    surf.evaluate()
	# Arrange calculated surface data for plotting
    surfpts_x = []
    surfpts_y = []
    surfpts_z = []
    for spt in surf.surfpts:
        surfpts_x.append(spt[0])
        surfpts_y.append(spt[1])
        surfpts_z.append(spt[2])
    surfpts_x = np.array(surfpts_x)
    surfpts_y = np.array(surfpts_y)
    surfpts_z = np.array(surfpts_z)
    full_vect['est'] = np.concatenate((surfpts_x,surfpts_y,surfpts_z))
    print("Estim surface %d created\n"% t)
	#--------------------------------------------------------------------------------------------------------------------------------------
    print("creating orig surface--%d\n"% t)
    points = D_data[t+cardinality-test_num,:].reshape(sizex,sizey,sizez)
    twoD_list = []
    oneD_list = []
    for i in range(points.shape[1]):
        temp = []
        for j in range(points.shape[0]):
            temp.append(weights[j][i]*[float(points[j][i][0]),float(points[j][i][1]),float(points[j][i][2])])
            oneD_list.append(weights[j][i]*[float(points[j][i][0]),float(points[j][i][1]),float(points[j][i][2])])
        twoD_list.append(temp)
    surf = ns.Surface()
    surf._reset_ctrlpts()
    surf._reset_surface()
    surf._mCtrlPts_sizeU = len(twoD_list)
    surf._mCtrlPts_sizeV = len(twoD_list[0])
    surf._mCtrlPts2D = twoD_list
    surf._mCtrlPts = oneD_list
    surf._mWeights = [1.0] * surf._mCtrlPts_sizeU * surf._mCtrlPts_sizeV
    surf.degree_u = surface['udegree'].reshape(np.size(surface['udegree']))#3
    surf.degree_v = surface['vdegree'].reshape(np.size(surface['vdegree']))#3
    surf.knotvector_u = surface['uknot'].reshape(np.size(surface['uknot']))# [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    surf.knotvector_v = surface['vknot'].reshape(np.size(surface['vknot']))#[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
	# Calculate surface points
    surf.evaluate()
	# Arrange calculated surface data for plotting
    surfpts_x = []
    surfpts_y = []
    surfpts_z = []
    for spt in surf.surfpts:
        surfpts_x.append(spt[0])
        surfpts_y.append(spt[1])
        surfpts_z.append(spt[2])
    surfpts_x = np.array(surfpts_x)
    surfpts_y = np.array(surfpts_y)
    surfpts_z = np.array(surfpts_z)
    full_vect['orig'] = np.concatenate((surfpts_x,surfpts_y,surfpts_z))
    print ("Orig surface %d created\n" % t)
	#--------------------------------------------------------------------------------------------------------------------------------------
    error = full_vect['orig'] - full_vect['est']
    measure[t] = np.mean(error**2)/np.mean(full_vect['orig']**2)
    measure[t] = measure[t]*100
    print("-------------------------------------------------------------------\n")


 
# test = D_out[33,:]
# test = test.reshape(sizex,sizey,sizez)
# dictn={}
# dictn['surface_est'] = test
# dictn['surface_real'] = D_data[33+cardinality-test_num,:].reshape(sizex,sizey,sizez)
# scipy.io.savemat("surfacetest1.mat",dictn)
# #test1 = scipy.io.loadmat('surfacetest.mat')
dictn={}
 
dictn['d_40'] = measure
# dictn['d_10'] = scipy.io.loadmat('error_measures.mat')['d_10']
scipy.io.savemat("error_measures_40.mat",dictn)
 