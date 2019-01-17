# Manifold-Learning of 3D objects


Most of the data that we encounter in our day-to-day life in its crude form is high dimensional. There are various problems associated in dealing with high dimensional data. 

The problems include: 
* Visualization
* Storage
* Processing time
* Curse of dimensionality etc...

However, any system that generates the data follows certain model and hence we can always bring this data into a representation where it is purely governed by degrees of freedom (dof) that the system generating it allows. 

*Or in other words, the data has an intrinsic dimensionality which may be different from ambient dimensionality (the no. of dimensions in raw data representation).*

Manifold learning deals with learning this intrinsic structure of the data in order to circumvent above mentioned problems associated with large number of dimensions.

In this project, intrinsic dimension of data is found through Grassberger-Procaccia (GP) algorithm. Later, LLE is used to embed this data on a low-dimensional subspace. The other important problem of finding the inverse mapping from low dimensional data to high dimensional data is also explored. RBFs have been used to interpolate the inverse mapping using the discrete set of mappings from high to low dimensions which is obtained through LLE.

Project overview can be found in my presentation [Manifold_learning.pdf](https://www.google.com/). And the implementation details can be found in [this](https://www.google.com/) report.
