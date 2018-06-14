# -*- coding: utf-8 -*-
#for my sily girl
#these functiosn slice the orignal data matrix into vectors that keras like
import numpy as np

def slidingwindow_slice(datamat,step,windowsize,vectornum = None):
    """slice the data matrix with a sliding window
       datamat: the row material whose shape is assumed to be 287 x ...
       step: the window moving step
       windowsize: the size of the window
       
       the function slice the matrix in the following way
       matrix of 287 x 100000
       |||||||||||||||||||||||||||||||||||||||
       <---window size---->
       [step]<----window size--->
       
       and stack the vectors into a 4d vector(tensor) whose four dims are:
       (numberoftvectors vectorheight vectorwidth channels )
       where vectorheigth is the same as the matrix height and the vector width 
       is equal to the window size"""

    matheight,matwidth = datamat.shape
    if vectornum is None:
       vectornum = int((matwidth - windowsize)/step) #compute the number of vector
    databuf = [datamat[:,i*step:windowsize+i*step].reshape(matheight,windowsize) for i in  range(vectornum) ] #这个看的懂咩
   
    #build the vector matrix of size (numberofvectors vectorheight vectorwidth)
    vectorstack = np.stack(databuf,axis = 0)
    #add the channel dim to the tensor. channel dim = 1 for your problem
    vectorstackwithchannel = np.expand_dims(vectorstack,3)
    return vectorstackwithchannel
       
       
