import sys
sys.path.append("/home/peiyong/Work/ZhangLe/Caffe-2StreamwithCenterLoss/python")
import caffe
import numpy as np
import scipy.io as sio

def hamdist(str1, str2):
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if int(ch1) != int(ch2):
            diffs += 1
    return diffs

class ECOC_OutputLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        params=eval(self.param_str)
        Cls=params['num_Cls']
        if len(bottom) != 2:
            raise Exception("needs two bottom!")
        if bottom[0].data.shape[1]!=Cls:
            raise Exception('Please include all classifiers!')
        #print(self.param_str)
        
        
    def reshape(self, bottom, top):
        top[0].reshape(1)
        
     
        
        

    def forward(self, bottom, top):
        Y_temp=bottom[0].data;
        nSample=Y_temp.shape[0]
        
       
        Y=sio.loadmat('/home/peiyong/Work/ZhangLe/Caffe-2StreamwithCenterLoss/examples/pycaffe/layers/Y_1000.mat')
        Y=Y['Y']
        Y_final=np.zeros((nSample,1))
       
            
        
        for i in range (0,nSample):
            d_hamming=[hamdist(Y_temp[i,:],Y[j,:]) for j in range(0,Y.shape[0])]
            Y_final[i]=np.argmin(d_hamming)
        acc=0
        for y1, y2 in zip(Y_final, bottom[1].data):
               
	       if int(y1)==int(y2):
                   acc+=1
        top[0].data[0]=acc/(nSample+0.0)
        #print('num data:'+str(nSample)+'\n')
        #print('Predict value:'+str(Y_final)+'\n')
        #print('GT value:'+str(bottom[-1].data)+'\n')
        #print('Accuracy:'+str(top[0].data[0])+'\n')
            
            
            
            
           
        

    def backward(self, top, propagate_down, bottom):
        pass
