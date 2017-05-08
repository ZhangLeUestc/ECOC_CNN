import sys
sys.path.append("/home/peiyong/Work/ZhangLe/Caffe-2StreamwithCenterLoss/python")
import caffe
import numpy as np
import scipy.io as sio


class ECOCLayer(caffe.Layer):
    

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need one inputs to compute new label.")
        #print(self.param_str)
        
    def reshape(self, bottom, top):
        nSample=bottom[0].data.shape[0]
        params=eval(self.param_str)
        num_Cls=params['num_Cls']
        top[0].reshape(*(nSample,num_Cls))
        
     
        
        

    def forward(self, bottom, top):
        Y=sio.loadmat('/home/peiyong/Work/ZhangLe/Caffe-2StreamwithCenterLoss/examples/pycaffe/layers/Y_300.mat')
        Y=Y['Y']
        params=eval(self.param_str)
        num_Cls=params['num_Cls']
        
        for i in range(0,bottom[0].data.shape[0]):
            top[0].data[i,:]=[Y[int(bottom[0].data[i,:,:,:]),j] for j in range(0,num_Cls)]
        #print(' ECOC is:'+str(top[0].data)+'\n')
        
        

    def backward(self, top, propagate_down, bottom):
        pass
