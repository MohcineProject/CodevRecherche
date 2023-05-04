#from keras.datasets  import mnist
from MNIST import MNIST_Diffuse 
import numpy as np 
#import matplotlib.pyplot as plt 
from math import * 


#In this Class, we define a binary classifier using the SVM; it predict if the figure is a certain 
#number or not. We use this method so that we can use a one vs all classifier to classify digits
class Mysvm :

#First, we initialise the parameters : 
# C : The penalisation parameter 
#number  : The number that the SVM should predict if it is correct 
#alpha : the learning rate 
#n  : the number of iterations

    def __init__(self, C , n, alpha , number  ) :
        self.alpha = alpha 
        self.C = C 
        self.n = n 
        self.number = number 
        
#In this function we train module using the data X_e and the Labels Y_e 
    def fit( self , X_e , Y_e ) : 
        #Defining main elements 
        n_s  = X_e.shape [0] # the number of the data 
        self.Lambda = 2/(n_s * self.C ) # we define a factor used in minimizing the cost function
        n_f = X_e.shape[1] *  X_e.shape[2] # we define the number of features
        
        #Creating a list with new shapes from (28 ,28 ) to (784,)
        X = []
        for i in range(n_s) : 
            X.append( MNIST_Diffuse(X_e[i]).plat() )
        
        #creating a list of values 1 (when the label is number) and -1(when it is not) for Y
        Y = []
        for i in range(n_s) :
            if Y_e[i] == self.number : 
                Y.append(1)
            else : 
                Y.append(-1)
        

        #We start by giving w   a random value  
        self.w = np.random.rand(n_f)  


        #defining the objective function
        def f(x) : 
            return np.dot(x , self.w)
        
        #defining the hinge loss
        #hinge_loss =  lambda  i : max(0 , 1 - Y[i]*f(X[i]) ) 
        
        #defining the gradient descent for the hinge loss
        def gradient_hinge_loss(i) : 
            if 1 - Y[i] * f(X[i]) < 0 : 
                return 0 
            else : 
                return -Y[i] *X[i] 
        
        
        #We start the learning procedure 
        for i in range(self.n) : 
            sum = 0
            for j in range(n_s) :  
                sum = self.Lambda*self.w + gradient_hinge_loss(j) + sum 
            self.w  = self.w -  self.alpha*((1/n_s)*sum)

        #We finalise the learning 
        for i in range(n_s) : 
            if Y[i]*f(X[i]) < 1 : 
                self.w = self.w - self.alpha*(self.Lambda*self.w - Y[i]*f(X[i]))
            else :
                self.w = self.w -  self.alpha*(self.Lambda*self.w ) 

    #def test(self) : 
        #print(self.w)
    
    #We use the predict function to test the model
    def predict(self , x) : 
        x = MNIST_Diffuse( x).plat()
        approx = np.dot(x,self.w) 
        return np.sign(approx) 

