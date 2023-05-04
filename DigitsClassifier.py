from classification import Mysvm 
from keras.datasets import mnist 
import numpy as np 
import matplotlib.pyplot as plt 

(X_train , Y_train) , (X_test,Y_test) = mnist.load_data() 
X_train = X_train[:400]
Y_train = Y_train[:400]
Y_test  = Y_test [:400]
X_test  = X_test [:400] 

 #In this class we will be using a one versus all classification 
class DigitsClassifier() :

    #In the initialisation method we define : 
    #C     : regularisation factor 
    #n     : number of iterations 
    #Alpha : the learning rate
    #SVMi  : the binary classifier for the number i 
    #SVML  : The list containing all the classifiers 
    def __init__(self ,C , n , Alpha ) : 
    
        self.C = C 
        self.n = n 
        self.Alpha = Alpha 
        self.SVM0 = Mysvm(C , n , Alpha , 0 )
        self.SVM1 = Mysvm(C , n , Alpha , 1 )
        self.SVM2 = Mysvm(C , n , Alpha , 2 )
        self.SVM3 = Mysvm(C , n , Alpha , 3 )
        self.SVM4 = Mysvm(C , n , Alpha , 4 )
        self.SVM5 = Mysvm(C , n , Alpha , 5 )
        self.SVM6 = Mysvm(C , n , Alpha , 6 )
        self.SVM7 = Mysvm(C , n , Alpha , 7 )
        self.SVM8 = Mysvm(C , n , Alpha , 8 )
        self.SVM9 = Mysvm(C , n , Alpha , 9 )
        self.SVML = [self.SVM0 ,self.SVM1 , self.SVM2 , self.SVM3 , self.SVM4 , self.SVM5 , self.SVM6 , self.SVM7 , self.SVM8 , self.SVM9   ]


 

    #training the classifiers 
    def fit(self, X_train, Y_train) : 
        self.SVM0.fit(X_train , Y_train)
        self.SVM1.fit(X_train , Y_train)
        self.SVM2.fit(X_train , Y_train)
        self.SVM3.fit(X_train , Y_train)
        self.SVM4.fit(X_train , Y_train)
        self.SVM5.fit(X_train , Y_train)
        self.SVM6.fit(X_train , Y_train)
        self.SVM7.fit(X_train , Y_train)
        self.SVM8.fit(X_train , Y_train)
        self.SVM9.fit(X_train , Y_train)
        self.weights = []  # we define a list where we put all values of the weights of each SVM
        for i in range(10) : 
            self.weights.append(np.linalg.norm(self.SVML[i].w)) 

    #We predict the number represented in X 
    def predict(self, X) : 
        List_of_accepted = [] #We define a list of the SVMs that classify positively the data
        for i in range(10) : 
            if self.SVML[i].predict(X) == 1 : 
                List_of_accepted.append(i)
        prediction =self.weights.index( min([self.weights[i] for i in List_of_accepted])) #we decide the result of the prediction based on the value of the weight
        return prediction

#This is a testing phase where we seperated the training data according to their digits so we can test the result 
         
FILTER = np.where(Y_train == 0)
X_train0 = X_train[FILTER]
Y_train0 = Y_train[FILTER]

FILTER = np.where(Y_train == 1)
X_train1 = X_train[FILTER]
Y_train1 = Y_train[FILTER]

FILTER = np.where(Y_train == 2)
X_train2 = X_train[FILTER]
Y_train2 = Y_train[FILTER]

FILTER = np.where(Y_train == 3)
X_train3 = X_train[FILTER]
Y_train3 = Y_train[FILTER]

FILTER = np.where(Y_train == 4)
X_train4 = X_train[FILTER]
Y_train4 = Y_train[FILTER]
FILTER = np.where(Y_train == 5)
X_train5 = X_train[FILTER]
Y_train5 = Y_train[FILTER]

FILTER = np.where(Y_train == 6)
X_train6 = X_train[FILTER]
Y_train6 = Y_train[FILTER]

FILTER = np.where(Y_train == 7)
X_train7 = X_train[FILTER]
Y_train7 = Y_train[FILTER]

FILTER = np.where(Y_train == 8)
X_train8 = X_train[FILTER]
Y_train8 = Y_train[FILTER]

FILTER = np.where(Y_train == 9)
X_train9 = X_train[FILTER]
Y_train9 = Y_train[FILTER]
    

d = DigitsClassifier(10 , 100 , 0.8) 
d.fit(X_train, Y_train)

print(d.predict(X_train5[1]))

