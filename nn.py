import arff 
import numpy as np

##for ploting ROC ,getting accuracy score, recall, precision

from sklearn.metrics import accuracy_score,average_precision_score,precision_score,f1_score 



###defining the functions for  getting accuracy score, recall, precision and F_measure

#in all these functions we pass the Target and the Output to get their respective metrics

def get_accuracy(Target, Output):
    accuracy = accuracy_score( Target,Output, normalize=False)
    return accuracy

def get_Recall(Target, Output):
    Recall = average_precision_score(Target, Output)
    return Recall

def get_Precision(Target,Output):
    Precision = precision_score(Target,Output,average="macro")
    return Precision

def get_F1 (Target, Output):
    F_measure = f1_score(Target,Output,average="macro")    
    return F_measure







#to install this library using pip install liac-arff
data = arff.load(open('data/Training-Dataset.arff'))


dataExtract = data['data']




        
#Sigmoid function for the  forward propagation
def sigTransform(X):
    return 1/(1+np.exp(-X))   

#Derivative of the Sigmoid function for the  backward propagation
def sigDerivative (R):
    return R * (1 - R)

#tanh activation function
def tanhDerivative(Z):
    return (1 - np.power(Z, 2))


def test(inputData, Result, W1,W2,biasW1,biasw2):

      


      
        
    for i in range(1000):


         # Forward Propagation
        
        layer0 = np.dot(inputData,W1) 

        layer1 = np.tanh(layer0)
       
           

        layer1Sum = np.dot(layer1,W2) + biasw2
  
        layer2 = np.tanh(layer1Sum)

    ModelAccuracy = get_accuracy(Result,layer2)   
    ModelRecall = get_Recall(Result,layer2)
    ModelPrecision = get_Precision(Result,layer2)
    ModelF_Measure = get_F1(Result,layer2)  

   
    Measures = []
    Measures[0] = ModelAccuracy
    Measures[1] = ModelRecall
    Measures[2] = ModelPrecision
    Measures[3] = ModelF_Measure
    return 




#Function Train
#This function takes the following parameters
# dataExtract = the data to be split into features and label
# Activation = The activation functio to use, can either be Sigmoid or Tan . The default is Sigmoid
# NoOfNeurons = Used for selecting the number of neurons in the hidden layer. The default is 70
# lossFunction = used for detecting the loss function in backpropagation. The default is Root mean Square ('rms')
# TEst= If set to true the train function will call the test function and split the data into train and test set
#       after splitting and running the model it will print the accuracy, recall, precision, f measure
# Voice = Used to give audio output of the results, though was not implemented in this program. The default is false. will be implemented in next version by me (Emeka Onyebuchi, AI, Uni of Aberdeen, Student ID 51880727 )

# The train function gives the following as output
# Prints the error at every epoch
#At the end of the epoch prints The Target, the Output, Some values of the target, some  values of the output and the root mean square

#how the function works
# The function accepts data and then splits the data into features and label
# the if the activation function is sigmoid it standardizes the feature data i.e to 1 and 0.
# but for tanh activation formular, i standardized between 1 and -1
# We then run forward and backward propagation and display the output

def Train(dataExtract,  Activation='Sigmoid',NoOfNeurons=70,Test=False,lossFunction='rms',Voice=False ):

    # defining the array for the attributes for features
    X =[]

    # defining the array for the attribute for label
    Y= []

    #use a for loop to iterate through all values of the data, extract the features and the label 
    # and store to both X and Y repectively
    for i in range(len(dataExtract)):
        X.append([])
        Y.append([])
        for j in range(len(dataExtract[0])):
       
            if(j >= 0 and j < 30):
         
                X[i].insert(j,dataExtract[i][j])
        else:
           
            Y[i].insert(j, dataExtract[i][j])


    # Make X a numpy array of type float and  assign to A
    A = np.array(X,dtype=float)

    A[A == -1] = 0
     # Make Y a numpy array of type float and  assign to B, which will be the label
    B= np.array(Y,dtype=float)

    # Because the sigmoid function works better with 0 and 1 
    # We will work with the label B
    # the following will be classified as 1 as 1 and -1 as 0
    
    B[B == -1] = 0

    # making sure B is a  numpy array of type float
    B= np.array(B,dtype=float)

    # using np.random.seed makes sure that everywhere each time we run the code we regenerate
    # similar values
    np.random.seed(1)

    #our learning rate
    alpha = 0.00000001 #1st learning rate

    #assign A to our input Data. This are the features for classification
    #Assign B to Result. This will be our feature
    inputData = A
    Result = B 
  
    #generating the weights
    
    # W0, weight is assigned to our Input data
    W0 = inputData

    #W1, Weight 1 for the synapses between the input layer and the hidden layer
    # It is generated by generating random numbers using the numpy random function,
    # the value 30 is for the number of inputs which also the number of input neurons
    #NoOfNeurons is the number of neurons in the hidden layer
    W1 = 2*np.random.random((30,NoOfNeurons))-1

    #W2,weight2 is for thr synapses between the hidden layer and the output layer
    #NoOfNeurons is the number of neurons in the hidden layer
    # the value 1 which is the second parameter is the number of neurons in the output layer
    W2 = 2*np.random.random((NoOfNeurons,1))-1
   
    #biasW1 = This is the matrix for the weights for the bias used in the first layer connection
    #between the input and the hidden layer. It uses the numpy zeros function and passes the 
    #number of neurons in the hidden layer as input
    #biasW2 = This tis the matrix for the weights for the bias used in the second layer 
    #connection between the hidden layer and the output. It uses the numpy zeros function and
    #passes the number of neurons in the hidden layer as input

    biasW1 = np.zeros(NoOfNeurons)
    biasw2= np.zeros(1)
 
    #initiates the following code if the activation function is sigmoid

    if (Activation == 'Sigmoid'):
        #here we pass the number of iterations, which can also be referred to as number 
        #of epochs as the parameter for ranch

        for i in range(50000):


            #####Forward Propagation
            # layer0 = is the sumation of the  dot product of inputs and their respective weights
            #and the bias for the layer
            layer0 = np.dot(inputData,W1)+ biasW1
            
            #layer1 = is the application of the sigmoid function on the  output of layer0
            #here we have gotten values of one part of our neural network
            layer1 = sigTransform(layer0)
       

            #here we are going to evaluate the second part of the feedforward
            #layer1Sum = summation of the dot product of the input from layer 1 and their weights; 
            # and the bias
            layer1Sum = np.dot(layer1,W2) + biasw2


            #layer2 = the application of sigmoid function on layer1Sum and is output is the 
            #result of the forward propagation
            layer2 = sigTransform(layer1Sum) 


            ### Back Propagation
            #now we are done with the forward propagation the next thing is the 
            #back propagation which is trying to send signals to correct errors in output
            #the delta values will be used to adjust the weights

            #Layer2error is the difference between Target and the Output
            layer2_error =  Result - layer2
            
            
            #here we monitor the error in all iterations(or epochs) where the ith_term%1000= 0
            #We print the root mean value
            if(i%1000) == 0 :
   
                print ('Epoch '+ str(i)  +' Error Difference: ' + str(np.mean(np.abs(layer2_error))))
            
            #layer2_delta is the summation of the layer2_error and the sigmoid derivative of layer1
            #with layer2_delta we have the delta value to update weights in second layer
            #layer2_delta = 1/m* layer2_error * sigDerivative(layer2)
            layer2_delta= np.multiply(layer2_error,sigDerivative(layer2)) 

        
            #layer1_error is the dot product  of layer_2 delta and the transpose of the
            # Weights in layer 2
            layer1_error =  layer2_delta.dot(W2.T)


            #layer1_delta is the product of layer1_error and the sigmoid derivative of layer 1
            #layer1_delta = 1/m*layer1_error * sigDerivative(layer1)
            layer1_delta= np.multiply(layer1_error,sigDerivative(layer1))


            #updating the weights 

            #w2 = W2 - alpha*(layer1.T.dot(layer2_delta))
            W2 -= alpha*(layer1.T.dot(layer2_delta))

            #  W1 = W1- alpha*(W0.T.dot(layer1_delta))
            W1 -= alpha*(W0.T.dot(layer1_delta))

    elif (Activation=='Tan'):
         ##here we are basically repeating the processes above except in the following instances
        #  We will discretize the values as 1 and -1 since tanh gives values from -1 to 1 
        #  We will also use the tanh activation funtion for the forward Propagation
        #  We will use the derivative of the tanh for the back back propagation
        R =[]

    
        F= []
        for i in range(len(dataExtract)):
            R.append([])
            F.append([])
            for j in range(len(dataExtract[0])):
       
                if(j >= 0 and j < 30):
         
                    R[i].insert(j,dataExtract[i][j])
                else:
           
                    F[i].insert(j, dataExtract[i][j])

        Q = np.array(R,dtype=float)
        L= np.array(F,dtype=float)

        Q[Q >= 0] = 1
        
        L[L >= 0] = 1
        ###re update this value for tanh
        alpha = 0.0003
        
        np.random.seed(1)
        inputData = Q
        Result = L 

        W0 = inputData

        W1 = 2*np.random.randn(30,NoOfNeurons)-1
        W2 = 2*np.random.randn(NoOfNeurons,1)-1


        biasW1 = np.zeros(NoOfNeurons)
        biasw2= np.zeros(1)
        
        for i in range(10000):


            # Forward Propagation
        
            layer0 = np.dot(inputData,W1)

            layer1 = np.tanh(layer0)
       
           

            layer1Sum = np.dot(layer1,W2) + biasw2
  
            layer2 = np.tanh(layer1Sum) 

            #Back Propagation
            layer2_error =  layer2 - Result
            
            
            if(i%100) == 0 :
   
                print ('Epoch '+ str(i)  +' Error Difference: ' + str(np.mean(np.abs(layer2_error))))
                 
           
            layer2_delta =  np.multiply(layer2_error,tanhDerivative(layer2))
           
        
        
            layer1_error = layer2_delta.dot(W2.T)
            layer1_delta = np.multiply(layer1_error,tanhDerivative(layer1))
            


        #updating the weights well
        
            W2 -= alpha*(layer1.T.dot(layer2_delta))
            W1 -= alpha*(W0.T.dot(layer1_delta))
            
       
   
    
    Root_mean_square = np.mean(np.abs(layer2_error))
    print('-------' * 40)
    print('Target Output')
    print(Result)
    print('Output after Training')
    
    
    print (layer2) 
    print('-------'*40)
    print('++' + str(Result[0]) +'++')
    print('++'+ str(Result[1]) +'++')
    print('++'+ str(Result[2]) +'++')
    print('++'+ str(Result[3]) +'++')
    print('++'+ str(Result[4]) +'++')
    print('++'+ str(Result[5]) +'++')
    print('-------'*40)
    print('++'+ str(layer2[0]) +'++')
    print('++'+ str(layer2[1]) +'++')
    print('++'+ str(layer2[2]) +'++')
    print('++'+ str(layer2[3]) +'++')
    print('++'+ str(layer2[4]) +'++')
    print('++'+ str(layer2[5]) +'++')
    print('-------'*40) 
    print('Error root mean square' +str(Root_mean_square))

    MeasuresA = []
    if(Test=="OK"):
        MeasuresA = Test(inputData,Result,W1,W2,biasW1,biasw2)
        print('ohhh Measure A')
    print('-------' * 40)
    print('Target Output')
    print(Result)
    print('Output after Training')
    print (layer2) 
    print('-------'*40)
    print ('Accuracy -- '+str(MeasuresA)) 
    print('-------'*40)
    print ('Recall' +str(MeasuresA[1])) 
    print('-------'*40)
    print ('Precision' +str(MeasuresA[2])) 
    print('-------'*40)
    print ('F_measure --' +str(MeasuresA[3])) 
    print('-------'*40)


    
        



# Run the training by running the Train function and
#  specifying the dataExtract and sigmoid function (Sigmoid or Tan)
Train(dataExtract) #uses default value with sigmoid activation

##*****you can uncomment this to get the tangent function*******
# remeber to also change the alpha on Line 254
#also alter epoch in line 269
#Train(dataExtract,Activation='Tan',NoOfNeurons=99)


###To try out test,  remember alter epoch on Lunke 269
### just uncomment and run
#Train(dataExtract,'Tan',99,True)



# References:
# The following tutorials and blogs helped me when i wrote this code
# I did not copy from them but understood them applied made my implementations

# Siraj Raval/ HackingEDU, Building a Neural Network from Scratch in Python , HACKEDU Youtube Channel , https://www.youtube.com/watch?v=262XJe2I2D0
# giant_neural_network, Beginner Intro to Neural Networks 4: First Neural Network in Python , giant_neural_network Youtube Channel , https://www.youtube.com/watch?v=gwitf7ABtK8nk
# Siraj Raval, How to Make a Neural Network (LIVE) , Siraj Raval Youtube Channel , https://www.youtube.com/watch?v=vcZub77WvFA
# giant_neural_network, Beginner Intro to Neural Networks 12: Neural Network in Python from Scratch , giant_neural_network Youtube Channel  ,https://www.youtube.com/watch?v=LSr96IZQknc
# Eric Schles, Writing Neural Networks from Scratch , Next Day Video Youtube Channel , https://www.youtube.com/watch?v=4PjBAO7Uy_Y
# James Loy, How to build your own Neural Network from scratch in Python , Towards Data Science , https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# Daphne Cornelisse, How to build a three-layer neural network from scratch , Free Code Camp , https://medium.freecodecamp.org/building-a-3-layer-neural-network-from-scratch-99239c4af5d3
# Denny Britz, Implementing a Neural Network from Scratch in Python – An Introduction , WildML , http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# JOnathan Weisberg, Building a Neural Network from Scratch: Part 1 , Jonathan Weisberg Blog , https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/
# Sunil Ray, Understanding and coding Neural Networks From Scratch in Python and R , Analytics Vidhya , https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
# Assaad MOAWAD, Neural networks and back-propagation explained in a simple way , Medium , https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e
# Usman Malik, Creating a Neural Network from Scratch in Python , Stack Abuse , https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/
# Usman Malik, Creating a Neural Network from Scratch in Python: Adding Hidden Layers , Stack Abuse , https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-adding-hidden-layers/
# Anni Sap, Build your first neural network in Python , Medium , https://medium.com/@annishared/build-your-first-neural-network-in-python-c80c1afa464
# Samay Shamdasani, Build a Neural Network , Enlight , https://enlight.nyc/projects/neural-network/
# Kawaraha.ca, How to Compute the Derivative of a Sigmoid Function (fully worked example) , Kawahara.ca , http://kawahara.ca/how-to-compute-the-derivative-of-a-sigmoid-function-fully-worked-example/
# Hans Lundmark, First Answer: Derivative of Sigmoid Function , Stack Exchange , https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x/78578
# Eric Schles,  theroetical introduction to neural networks , Github , https://github.com/EricSchles/intro_to_ml/blob/master/lectures/theortical_introduction_to_neural_networks.md
# Trask, A Neural Network in 11 lines of Python (Part 1) , I am Trask , https://iamtrask.github.io/2015/07/12/basic-python-network/
# Trask, A Neural Network in 13 lines of Python (Part 2 - Gradient Descent) , I am Trask , https://iamtrask.github.io/2015/07/27/python-network-part2/
# liac-arff , liac-arff documentation , Github , https://github.com/renatopp/liac-arff
# Ravindra Parmar, Common Loss functions in machine learning , Towards Data Science , https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23
# Daphne Cornelisse,  Kaggle , https://www.kaggle.com/daphnecor/week-1-3-layer-nn?scriptVersionId=2495447
# Author, Title , Youtube , Link
# Author, Title , Youtube , Link
# Author, Title , Youtube , Link







