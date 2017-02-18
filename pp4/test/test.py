# import useful packages

import numpy as np

# Read arff file

def readArff(filename):
    """
    Input: filename, string, name of the arff file
    Return: list of lists, each inside list is an example
    """
    
    f = open("./"+filename,"rb")
    
    data_flag = False
    data_result = []
    
    for row in f:
        if data_flag and (len(row) > 0):
            temp_row = row.strip().split(",")
            temp_data = [float(item) for item in temp_row]
            data_result.append(temp_data)
            
        if row.strip().startswith("@DATA"):
            data_flag = True
    return data_result
    
# calculate the kernel

def getKernel(example1, example2,d,s):
    """
    Input: two vector, if d < 0, use RBF kernel, else use polynomial kernel
    return: Kernel product value
    """
    if d < 0:
        return np.exp(-(np.linalg.norm(np.array(example1) - np.array(example2)))**2 /(2.0 * (s**2)))
    else:
        return (np.dot(example1,example2) + 1)**d
        
# helper function to classify for PPwM

def classifyPPwMval(weight,example):
    
    return np.dot(weight,example)

def classifyKPwMval(alphas,labels,examples,test_example,d,s):
    """
    Input: alphas,vector, labels, list, examples, list of lists, test_example, list, d and s are parameters of kernels
    Return: float number
    """
    
    N = len(examples)
    
    temp = []
        
    for k in range(N):
        temp.append(getKernel(examples[k],test_example,d,s))
    
    temp = np.array(temp)
    
    return np.sum(np.multiply(np.multiply(np.array(alphas),np.array(labels)),temp))
    

# Primal Perceptron with Margin

def PPwM(train_data):
    """
    Input: train_data,list of lists
    Return: array for weights
    """
    
    examples = []
    labels = []
    
    I = 50
    
    N = len(train_data)
    
    m = len(train_data[0])
    
    A = 0.0
    
    for i in range(N):
        temp_example = [1] + train_data[i][0:m-1]
        temp_label = train_data[i][-1]
        A += np.linalg.norm(temp_example)
        examples.append(temp_example)
        labels.append(temp_label)
    
    A = A/N
    
    tau = 0.1 * A
    
    weights = [0.0 for dummy in range(m)]
    
    for dummy in range(I):
        
        for i in range(N):
        
            if (labels[i] * classifyPPwMval(weights,examples[i]) < tau):
                for k in range(m):
                    weights[k] += labels[i] * examples[i][k]
    return weights
    

# Dual Perceptron with Margin

#def DPwM(filename):
#    """
#    Input: filename, string, name of the arff file
#    Return: array for alphas
#    """
#    train_data = readArff(filename)
#    examples = []
#    labels = []
#    
#    I = 50
#    
#    N = len(train_data)
#    
#    m = len(train_data[0]) - 1
#    
#    A = 0.0
#    
#    for i in range(N):
#        temp_example = train_data[i][0:m]
#        temp_label = train_data[i][-1]
#        A += np.linalg.norm(temp_example)
#        examples.append(temp_example)
#        labels.append(temp_label)
#    
#    A = A/N
#    
#    tau = 0.1 * A
#    
#    alphas = [0.0 for dummy in range(N)]
#    
#    for dummy in range(I):
#        for i in range(N):
#            if (labels[i] * np.sum(np.multiply(np.multiply(np.array(alphas),np.array(labels)),np.dot(np.array(examples),np.array(examples[i])))) < tau):
#                alphas[i] += 1
#    return alphas

# Kernel Perceptron with Margin


def KPwM(train_data,d,s):
    """
    Input: train_data,list of lists
    Return: array for alphas
    """
    
    examples = []
    labels = []
    
    I = 50
    
    N = len(train_data)
    
    m = len(train_data[0]) - 1
    
    A = 0.0
    
    for i in range(N):
        
        temp_example = train_data[i][0:m]
        temp_label = train_data[i][-1]
        
        A += np.sqrt(getKernel(temp_example,temp_example,d,s))
        
        examples.append(temp_example)
        labels.append(temp_label)
    
    A = A/N
    
    tau = 0.1 * A
    
    alphas = [0.0 for dummy in range(N)]
    
    for dummy in range(I):
        for i in range(N):
            if (labels[i] * classifyKPwMval(alphas,labels,examples,examples[i],d,s)) < tau:
                alphas[i] += 1
    return alphas
 
def testPPwM(test_data,weights):
    num_test_data = len(test_data)
    result = 0.0
    m = len(test_data[0])
    
    for i in range(num_test_data):
        temp_example = [1] + test_data[i][0:(m-1)]  
        temp_label = test_data[i][-1]
        
        if temp_label == np.sign(classifyPPwMval(weights,temp_example)):
            result += 1
    
    return result/num_test_data
    
def testKPwM(examples,labels,test_data,alphas,d,s):
    num_test_data = len(test_data)
    result = 0.0
    m = len(test_data[0])-1
    
    
    
    for i in range(num_test_data):
        temp_example = test_data[i][0:m]  
        temp_label = test_data[i][-1]
        
        if temp_label == np.sign(classifyKPwMval(alphas,labels,examples,temp_example,d,s)):
            result += 1
    
    return result/num_test_data
    

# Primal 1 Nearest Neighbor

def KNNP(examples,labels,test_example):
    """
    Input: train_data,list of lists
            test_example, vector
    return: 1 or -1, the label of the nearest neighbor
    """

    
    N = len(examples)
    
    m = len(examples[0]) + 1

    
    result = None
    
    min_val = None
    
    for i in range(N):
        temp = np.linalg.norm(np.array([1] + examples[i]) - np.array([1] + test_example))
        
        if min_val == None or temp < min_val:
            result = labels[i]
            min_val = temp
    return result
    

# Kernel 1 Nearest Neighbor

def KNNK(examples,labels,test_example,d,s):
    """
    Input: examples,list of lists
            labels, list
            d is for kernel type
            s is for RBF kernel parameter
    return: 1 or -1, the label of the nearest neighbor
    """
    
    
    N = len(examples)
    
    m = len(examples[0])
    
    result = None
    min_val = None
    
    for i in range(N):
        
        temp = np.sqrt(getKernel(examples[i],examples[i],d,s) + getKernel(test_example,test_example,d,s) - 2 * getKernel(examples[i],test_example,d,s))
        
        if min_val == None or temp < min_val:
            result = labels[i]
            min_val = temp
    return result
    
def testKNNP(examples,labels,test_data):
    num_test_data = len(test_data)
    result = 0.0
    
    m = len(test_data[0]) - 1
    
    for i in range(num_test_data):
        temp_example = test_data[i][0:m]
        temp_label = test_data[i][-1]
        
        if temp_label == KNNP(examples,labels,temp_example):
            result += 1
    return result/num_test_data
    
def testKNNK(examples,labels,test_data,d,s):
    num_test_data = len(test_data)
    result = 0.0
    
    m = len(test_data[0]) - 1
    
    for i in range(num_test_data):
        temp_example = test_data[i][0:m]
        temp_label = test_data[i][-1]
        
        if temp_label == KNNK(examples,labels,temp_example,d,s):
            result += 1
    return result/num_test_data
    
    

if __name__ == "__main__":
    
    train_dataset = ["ATrain.arff","BTrain.arff","backTrain.arff","sonarTrain.arff"]
    test_dataset = ["ATest.arff","BTest.arff","backTest.arff","sonarTest.arff"]
    
    d_vals = [1,2,3,4,5,-1,-1,-1]
    s_vals = [1,1,1,1,1,0.1,0.5,1]
    
    
        
    file_name = "additionalTraining.arff"
        
    train_data = readArff(file_name)
    test_data = readArff("additionalTest.arff")
        
    train_examples = []
    train_labels = []
    test_examples = []
    test_labels = []
        
    N = len(train_data)
        
    N2 = len(test_data)
        
    for num in range(N):
        train_examples.append(train_data[num][0:(len(train_data[num])-1)])
        train_labels.append(train_data[num][-1])
        
    for num in range(N2):
        test_examples.append(test_data[num][0:(len(test_data[num])-1)])
        test_labels.append(test_data[num][-1])
        
    result1 = []
    result2 = []
        
    for j in range(-1,8):
        if j < 0:
            result1.append(testKNNP(train_examples,train_labels,test_data))
            result2.append(testPPwM(test_data,PPwM(train_data)))
        else:
            result1.append(testKNNK(train_examples,train_labels,test_data,d_vals[j],s_vals[j]))
            result2.append(testKPwM(train_examples,train_labels,test_data,KPwM(train_data,d_vals[j],s_vals[j]),d_vals[j],s_vals[j]))
                
                
                
    print result1[0],result1[1],result1[2],result1[3],result1[4],result1[5],result1[6],result1[7],result1[8]
    print result2[0],result2[1],result2[2],result2[3],result2[4],result2[5],result2[6],result2[7],result2[8]
        
            
        
    


            
    
    
    
    
    


