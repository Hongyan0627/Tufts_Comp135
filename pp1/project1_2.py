import arff
import numpy as np
import matplotlib.pyplot as plt
import random
#export CLASSPATH="/Users/hongyanwang/Google Drive/CS135/weka/WEKAINSTALL/weka.jar"


def eucDist(example1, example2):
    """
    input: two same lengh arrays
    output: Euclidean distance between two inputs
    """
    return np.sqrt(np.sum((np.array(example1) - np.array(example2))**2))
    
def weightDist(example1,example2,weight):
    """
    input: example1 and example2 are two data, weight is weights for each feature
    output:weighted distance
    """
    return np.sqrt(np.dot(weight,(np.array(example1) - np.array(example2))**2))
    
def getWeights(train_set,m):
    """
    input: training dataset
    output: an array of weights for each feature in the training dataset
    """
    weights = [0 for i in range(len(train_set[0])-1)]
    
    for i in range(m):
        x = random.choice(train_set)
        label_x = float(x[-1])
        nearest_hit = None
        nearest_miss = None
        d1 = None
        d2 = None
        for i in range(len(train_set)):
            item = train_set[i]
            temp_d = eucDist(x[0:len(x)-1],item[0:len(x)-1])
            if temp_d == 0:
                continue
            else:
                if float(item[-1]) == label_x:
                    if nearest_hit == None or temp_d < d1:
                        d1 = temp_d
                        nearest_hit = i
                else:
                    if nearest_miss == None or temp_d < d2:
                        d2 = temp_d
                        nearest_miss = i
        hit_x = train_set[nearest_hit]
        miss_x = train_set[nearest_miss]
        for j in range(len(weights)):
            weights[j] += (abs(x[j]-miss_x[j]) - abs(x[j]-hit_x[j]))
    for i in range(len(weights)):
        if weights[i] < 0:
            weights[i] = 0
    return weights
    
def kNN(train_set,test_set,k):
    """
    input: train_set is the training set with size (n,m)
            test_set is the test set with size (l,m)
            k is the parameter for how many nearest neighbors
    output: (number of features, test set accuracy)
    """
    num_train_example = len(train_set)
    num_features = len(train_set[0]) - 1
    num_test_example = len(test_set)
    hit = 0.0
    total = 0.0
    for example in test_set:
        total += 1
        distances = []
        train_labels = []
        # calculate all the distances between example in test set and training set
        for item in train_set:
            distances.append(eucDist(example[0:num_features],item[0:num_features]))
            train_labels.append(float(item[-1]))
        # sort the distances and take the k nearest neighbors
        if sum(zip(*sorted(zip(distances,train_labels)))[1][0:k]) < k/2.0:
            pred = 0.0
        else:
            pred = 1.0
        
        if pred == float(example[-1]):
            hit += 1
    accuracy = hit/total
    return accuracy
    

def kNN_WDist(train_set,test_set,k,m):
    """
    input: train_set is the training set with size (n,m)
            test_set is the test set with size (l,m)
            k is the parameter for how many nearest neighbors
    output: (number of features, test set accuracy)
    """
    weights = getWeights(train_set,m)
    num_train_example = len(train_set)
    num_features = len(train_set[0]) - 1
    num_test_example = len(test_set)
    hit = 0.0
    total = 0.0
    for example in test_set:
        total += 1
        distances = []
        train_labels = []
        # calculate all the distances between example in test set and training set
        for item in train_set:
            distances.append(weightDist(example[0:num_features],item[0:num_features],weights))
            train_labels.append(float(item[-1]))
        # sort the distances and take the k nearest neighbors
        if sum(zip(*sorted(zip(distances,train_labels)))[1][0:k]) < k/2.0:
            pred = 0.0
        else:
            pred = 1.0
        
        if pred == float(example[-1]):
            hit += 1
    accuracy = hit/total
    return accuracy
    
      
def kNN_feature_select(train_set,test_set,k,m):
    """
    input: train_set is the training set with size (n,m)
            test_set is the test set with size (l,m)
            k is the parameter for how many nearest neighbors
    output: (number of features, test set accuracy)
    """
    weights = getWeights(train_set,m)
    num_train_example = len(train_set)
    num_features = len(train_set[0]) - 1
    num_test_example = len(test_set)
    hit = 0.0
    total = 0.0
    for example in test_set:
        total += 1
        distances = []
        train_labels = []
        # calculate all the distances between example in test set and training set
        for item in train_set:
            distances.append(eucDist(zip(*sorted(zip(weights,example[0:num_features])))[1][-14:],zip(*sorted(zip(weights,item[0:num_features])))[1][-14:]))
            train_labels.append(float(item[-1]))
        # sort the distances and take the k nearest neighbors
        if sum(zip(*sorted(zip(distances,train_labels)))[1][0:k]) < k/2.0:
            pred = 0.0
        else:
            pred = 1.0
        
        if pred == float(example[-1]):
            hit += 1
    accuracy = hit/total
    return accuracy  
    
    

file_list = [14,24,34,44,54,64,74,84,94]   

###
### For Question 3.3 in project1
###
def Q3_3(flag):
    if flag:
        accuracy_result1 = []
        accuracy_result5 = []
        K1 = 1
        K5 = 5
        for num in file_list:
            train_set = arff.load(open("./" + str(num) + "_train_norm.arff", 'rb'))['data']
            test_set  = arff.load(open("./" + str(num) + "_test_norm.arff", 'rb'))['data']
            temp_accuracy1 = kNN(train_set,test_set,K1)
            temp_accuracy5 = kNN(train_set,test_set,K5)
            accuracy_result1.append(temp_accuracy1)
            accuracy_result5.append(temp_accuracy5)
    
        plt.plot(file_list,accuracy_result1,'bs',markersize=4,label = "K = 1")
        plt.plot(file_list,accuracy_result5,'ro',markersize=4,label = "K = 5")
        plt.plot(file_list,accuracy_result1,'b')
        plt.plot(file_list,accuracy_result5,'r')
        plt.xlabel("Number of features")
        plt.ylabel("Accuracy for test sets")
        plt.legend(["K = 1","K = 5"])
        plt.title("Accuracy for different number of fetures when K = 1 and K = 5")
        plt.grid(True)
        plt.show()
Q3_3(False)
###
### For Question 3.4.1 in project1
###
def Q3_4_1(flag):
    if flag:
        accuracy_result1 = []
        accuracy_result2 = []
        accuracy_result3 = []
        K = 1
        for num in file_list:
            train_set = arff.load(open("./" + str(num) + "_train_norm.arff", 'rb'))['data']
            test_set  = arff.load(open("./" + str(num) + "_test_norm.arff", 'rb'))['data']
            temp_accuracy1 = kNN(train_set,test_set,K)
            temp_accuracy2 = kNN_WDist(train_set,test_set,K,10000)
            temp_accuracy3 = kNN_feature_select(train_set,test_set,K,10000)
            accuracy_result1.append(temp_accuracy1)
            accuracy_result2.append(temp_accuracy2)
            accuracy_result3.append(temp_accuracy3)
            
        plt.plot(file_list,accuracy_result1,'bs',markersize=4,label = "KNN without Relief")
        plt.plot(file_list,accuracy_result2,'ro',markersize=4,label = "KNN with weighted distance")
        plt.plot(file_list,accuracy_result3,'go',markersize=4,label = "KNN with feature selection")
        plt.plot(file_list,accuracy_result1,'b')
        plt.plot(file_list,accuracy_result2,'r')
        plt.plot(file_list,accuracy_result3,'g')
        plt.xlabel("Number of features")
        plt.ylabel("Accuracy for test sets")
        plt.legend(["KNN without Relief","KNN with weighted distance","KNN with feature selection"])
        plt.title("K = "+str(K))
        plt.grid(True)
        plt.show()

Q3_4_1(False)
###
### For Question 3.4.2 in project1
###
def Q3_4_2(flag):
    if flag:
        num = 94
        K = 1
        train_set = arff.load(open("./" + str(num) + "_train_norm.arff", 'rb'))['data']
        test_set  = arff.load(open("./" + str(num) + "_test_norm.arff", 'rb'))['data']
        accuracy_result1 = []
        accuracy_result2 = []
        m_list = range(20,1020,20)
        for m in m_list:
            temp_accuracy1 = kNN_WDist(train_set,test_set,K,m)
            temp_accuracy2 = kNN_feature_select(train_set,test_set,K,m)
            accuracy_result1.append(temp_accuracy1)
            accuracy_result2.append(temp_accuracy2)
            
        plt.plot(m_list,accuracy_result1,'bs',markersize=4,label = "KNN with weighted distance")
        plt.plot(m_list,accuracy_result2,'ro',markersize=4,label = "KNN with feature selection")
        plt.plot(m_list,accuracy_result1,'b')
        plt.plot(m_list,accuracy_result2,'r')
        plt.xlabel("m")
        plt.ylabel("Accuracy for test sets")
        plt.legend(["KNN with weighted distance","KNN with feature selection"])
        plt.title("K = "+str(K)+" and number of features is 94")
        plt.ylim([0.4,1.0])
        plt.grid(True)
        plt.show()

Q3_4_2(True)
            
        
            
            
        
        
        

