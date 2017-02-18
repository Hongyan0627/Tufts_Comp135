# import useful packages

import arff
import numpy as np
import matplotlib.pyplot as plt
import random

# set random seed
random.seed(35)

# Read arff files in Python
# artdata_norm.arff has class {A,B,C}
# ionosphere_norm.arff has class {g,b}
# iris_norm.arff has class {Iris-setosa,Iris-versicolor,Iris-virginica}
# seeds_norm.arff has class {1,2,3}

#file_artdata = arff.load(open("./data/artdata.arff", 'rb'))['data']
#file_iris = arff.load(open("./data/iris.arff", 'rb'))['data']
#file_ionosphere = arff.load(open("./data/ionosphere.arff", 'rb'))['data']
#file_seeds = arff.load(open("./data/seeds.arff", 'rb'))['data']



# distance function to calculate the distance between examples i and j
def dist(example_i,example_j):
    """
    Input: example_i and example_j are both arrays for example i and j
    Return: float number for the distance between example i and example j
    """
    return np.linalg.norm(np.array(example_i)-np.array(example_j))
    


# helper function for checking two cluster centers are same
def stopKmeans(center1,center2):
    """
    Input: two cluster centers
    Return: true if there are the same, false otherwise
    """
    return center1 == center2

# function to implement K means clustering
def runKmeans(file_name,k):
    """
    Input: file_name is string for the dataset name, k is an integer for the number of clusters
    Return: k clusters
    """
    
    # load the arff dataset into Python
    data = arff.load(open("./data/"+file_name, 'rb'))['data']
    
    data_features = [item[0:-1] for item in data]
    data_labels = [item[-1] for item in data]
    
    # number of features and number of examples
    num_feature = len(data_features[0])
    num_example = len(data_features)
    
    
    # select k examples as initial cluster centers
    cluster_centers = random.sample(data_features,k)
    
    old_cluster_centers = None
    
    cluster_labels = [None for dummy in range(num_example)]
    
    while ((old_cluster_centers == None) or (not stopKmeans(old_cluster_centers,cluster_centers))):
        old_cluster_centers = cluster_centers
        
        for i in range(num_example):
            temp_index = None
            temp_min = None
            for j in range(k):
                temp_d = dist(data_features[i],cluster_centers[j])
                if temp_min == None or temp_d < temp_min:
                    temp_min = temp_d
                    temp_index = j
            cluster_labels[i] = temp_index
        
        for j in range(k):
            temp_examples = np.array([0 for dummy in range(num_feature)])
            temp_count = 0.0
            
            for i in range(num_example):
                if cluster_labels[i] == j:
                    temp_examples += np.array(data_features[i])
                    temp_count += 1
            if temp_count > 0:
                cluster_centers[j] = list(temp_examples/temp_count)
    return cluster_labels,cluster_centers

# function to compute CS
def computeCS(file_name,k):
    """
    Input: file_name is string for the dataset name, k is an integer for the number of clusters
    Return: float number for Cluster Scatter
    """
    # load the arff dataset into Python
    data = arff.load(open("./data/"+file_name, 'rb'))['data']
    
    data_features = [item[0:-1] for item in data]
    data_labels = [item[-1] for item in data]
    
    num_feature = len(data_features[0])
    num_example = len(data_features)
    
    # get clusters and labels for each example
    cluster_labels,cluster_centers = runKmeans(file_name,k)
    
    result = 0.0
    for i in range(num_example):
        result += (dist(data_features[i],cluster_centers[cluster_labels[i]]))**2
    return result
    
# function to compute NMI

def computeNMI(file_name,k):
    """
    Input: file_name is string for the dataset name, k is an integer for the number of clusters
    Return: float number for NMI
    """
    # load the arff dataset into Python
    data = arff.load(open("./data/"+file_name, 'rb'))['data']
    
    data_features = [item[0:-1] for item in data]
    data_labels = [item[-1] for item in data]
    
    num_feature = len(data_features[0])
    num_example = len(data_features)
    
    # get clusters and labels for each example
    cluster_labels,cluster_centers = runKmeans(file_name,k)
    
    U = list(set(cluster_labels))
    V = list(set(data_labels))
    
    U_V = [[0 for col in range(len(V))] for row in range(len(U))]
    
    for i in range(num_example):
        row_index = U.index(cluster_labels[i])
        col_index = V.index(data_labels[i])
        U_V[row_index][col_index] += 1
    
    a_values = [sum(item) for item in U_V]
    b_values = []
    for j in range(len(V)):
        temp_sum = 0.0
        for i in range(len(U)):
            temp_sum += U_V[i][j]
        b_values.append(temp_sum)
    
    N = num_example + 0.0
    
    H_U =  0.0
    for i in range(len(U)):
        H_U += -((a_values[i])/N) * np.log(a_values[i]/N)
    
    H_V = 0.0
    for j in range(len(V)):
        H_V += -((b_values[j])/N) * np.log(b_values[j]/N)
        
    I_U_V = 0.0
    for i in range(len(U)):
        for j in range(len(V)):
            if (U_V[i][j] > 0):
                I_U_V += (U_V[i][j]/N) * np.log((U_V[i][j]/N)/((a_values[i]*b_values[j])/(N**2)))
            else:
                pass
    return 2 * I_U_V/(H_U + H_V)


def question3_2():
    file_name_list = ["artdata.arff","iris.arff","ionosphere.arff","seeds.arff"]
    k_values = [3,3,2,3]
    
    x = range(1,11,1)
    
    cs_results = [[] for i in range(len(k_values))]
    nmi_results = [[] for i in range(len(k_values))]
    
    for repeat_time in range(10):
        for i in range(len(file_name_list)):
            
            file_name = file_name_list[i]
            k = k_values[i]
            
            temp_cs = computeCS(file_name,k)
            
            temp_nmi = computeNMI(file_name,k)
            
            cs_results[i].append(temp_cs)
            nmi_results[i].append(temp_nmi)
    
    
    for i in range(4):
        plt.plot(x,cs_results[i],'ro',markersize=10)
        plt.xlabel('Different random initializations')
        plt.ylabel('Cluster Scatter')
        plt.title("Cluster Scatter for " + file_name_list[i])
        plt.grid(True)
        plt.savefig("figure"+str(i+1)+".png")
        plt.close()
    
        plt.plot(x,nmi_results[i],'gs',markersize=10)
        plt.xlabel('Different random initializations')
        plt.ylabel('NMI')
        plt.title("NMI for "+file_name_list[i])
        plt.grid(True)
        plt.savefig("figure"+str(i+5)+".png")
        plt.close()
    


def question3_3():
    file_name_list = ["artdata.arff","iris.arff","ionosphere.arff","seeds.arff"]
    k_values = range(1,16,1)
    
    count = 9
    
    for file_name in file_name_list:
        result = []
        for k in k_values:
            temp_min_cs = None
            for j in range(10):
                temp_cs = computeCS(file_name,k)
                if temp_min_cs == None or temp_cs < temp_min_cs:
                    temp_min_cs = temp_cs
            
            result.append(temp_min_cs)
            
        plt.plot(k_values,result,'gs',markersize=10)
        plt.xlabel('k values')
        plt.ylabel('Cluster Scatter')
        plt.title("CS for "+file_name)
        plt.grid(True)
        plt.savefig("figure"+str(count)+".png")
        plt.close()
        count += 1
    


# Following lines for Question 3.2 and Question 3.3

question3_2()

question3_3()
    
    
    
    
    
        




