"""
Put this file into the same directory as the datasets "ibmmac" and "sport"
Call experiment1(dataset_name,figure_name,figure_name2) or experiment2(dataset_name,figure_name) to get the figures. 
"""

# import the stem module to stem the words
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# import some useful modules
import matplotlib.pyplot as plt
import math
import random
import numpy as np


def cal_accuracy(train_file,train_label,test_file,test_label,m,folder_name):
    """
    Input: train_file is the dataset for training
           train_label is the labels for training
           test_file is the dataset for testing
           test_label is the labels for testing
           m is the parameter for smoothing
           folder_name is the name for dataset
    return: accu is a float number, the accuracy for naive bayes model on the test dataset
    """
    
    accu = 0.0
    
    # dic_train_pos is for the counts of words in class yes for training set
    # dic_train_neg is for the counts of words in class no for training set
    dic_train_pos = {}
    dic_train_neg = {}
        
    # num_pos is the number of positive examples in training dataset
    # num_neg is the number of negtive example in training dataset
    num_pos = 0.0
    num_neg = 0.0

    # total_count_pos is the total number of words in class yes
    # total_count_neg is the total number of words in class no
    total_count_pos = 0.0
    total_count_neg = 0.0

    for i in range(len(train_file)):
        if train_label[i] == 1:
            num_pos += 1
        else:
            num_neg += 1
    
        file_name = folder_name + train_file[i] +'.clean'
        file_open = open(file_name)
    
        for row in file_open:
            word_list = row.strip().split(" ")
            if len(word_list) > 0:
                for item in word_list:
                    if len(item) > 0:
                        # stem each word
                        try:
                            temp_word = stemmer.stem(item)
                        except:
                            temp_word = item
                            
                        if train_label[i] == 1:
                            dic_train_pos[temp_word] = dic_train_pos.get(temp_word,0) + 1
                            total_count_pos += 1
                        else:
                            dic_train_neg[temp_word] = dic_train_neg.get(temp_word,0) + 1
                            total_count_neg += 1
        file_open.close()

    # V_pos is the number of different words for class yes
    # V_neg is the number of different words for class no

    V_pos = len(dic_train_pos.keys())
    V_neg = len(dic_train_neg.keys())

    # prob_pos is the probability for positive examples in training dataset
    # prob_neg is the probability for negtive examples in training dataset

    prob_pos = num_pos / (num_pos + num_neg)
    prob_neg = num_neg/ (num_pos + num_neg)

    # zero_flag_pos records whether some word appears with some class but not the other
    zero_flag_pos = False
    zero_flag_neg = False

    # preds is the predictions for testing dataset
    preds = []

    for i in range(len(test_file)):
        file_name = folder_name + test_file[i] +'.clean'
        file_open = open(file_name)
    

        scores_pos = math.log(prob_pos)
        scores_neg = math.log(prob_neg)
    
        for row in file_open:
            word_list = row.strip().split(" ")
            if len(word_list) > 0:
                for item in word_list:
                    if len(item) > 0:
                        
                        try:
                            # stem each word
                            temp_word = stemmer.stem(item)
                        except:
                            temp_word = item
                        # skip the word if it does not appear in the training set
                        if (temp_word not in dic_train_pos) and (temp_word not in dic_train_neg):
                            continue
                        else:  
                            if m > 0:
                                scores_pos += math.log((dic_train_pos.get(temp_word,0) + m)/(total_count_pos + m * V_pos))
                                scores_neg += math.log((dic_train_neg.get(temp_word,0) + m)/(total_count_neg + m * V_neg))
                            else:
                                if dic_train_pos.get(temp_word,0) == 0:
                                    zero_flag_pos = True
                                else:
                                    scores_pos += math.log((dic_train_pos.get(temp_word,0) + m)/(total_count_pos + m * V_pos))
                        
                                if dic_train_neg.get(temp_word,0) == 0:
                                    zero_flag_neg = True    
                                else:
                                    scores_neg += math.log((dic_train_neg.get(temp_word,0) + m)/(total_count_neg + m * V_neg))
                                
        file_open.close()
    
        if m == 0:
            if zero_flag_pos:
                scores_pos = 0
            if zero_flag_neg:
                scores_neg = 0        
                            
        if scores_pos > scores_neg:
            preds.append(1)
        elif scores_pos < scores_neg:
            preds.append(0)
        else:
            preds.append(random.choice([0,1]))
        
    for ii in range(len(preds)):
        if preds[ii] == test_label[ii]:
            accu += 1.0
    accu = accu / len(preds)
        
    return accu


def get_cross_validation(folder_name,k):
    """
    Input: folder_name is a String, for the dataset. K is the number of folds
    Return: k folds and k fold labels
    """
    
    # f is the whole dataset 
    f = open(folder_name + "index.Full")

    # file_list to save all the index for data
    # labels to save corresponding labels for data, 1 for yes and 0 for no

    file_list = []
    labels = []

    for row in f:
        temp_row = row.split('|')
        file_list.append(temp_row[0])
        if temp_row[1].startswith('y'):
            labels.append(1)
        else:
            labels.append(0)
    f.close()

    # take the index for data and the labels, randomly shuffle them
    my_set = zip(file_list,labels)
    random.shuffle(my_set)
    file_list,labels = zip(*my_set)



    pos_list = []
    neg_list = []

    for i in range(len(file_list)):
        if labels[i] > 0:
            pos_list.append(file_list[i]) 
        else:
            neg_list.append(file_list[i])

    # shuffle the positive examples and negative examples
    random.shuffle(pos_list)
    random.shuffle(neg_list)

    pos_folds = [pos_list[int(i * len(pos_list)/k):int((i+1) * len(pos_list)/k)] for i in range(k)]
    neg_folds = [neg_list[int(i * len(neg_list)/k):int((i+1) * len(neg_list)/k)] for i in range(k)]

    random.shuffle(pos_folds)
    random.shuffle(neg_folds)

    folds = []
    labels_folds = []

    for i in range(k):
        temp = zip(pos_folds[i]+neg_folds[i],[1 for dummy in range(len(pos_folds[i]))] + [0 for dummy in range(len(neg_folds[i]))])
        random.shuffle(temp)
        temp2 = zip(*temp)
        folds.append(list(temp2[0]))
        labels_folds.append(list(temp2[1]))

    my_set = zip(folds,labels_folds)
    random.shuffle(my_set)

    # get k folds with labels
    folds,labels_folds = zip(*my_set)
    
    return folds, labels_folds
    


def experiment1(folder_name,figure_name,figure_name2):
    
    # change folder_name to ./ibmmac/ or ./sport/ to set the dataset
    k = 10
    
    folds,labels_folds = get_cross_validation(folder_name,k)
    
    # accuracy_size to record all the accuracies    
    accuracy_size_with_smooth = []
    accuracy_size_without_smooth = []

    for i in range(k):
        train_set = []
        train_lab = []
    
        test_label = labels_folds[i]
        test_file = folds[i]
    
        accu_with_smooth = []
        accu_without_smooth = []
    
        for j in range(k):
            if j != i:
                train_set += folds[j]
                train_lab += labels_folds[j]
    
        # n is the total number of data in the dataset
        N = len(train_set)

        # use different size of training dataset
    
        for s in range(1,11,1):
            train_file = train_set[0:int(s * N/10)]
            train_label = train_lab[0:int(s * N/10)]
            
            temp_accu1 = cal_accuracy(train_file,train_label,test_file,test_label,1.0,folder_name)
            temp_accu2 = cal_accuracy(train_file,train_label,test_file,test_label,0,folder_name)
            accu_with_smooth.append(temp_accu1)
            accu_without_smooth.append(temp_accu2)
        
        accuracy_size_with_smooth.append(accu_with_smooth)
        accuracy_size_without_smooth.append(accu_without_smooth)
    
    accuracy_size_with_smooth = np.array(accuracy_size_with_smooth)
    accuracy_size_without_smooth = np.array(accuracy_size_without_smooth)
    
    set_size = np.arange(0.1,1.1,0.1)
    
    avgs_with_smooth = np.mean(accuracy_size_with_smooth,0)
    stds_with_smooth = np.std(accuracy_size_with_smooth,0) 
    avgs_without_smooth = np.mean(accuracy_size_without_smooth,0)
    stds_without_smooth = np.std(accuracy_size_without_smooth,0)   
    
    plt.plot(set_size,avgs_with_smooth,'ro',markersize=4,label = 'averages with smooth')
    plt.plot(set_size,avgs_without_smooth,'g^',markersize=4,label = 'averages without smooth')
    plt.xlabel('training set size')
    plt.ylabel('The averages for accuracy')
    plt.title("Experiment 1 for dataset "+ folder_name)
    plt.grid(True)
    plt.ylim([0.2,1.4])
    plt.legend(loc = 'upper right')
    plt.savefig(figure_name)
    plt.close()
    #plt.show()
    
    plt.plot(set_size,stds_with_smooth,'ro',markersize=4,label = 'std with smooth')
    plt.plot(set_size,stds_without_smooth,'g^',markersize=4,label = 'std without smooth')
    plt.xlabel('training set size')
    plt.ylabel('The standard deviations for accuracy')
    plt.title("Experiment 1 for dataset "+ folder_name)
    plt.grid(True)
    plt.ylim([0,0.3])
    plt.legend(loc = 'upper right')
    plt.savefig(figure_name2)
    plt.close()

def experiment2(folder_name,figure_name):
    
    k = 10
    
    m = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
    
    folds,labels_folds = get_cross_validation(folder_name,k)

    # accuracy_size to record all the accuracies    
    accuracy = []

    for i in range(k):
        train_set = []
        train_lab = []
    
        test_label = labels_folds[i]
        test_file = folds[i]
    
        accu = []
    
        for j in range(k):
            if j != i:
                train_set += folds[j]
                train_lab += labels_folds[j]
    
        # n is the total number of data in the dataset
        N = len(train_set)

        # use 0.5N of training dataset
    
        train_file = train_set[0:int(0.5 * N)]
        train_label = train_lab[0:int(0.5 * N)]
        
        for mm in m:
            temp_accu = cal_accuracy(train_file,train_label,test_file,test_label,mm,folder_name)
            accu.append(temp_accu)
        
        accuracy.append(accu)
        
    
    accuracy = np.array(accuracy)
    
    avgs = np.mean(accuracy,0)
    #stds = np.std(accuracy,0)  
    
    plt.plot(m,avgs,'ro',markersize=6)
    plt.xlabel('m values')
    plt.ylabel('Accuracy')
    plt.title("Experiment 2 for dataset "+ folder_name)
    plt.grid(True)
    plt.xlim([-3,12])
    #plt.ylim([0.4,1])
    #plt.ylim([0,1.2])
    #plt.legend(loc = 'upper right')
    plt.savefig(figure_name)
    plt.close()
    #plt.show()



folder_name1 = "./ibmmac/"
folder_name2 = "./sport/"

experiment1(folder_name1,"figure1.png","figure5.png")    
#experiment2(folder_name1,"figure2.png")
experiment1(folder_name2,"figure3.png","figure6.png")    
#experiment2(folder_name2,"figure4.png")


