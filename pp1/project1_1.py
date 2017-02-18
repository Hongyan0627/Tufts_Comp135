import os
import matplotlib.pyplot as plt
#export CLASSPATH="/Users/hongyanwang/Google Drive/CS135/weka/WEKAINSTALL/weka.jar"
#os.system("java weka.classifiers.trees.J48 -t ./14_train_norm.arff -d ./j48_14.model -o -v")
#os.system("java weka.classifiers.trees.J48 -l ./j48_14.model -T ./14_test_norm.arff -o> out14.txt")
file_list = [14,24,34,44,54,64,74,84,94]
def find_accuracy(filename):
    """
    input: a txt file name
    output: accuracy on test set
    """
    accu = 0.0
    f = open(filename,'r')
    for row in f.readlines():
        if row.startswith("Correctly Classified Instances"):
            temp_list = row.strip().split(" ")
            count = 0
            for item in temp_list:
                if len(item) > 0:
                    count += 1
                    if count == 5:
                        accu =  0.01 * float(item)
                        break
    return accu
accuracy = []

for num in file_list:
    com1 = "java weka.classifiers.trees.J48 -t ./"+str(num)+"_train_norm.arff -d ./j48_"+str(num)+".model -o -v"
    com2 = "java weka.classifiers.trees.J48 -l ./j48_"+str(num)+".model -T ./"+str(num)+"_test_norm.arff -o> out"+str(num)+".txt"
    os.system(com1)
    os.system(com2)
    filename = "out"+str(num)+".txt"
    accuracy.append(find_accuracy(filename))
    


plt.plot(file_list,accuracy,'bs',markersize=4)
plt.plot(file_list,accuracy,'r')
plt.xlabel("Number of features")
plt.ylabel("Accuracy for test sets")
plt.title("Evaluation for J48 algorithm")
plt.grid(True)
plt.show()
        