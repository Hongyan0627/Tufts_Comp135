import csv
import sys
f = csv.reader(open("14_train_norm.csv"))
for row in f:
    print row