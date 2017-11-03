import pandas as pd
import seaborn as sns
import matplotlib.pyplot as p

#Template to plot train vs test accuracy

train = pd.read_csv('train_accuracy.csv', sep="\s+", header = None, names = ['y'])
test = pd.read_csv('test_accuracy.csv', sep="\s+", header = None, names = ['y'])
a = pd.read_csv('a.csv', sep=',', header=None, names=['epoch','accuracy'])
print(train.describe())
print(train.head())
print(train['y'])

print(a.describe())
print(a.head())
print(a['epoch'])
print(a['accuracy'])

#sns.lmplot(x='epoch', y='accuracy', data=a, hue='accuracy', legend=True)

