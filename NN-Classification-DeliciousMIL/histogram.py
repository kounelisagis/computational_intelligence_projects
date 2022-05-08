from reader import *
import seaborn as sns
import matplotlib.pyplot as plt

train_text = read_data(filename='data/train-data.dat')
# flatten the list of lists of lists of lists
train_text = [item for sublist1 in train_text for sublist2 in sublist1 for item in sublist2]

sns.set(style="whitegrid")
sns.histplot(data=train_text, bins=100)
plt.xlabel('Word id')
plt.show()
