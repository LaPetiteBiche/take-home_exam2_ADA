import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# a)
# Data
file = 'dat_ex/pokemon.csv'
df = pd.read_csv(file)

# Count occurence of type + order
x = df.Type.unique()
z = df['Type'].value_counts()
count_ordered=[]
for i in x :
    count_ordered.append(z[i])

# Plot
plt.bar(x, count_ordered)
plt.xticks(rotation=45)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(r'res/ex1_a.png')
plt.close()
plt.savefig(sys.stdout.buffer)

# b)

# Create dummies
top = ["WATER","NORMAL", "FLYING"]
df2 = pd.DataFrame()
for i in top:
    temp_df = df[df['Type'] == i]
    df2 = df2.append(temp_df)

# Infos for plot
columns = ["HP", "Attack", "Defense", "Special Attack", "Special Defense", "Speed"]
nb_col = 6
nb_row = 6
count = 1
plt.figure(figsize=[6.4*nb_col, 6.4*nb_row])

# Plot 2 by 2
for i in columns : 
    test1 = df[df['Type'] == "WATER"]
    wat1 = test1[i].values.tolist()
    test2 = df[df['Type'] == "NORMAL"]
    nor1 = test2[i].values.tolist()
    test3 = df[df['Type'] == "FLYING"]
    fly1 = test3[i].values.tolist()
    for ii in columns :
        test11 = df[df['Type'] == "WATER"]
        wat11 = test1[ii].values.tolist()
        test21 = df[df['Type'] == "NORMAL"]
        nor11 = test2[ii].values.tolist()
        test31 = df[df['Type'] == "FLYING"]
        fly11 = test3[ii].values.tolist()
        
        plt.subplot(nb_row,nb_col,count)
        plt.scatter(wat1, wat11,color='g',label='WATER')
        plt.scatter(nor1, nor11,color='b',label='NORMAL')
        plt.scatter(fly1, fly11,color='r',label='FLYING')
        plt.subplot(nb_row,nb_col,count).set(xlabel = i)
        plt.subplot(nb_row,nb_col,count).set(ylabel = ii)
        plt.subplot(nb_row,nb_col,count).legend()
        plt.grid()
        count += 1
        
plt.savefig(r'res/ex1_b.png')
plt.close()
plt.savefig(sys.stdout.buffer)

# c)
# Create dataframe
types = ['WATER','FIRE','GRASS','NORMAL']
df3 = pd.DataFrame()
for i in types:
    temp_df = df[df['Type'] == i]
    df3 = df3.append(temp_df)

# X values
dfml = df3[['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']]
X = dfml.values

# Dummies
for i in types :
    df3[i] = (df3['Type']==i)*1

# Matrix with dummy's types -> flatten does not make sense here
y = df3[df3.columns[-4:]].values

# Matrix does not seem to work so we use a vector instead -> But then we only predict NORMAL type
y = df3[df3.columns[-1:]].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size= 120/len(df3.index),random_state=0) 

# d)

c_values = [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]

# Standardize
scaler = StandardScaler()
scaler.fit(X_train)
X_train_standardized = scaler.transform(X_train)
X_test_standardized = scaler.transform(X_test)
# test model for all c_values
for i in c_values :
    svm = SVC(kernel='linear', C=i, random_state=0)
    svm.fit(X_train,y_train.ravel())
    y_predicted = svm.predict(X_test_standardized)
    print("Accuracy :\n", accuracy_score(y_true=y_test, y_pred=y_predicted))

# From 0.1 and onward, no improvement on Accuracy
# We check the accuracy for the 3 other types too, to see if it changes or not
types2 = ['WATER','FIRE','GRASS']
for i in types2 :
    print ("-" * 20)
    y = df3[i].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size= 120/len(df3.index),random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_standardized = scaler.transform(X_train)
    X_test_standardized = scaler.transform(X_test)
    for i in c_values :
        svm = SVC(kernel='linear', C=i, random_state=0)
        svm.fit(X_train,y_train.ravel())
        y_predicted = svm.predict(X_test_standardized)
        print("Accuracy :\n", accuracy_score(y_true=y_test, y_pred=y_predicted))
    
# For the 3 other categories no change at all with diffent C -> Go with best parameter = 0.1
# e)
# Train again only with C = 0.1
types2 = ['WATER','FIRE','GRASS', 'NORMAL']
for i in types2 :
    print ("-" * 20)
    y = df3[i].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size= 120/len(df3.index),random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_standardized = scaler.transform(X_train)
    X_test_standardized = scaler.transform(X_test)
    svm = SVC(kernel='linear', C= 0.1, random_state=0)
    svm.fit(X_train,y_train.ravel())
    y_predicted = svm.predict(X_test_standardized)
    confusion_matrix2 = confusion_matrix(y_true=y_test, y_pred=y_predicted)
    print("EX1e) confusion: " + i +" '\n'", confusion_matrix2)

# Look for index of support pokemon, find in df3 and save into csv
sup = svm.support_
names = []
for i in sup :
    df4 = df3.iloc[i]
    names.append(df4['Name'])
df_fin = pd.DataFrame(names,columns=['Pokemon Support Name'])
df_fin.to_csv('res/ex1_e.csv')

# Test with categories values instead of dummies and LinearSVC - Idk if better
df3["Type"] = df3["Type"].astype('category')
df3['type_cat'] = df3['Type'].cat.codes
y = df3['type_cat'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size= 120/len(df3.index),random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_standardized = scaler.transform(X_train)
X_test_standardized = scaler.transform(X_test)
lin_clf = LinearSVC(C=0.1, max_iter=10000)
lin_clf.fit(X_train,y_train.ravel())
y_predicted = lin_clf.predict(X_test_standardized)
confusion_matrix2 = confusion_matrix(y_true=y_test, y_pred=y_predicted)
print("EX1e) confusion with categorical variables: '\n'", confusion_matrix2)

