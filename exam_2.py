import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
file = 'dat_ex/pokemon.csv'
df = pd.read_csv(file)

# a)
# X
dfml = df[['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']]
X = dfml.values
# Model
model = GaussianMixture(2)
model.fit(X)
pred = model.predict(X)
# Store predicted values
df["Class"] = pred

# Count for class 0 + reorder
class_types = []
df2 = df[df["Class"] == 0]
x = df2.Type.unique()
z = df2['Type'].value_counts()
count_ordered=[]
for i in x :
    count_ordered.append(z[i])

# Get max value
max_value = max(count_ordered)
max_index = count_ordered.index(max_value)
class_types.append(x[max_index])

# Count for class 1 + reorder
df2 = df[df["Class"] == 1]
x = df2.Type.unique()
z = df2['Type'].value_counts()
count_ordered=[]
for i in x :
    count_ordered.append(z[i])

# Get max value
max_value = max(count_ordered)
max_index = count_ordered.index(max_value)
class_types.append(x[max_index])

# Answer
print("EX2a) confusion: '\n'", class_types)

# b)
# Same idea as before but with a loop
n = 2
same = 0
dfml = df[['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']]
X = dfml.values
while same == 0 :
    model = GaussianMixture(n)
    model.fit(X)
    pred = model.predict(X)
    df["Class"] = pred

    class_types = []
    
    for i in range(n) :
        df2 = df[df["Class"] == i]
        x = df2.Type.unique()
        z = df2['Type'].value_counts()
        count_ordered=[]
        for i in x :
            count_ordered.append(z[i])

        max_value = max(count_ordered)
        max_index = count_ordered.index(max_value)
        class_types.append(x[max_index])
        
    a_set = set(class_types)
    if len(class_types) != len(a_set) :
        break
    n += 1

print("EX2b) ")
print(class_types)
print("#Gaussians = '\n'", n)

