import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline

data = pd.read_csv('mnist_test.csv')

data.head()

a = data.iloc[4,1:].values

a = a.reshape(28,28).astype('uint8')
plt.imshow(a)

df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(df_x,df_y, test_size= 0.2,random_state=4)

y_train.head()

rf = RandomForestClassifier(n_estimators=100)

rf.fit(x_train,y_train)

pred = rf.predict(x_test)

pred


s = y_test.values


count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count + 1
        
        
count
# count is 1900


# total values that the prediction code was run on
len(pred)
# pred was 2000

#accuracy
1900/2000
0.95
