```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
data=pd.read_csv("C:\\Users\\Admin\\Downloads\\archive (22)\\TravelInsurancePrediction.csv")
```


```python
data.drop("Unnamed: 0", axis=1, inplace=True)
```


```python
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, fmt='.1g', cmap="viridis", cbar=True);
```

    C:\Users\Admin\AppData\Local\Temp\ipykernel_8752\919454464.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      sns.heatmap(data.corr(), annot=True, fmt='.1g', cmap="viridis", cbar=True);
    


    
![png](output_3_1.png)
    



```python

```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Employment Type</th>
      <th>GraduateOrNot</th>
      <th>AnnualIncome</th>
      <th>FamilyMembers</th>
      <th>ChronicDiseases</th>
      <th>FrequentFlyer</th>
      <th>EverTravelledAbroad</th>
      <th>TravelInsurance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31</td>
      <td>Government Sector</td>
      <td>Yes</td>
      <td>400000</td>
      <td>6</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31</td>
      <td>Private Sector/Self Employed</td>
      <td>Yes</td>
      <td>1250000</td>
      <td>7</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>Private Sector/Self Employed</td>
      <td>Yes</td>
      <td>500000</td>
      <td>4</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>Private Sector/Self Employed</td>
      <td>Yes</td>
      <td>700000</td>
      <td>3</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private Sector/Self Employed</td>
      <td>Yes</td>
      <td>700000</td>
      <td>8</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>AnnualIncome</th>
      <th>FamilyMembers</th>
      <th>ChronicDiseases</th>
      <th>TravelInsurance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1987.000000</td>
      <td>1.987000e+03</td>
      <td>1987.000000</td>
      <td>1987.000000</td>
      <td>1987.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>29.650226</td>
      <td>9.327630e+05</td>
      <td>4.752894</td>
      <td>0.277806</td>
      <td>0.357323</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.913308</td>
      <td>3.768557e+05</td>
      <td>1.609650</td>
      <td>0.448030</td>
      <td>0.479332</td>
    </tr>
    <tr>
      <th>min</th>
      <td>25.000000</td>
      <td>3.000000e+05</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>6.000000e+05</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>29.000000</td>
      <td>9.000000e+05</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>32.000000</td>
      <td>1.250000e+06</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>35.000000</td>
      <td>1.800000e+06</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.isnull().sum()
```




    Age                    0
    Employment Type        0
    GraduateOrNot          0
    AnnualIncome           0
    FamilyMembers          0
    ChronicDiseases        0
    FrequentFlyer          0
    EverTravelledAbroad    0
    TravelInsurance        0
    dtype: int64




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1987 entries, 0 to 1986
    Data columns (total 9 columns):
     #   Column               Non-Null Count  Dtype 
    ---  ------               --------------  ----- 
     0   Age                  1987 non-null   int64 
     1   Employment Type      1987 non-null   object
     2   GraduateOrNot        1987 non-null   object
     3   AnnualIncome         1987 non-null   int64 
     4   FamilyMembers        1987 non-null   int64 
     5   ChronicDiseases      1987 non-null   int64 
     6   FrequentFlyer        1987 non-null   object
     7   EverTravelledAbroad  1987 non-null   object
     8   TravelInsurance      1987 non-null   int64 
    dtypes: int64(5), object(4)
    memory usage: 139.8+ KB
    


```python

```


```python
data["FrequentFlyer"]=data["FrequentFlyer"].map({"Yes":1,"No":0})
data["EverTravelledAbroad"]=data["EverTravelledAbroad"].map({"Yes":1,"No":0})
data["GraduateOrNot"]=data["GraduateOrNot"].map({"No":0, "Yes":1})
data["Employment Type"]=data["Employment Type"].map({"Government Sector":0,"Private Sector/Self Employed":1})
```


```python

y = data["TravelInsurance"] 
x = data[["Age","GraduateOrNot","Employment Type", "AnnualIncome", "FamilyMembers", "FrequentFlyer", "EverTravelledAbroad"]]  # Features

```


```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=42)
```


```python
from sklearn.tree import DecisionTreeClassifier
trr=DecisionTreeClassifier()
trr.fit(x_train,y_train)
pred=trr.predict(x_test)
from sklearn.metrics import accuracy_score
DecisionTreeScore=accuracy_score(y_test,pred)
```


```python
from sklearn.linear_model import LogisticRegression
logreg_model=LogisticRegression()
logreg_model.fit(x_train,y_train)
logpred=logreg_model.predict(x_test)
logisticaccscore=accuracy_score(y_test,logpred)
```


```python
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
rfpred=rf.predict(x_test)
randomforestaccscore=accuracy_score(y_test,rfpred)
```


```python
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn.fit(x_train,y_train)
knpred=kn.predict(x_test)
knnaccscore=accuracy_score(y_test,knpred)
```


```python
from sklearn.naive_bayes import GaussianNB
GNB=GaussianNB()
GNB.fit(x_train,y_train)
gnbpred=GNB.predict(x_test)
naivebayesaccscore=accuracy_score(gnbpred,y_test)
```

    0.8
    


```python
from sklearn.ensemble import BaggingClassifier
bg=BaggingClassifier(trr,n_estimators=10,random_state=42)
bg.fit(x_train,y_train)
bgpred=bg.predict(x_test)
decisiontreebaggingaccscore=accuracy_score(bgpred,y_test)
```


```python
bg1=BaggingClassifier(kn,n_estimators=10,random_state=42)
bg1.fit(x_train,y_train)
bg1pred=bg1.predict(x_test)
knnbaggingaccscore=accuracy_score(bg1pred,y_test)
```

    0.87
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
