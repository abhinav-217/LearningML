import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv('titanic.csv')
df.head()

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis='columns',inplace=True)
df.head()

inputs = df.drop('Survived',axis='columns')
target = df.Survived

inputs.Sex = inputs.Sex.map({'male':1,"female":2})
inputs.Age[:10]

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

X_train, X_test, Y_Train, Y_Test = train_test_split(inputs,target,test_size=0.2)

model = tree.DecisionTreeClassifier()
model.fit(X_train,Y_Train)

model.score(X_test,Y_Test)
print(model.predict([[1,2,22]]))