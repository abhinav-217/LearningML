import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
x = df[['trip_duration', 'num_of_passengers', 'distance_traveled', 'fare', 'tip', 'miscellaneous_fees', 'surge_applied']]
# kya chiz ko maan ke kya calculate karna h upar jo h woh independent variables h 
y = df['total_fare']
# this is the target value which is going to be predicted and it is dependent on the above variables
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.predict([[748.0,2.75,1.0,75.0,24,6.9,0]]))
print(reg.score(X_test,y_test))    