import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import json
import requests
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,mean_absolute_error
response = requests.get("https://api.covid19india.org/data.json")
todos = json.loads(response.text)
todos1=pd.DataFrame(todos['cases_time_series'])
date,totalconfirmed=todos1['date'],todos1['totalconfirmed']
new_date=todos1['date']
year = datetime.date.today().year
i=0
for t in date:
    totalconfirmed[i]=int(totalconfirmed[i])
    new_date[i]=datetime.datetime.strptime(t+str(year)+' 00:00:00', '%d %B %Y %H:%M:%S').strftime("%d/%m/%y")
    i=i+1
df=pd.DataFrame([new_date,totalconfirmed])
print(df)
d=[]
for i in range(len(new_date)):
    if i==0:
        d.append(totalconfirmed[0])
    else:
        d.append((totalconfirmed[i])-(totalconfirmed[i-1]))
cases=np.array(totalconfirmed).reshape(-1,1)
#print(cases.size)
days=np.arange(1,len(totalconfirmed)+1,1).reshape(-1,1)
#print(days.size)
x_train_confirmed,x_test_confirmed,y_train_confirmed,y_test_confirmed=train_test_split(days,totalconfirmed,test_size=0.25,shuffle=False)
poly=PolynomialFeatures(degree=3)
poly_x_train=poly.fit_transform(days)
poly_x_test=poly.fit_transform(x_test_confirmed)
model=LinearRegression()
model.fit(poly_x_train,totalconfirmed)
test_pred=model.predict(poly_x_test)
user=input('enter the date of prediction-')
a = datetime.datetime.strptime(user, "%d/%m/%y")
b = datetime.datetime.strptime(new_date[0], "%d/%m/%y")
d=a-b
d_final=np.array(d.days).reshape(-1,1)
poly_x_final=poly.fit_transform(d_final)
y_final=model.predict(poly_x_final)
print(y_final)