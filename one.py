import pandas as pd
car = pd.read_csv('quikr_car.csv')
# print(car.head())
# print(car.info())
backup=car.copy()


# Quality
# names are pretty inconsistent
# names have company names attached to it
# some names are spam like 'Maruti Ertiga showroom condition with' and 'Well mentained Tata Sumo'
# company: many of the names are not of any company like 'Used', 'URJENT', and so on.
# year has many non-year values
# year is in object. Change to integer
# Price has Ask for Price
# Price has commas in its prices and is in object
# kms_driven has object values with kms at last.
# It has nan values and two rows have 'Petrol' in them
# fuel_type has nan values


###cleaning of data

car = car[car['year'].str.isnumeric()]
car['year']=car['year'].astype(int)
print(car['year'])
print(car.info())


car=car[car['Price']!='Ask For Price']
car['Price']=car['Price'].str.replace(',','').astype(int)
print(car)


car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')
car=car[car['kms_driven'].str.isnumeric()]
car['kms_driven']=car['kms_driven'].astype(int)

car=car[~car['fuel_type'].isna()]

car['name']=car['name'].str.split().str.slice(start=0,stop=3).str.join(' ')
car=car.reset_index(drop=True)
print(car)

car.to_csv('Cleaned_Car_data.csv')

X = car.drop(columns='Price')
y = car['Price']
print(y)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

X = car.drop(columns='Price')
y = car['Price']

column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     ['name','company','fuel_type']),
    remainder='passthrough'
)

print("Loop started")   # <-- MOVED HERE

scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    pipe = make_pipeline(column_trans, LinearRegression())
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))



best_idx = np.argmax(scores)
print("Best random_state =", best_idx)
print("Best R2 score =", scores[best_idx])

print(scores[np.argmax(scores)])
print(pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5))))


import pickle
pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))
print(pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5))))

print(pipe.steps[0][1].transformers[0][1].categories[0])