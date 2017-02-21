from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np

dbname = 'cdc'
username = 'postgres'
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))
con = psycopg2.connect(database = dbname, user = username)


X1predict = pd.DataFrame.from_csv('X1predict.csv')

new = []
for each in list(X1predict):
    new.append(each.lower())

X1predict.columns = new

X1temp = X1predict.iloc[0:5000, :]

print ('sending to SQL')

X1temp.to_sql('demo3', engine, if_exists='replace')

print ('sent to SQL')

i = 5000
for x in range(15):
    X1temp = X1predict.iloc[i:i+5000, :]
    X1temp.to_sql('demo3', engine, if_exists='append')
    print ('sent to SQL')
    i += 5000

X1temp = X1predict.iloc[i:, :]
X1temp.to_sql('demo3', engine, if_exists='append')

print ('fin')