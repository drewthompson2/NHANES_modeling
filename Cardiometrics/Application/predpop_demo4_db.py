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


X1pred24 = pd.DataFrame.from_csv('X1pred24.csv')

new = []
for each in list(X1pred24):
    new.append(each.lower())

X1pred24.columns = new

X1temp = X1pred24.iloc[0:5000, :]

print ('sending to SQL')

X1temp.to_sql('demo4', engine, if_exists='replace')

print ('sent to SQL')

i = 5000
for x in range(11):
    X1temp = X1pred24.iloc[i:i+5000, :]
    X1temp.to_sql('demo4', engine, if_exists='append')
    print ('sent to SQL')
    i += 5000

X1temp = X1pred24.iloc[i:, :]
X1temp.to_sql('demo4', engine, if_exists='append')

print ('fin')