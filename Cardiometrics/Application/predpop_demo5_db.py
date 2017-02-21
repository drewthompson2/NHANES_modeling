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


XY24stats = pd.DataFrame.from_csv('XY24stats.csv')

new = []
for each in list(XY24stats):
    new.append(each.lower())

XY24stats.columns = new

X1temp = XY24stats.iloc[0:5000, :]

print ('sending to SQL')

X1temp.to_sql('demo5', engine, if_exists='replace')

print ('sent to SQL')

i = 5000
for x in range(11):
    X1temp = XY24stats.iloc[i:i+5000, :]
    X1temp.to_sql('demo5', engine, if_exists='append')
    print ('sent to SQL')
    i += 5000

X1temp = XY24stats.iloc[i:, :]
X1temp.to_sql('demo5', engine, if_exists='append')

print ('fin')