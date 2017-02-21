from flask import render_template, request, make_response, session, send_file
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from sklearn import linear_model
from sklearn import externals
 
user = 'postgres'          
host = 'localhost'
dbname = 'cdc'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)


@app.route('/')

@app.route('/cover')
def cover_magic():
    return render_template("coverOG.html")

@app.route('/input1')
def input1():
    return render_template("input1.html")

@app.route('/input2')
def imput2():

  # collect input from first page into named variables, convert to float where needed, scale into correct units where needed...
  age = request.args.get('age')
  age = float(age)
  gender = request.args.get('gender')
  gender = float(gender)
  marital = request.args.get('marital')
  marital = float(marital)

  height1 = request.args.get('height1')
  height2 = request.args.get('height2')
  height1 = float(height1)
  height2 = float(height2)
  height = (height1*30.48 + height2*2.54)
  weight1 = request.args.get('weight')
  weight1 = float(weight1)
  weight = weight1*0.453592

  bmi = weight/((height/100)*(height/100))

  ethnicity = request.args.get('ethnicity'); ethnicity = float(ethnicity)
  ethnicity1 = 0.0; ethnicity2 = 0.0; ethnicity3 = 0.0
  ethnicity4 = 0.0; ethnicity5 = 0.0; ethnicity6 = 0.0
  if ethnicity == 1.0: ethnicity1 = 1.0
  if ethnicity == 2.0: ethnicity2 = 1.0
  if ethnicity == 3.0: ethnicity3 = 1.0
  if ethnicity == 4.0: ethnicity4 = 1.0
  if ethnicity == 5.0: ethnicity5 = 1.0
  if ethnicity == 6.0: ethnicity6 = 1.0

  educ = request.args.get('educ'); educ = float(educ)
  income = request.args.get('income'); income = float(income)
  hhsize = request.args.get('hhsize'); hhsize = float(hhsize)
  if hhsize > 7.0:
    hhsize = 7.0
  
  # assemble first half of input vector as Xvect1, store as session variable
  Xvect1 = [age, gender, marital, bmi, ethnicity1, ethnicity2, ethnicity3, ethnicity4, ethnicity5, ethnicity6, educ, income, hhsize]
  session.clear()
  session['xvect1'] = Xvect1
   
  return render_template("input2.html")

@app.route('/output1')
def output1():
    
  # collect input from input page 2  
  mealsout = request.args.get('mealsout'); mealsout = float(mealsout)
  fastfood = request.args.get('fastfood'); fastfood = float(fastfood)
  vigweek = request.args.get('vigweek'); vigweek = float(vigweek)
  modweek = request.args.get('modweek'); modweek = float(modweek)
  tvcomp = request.args.get('tvcomp'); tvcomp = float(tvcomp)
  smoker = request.args.get('smoker'); smoker = float(smoker)

  smokelen = request.args.get('smokelen'); smokelen = float(smokelen)
  smokeamt = request.args.get('smokeamt'); smokeamt = float(smokeamt)
  smokeamt = smokelen*smokeamt

  alcfreq = request.args.get('alcfreq'); alcfreq = float(alcfreq)
  alcfreq = alcfreq
  alcamt = request.args.get('alcamt'); alcamt = float(alcamt)
  alcpoly = alcfreq*alcamt
  famheart = request.args.get('famheart'); famheart = float(famheart)
  famdiab = request.args.get('famdiab'); famdiab = float(famdiab)

  Xvect1 = session.get('xvect1', None)
  Xvect2 = [mealsout, fastfood, vigweek, modweek, tvcomp, smoker, smokelen, smokeamt, alcpoly, famheart, famdiab]
  #combine two vector halves into the vector that will be fed into the models
  # (xvect2 starts at index 13)
  Xvect = Xvect1 + Xvect2

  # load feature scalers and scale Xvect for each model separately 
  BPscaler = externals.joblib.load('BP24scaler.pkl')
  BP_X = BPscaler.transform(Xvect)
  CHOLscaler = externals.joblib.load('CHOL24scaler.pkl')
  CHOL_X = CHOLscaler.transform(Xvect)
  DIABscaler = externals.joblib.load('DIAB24scaler.pkl')
  DIAB_X = DIABscaler.transform(Xvect)

  # load logistic regression models for each target
  BPmodel = externals.joblib.load('BP24regr.pkl')
  CHOLmodel = externals.joblib.load('CHOL24regr.pkl')
  DIABmodel = externals.joblib.load('DIAB24regr.pkl')

  # return the risk probability for this patient for each target
  BPrisk = BPmodel.predict_proba(BP_X)[0][1]
  CHOLrisk = CHOLmodel.predict_proba(CHOL_X)[0][1]
  DIABrisk = DIABmodel.predict_proba(DIAB_X)[0][1]

  # transform Xvect and return the risk probability for this patient for each target 5 years from now
  Xvect[0] = Xvect[0]+5
  BP_Xfut = BPscaler.transform(Xvect)
  CHOL_Xfut = CHOLscaler.transform(Xvect)
  DIAB_Xfut = DIABscaler.transform(Xvect)
  futBPrisk = BPmodel.predict_proba(BP_Xfut)[0][1]
  futCHOLrisk = CHOLmodel.predict_proba(CHOL_Xfut)[0][1]
  futDIABrisk = DIABmodel.predict_proba(DIAB_Xfut)[0][1]

  # create arrays with the risk value for each target, to determine which factors to extract below
  risks = np.array([BPrisk, CHOLrisk, DIABrisk])
  futrisks = np.array([futBPrisk, futCHOLrisk, futDIABrisk])
  
  BPfactors = BP_X*(BPmodel.coef_[0])
  CHOLfactors = CHOL_X*(CHOLmodel.coef_[0])
  DIABfactors = DIAB_X*(DIABmodel.coef_[0])
  changeablefactors = ['your eating habits', 'your eating habits', 
                        'your exercise habits', 'your exercise habits', 
                          'your tv/computer use', 'your smoking', 'your smoking', 'your smoking', 'your drinking']

  # create variables here to use in the results/visualizations page
  risk1 = ''
  risk2 = ''
  futrisk1 = ''
  futrisk2 = ''
  yourrisk1 = ''
  yourrisk2 = ''
  BPmods = BPfactors[13:22]
  CHOLmods = CHOLfactors[13:22]
  DIABmods = DIABfactors[13:22]

  # if the individual is at risk for anything, determine which of the 'changeablefactors' are contributing the most to that risk value 
  if max(risks) > 0.5:
    if np.argmax(risks) == 0:
      factorsort = np.argpartition(-BPmods, 2)     
    if np.argmax(risks) == 1:
      factorsort = np.argpartition(-CHOLmods, 2)
    if np.argmax(risks) == 2:
      factorsort = np.argpartition(-DIABmods, 2)
    risk1 = changeablefactors[factorsort[0]]
    risk2 = changeablefactors[factorsort[1]]

  # if the individual was healthy now, but will be at risk in 5 years, determine risk factors from those results
  elif max(futrisks) > 0.5:
    if np.argmax(risks) == 0:
      factorsort = np.argpartition(-BPmods, 2)
    if np.argmax(risks) == 1:
      factorsort = np.argpartition(-CHOLmods, 2)
    if np.argmax(risks) == 2:
      factorsort = np.argpartition(-DIABmods, 2)
    futrisk1 = changeablefactors[factorsort[0]]
    futrisk2 = changeablefactors[factorsort[1]]

  # based on which are the risk factors, store the individual's data from the original (non-scaled) input vector, for plotting their 
  # value vs the population
  if factorsort[0] <= 1:
    yourrisk1 = Xvect[13]
  if factorsort[0] == 2 or factorsort[0] == 3 :
    yourrisk1 = Xvect[15]
  if factorsort[0] == 4:
    yourrisk1 = Xvect[17]
  if factorsort[0] == 5 or factorsort[0] == 6 or factorsort[0] == 7:
    yourrisk1 = Xvect[20]
  if factorsort[0] == 8:
    yourrisk1 = Xvect[21]

  if factorsort[1] <= 1:
    yourrisk2 = Xvect[13]
  if factorsort[1] == 2 or factorsort[0] == 3 :
    yourrisk2 = Xvect[15]
  if factorsort[1] == 4:
    yourrisk2 = Xvect[17]
  if factorsort[1] == 5 or factorsort[0] == 6 or factorsort[0] == 7:
    yourrisk2 = Xvect[20]
  if factorsort[1] == 8:
    yourrisk2 = Xvect[21]

  #
  # HERE IS WHERE WE SET THRESHOLDS THAT DETERMINE WHAT IS CONSIDERED RISK
  # Control flow is set such that moderate risk levels will only return a single warning 
  # (ranked in order of seriousness (diabetes > cholesterol > BP)
  # IF the risk is above 0.75 for multiple targets, then the warning returned will include both.

  myoutput = ''
  riskparams = ''
  future = ''
  if BPrisk >= 0.55:
    myoutput = 'You may be at moderate risk for high blood pressure'
    riskparams = 'BP'
  if CHOLrisk >= 0.55:
    myoutput = 'You may be at moderate risk for high cholesterol'
    riskparams = 'CHOL'
  if DIABrisk >= 0.6:
    myoutput = 'You may be at moderate risk for diabetes'
    riskparams = 'DIAB'

  if BPrisk >= 0.75:
    myoutput = 'You are at high risk for high blood pressure'
    riskparams = 'BP'
  if CHOLrisk >= 0.75:
    myoutput = 'You are at high risk for high cholesterol'
    riskparams = 'CHOL'
  if DIABrisk >= 0.75:
    myoutput = 'You are at high risk for diabetes'
    riskparams = 'DIAB'

  #declare which target are we going to report the risk parameters for
  session['riskparams'] = riskparams
  session['risk1'] = risk1
  session['risk2'] = risk2
  session['futrisk1'] = futrisk1
  session['futrisk2'] = futrisk2
  session['yourrisk1'] = yourrisk1
  session['yourrisk2'] = yourrisk2

  if BPrisk >= 0.75 and CHOLrisk >= 0.75:
    myoutput = 'You are at high risk for both high blood pressure and high cholesterol'
  if BPrisk >= 0.75 and DIABrisk >= 0.75:
    myoutput = 'You are at high risk for both high blood pressure and diabetes'
  if CHOLrisk >= 0.75 and DIABrisk >= 0.75:
    myoutput = 'You are at high risk for both high cholesterol and diabetes'

  # if no risk has been declared, tell them they are in good health
  if myoutput == '':
    myoutput = 'You are in good health!'
  
  # determine risk probabilities for 5 year forecast
  future = 1
  futout = ''
  futriskparams = ''
  if futBPrisk >= 0.55:
    futout = 'you may be at moderate risk for high blood pressure'
    futriskparams = 'BP'
  if CHOLrisk >= 0.55:
    futout = 'you may be at moderate risk for high cholesterol'
    futriskparams = 'CHOL'
  if DIABrisk >= 0.55:
    futout = 'you may be at moderate risk for diabetes'
    futriskparams = 'DIAB'

  if futBPrisk >= 0.65:
    futout = 'you will be at high risk for high blood pressure'
    futriskparams = 'BP'
  if futCHOLrisk >= 0.65:
    futout = 'you will be at high risk for high cholesterol'
    futriskparams = 'CHOL'
  if futDIABrisk >= 0.75:
    futout = 'you will be at high risk for diabetes'
    futriskparams = 'DIAB'

  condition1 = 'high blood pressure'
  yourrisk = BPrisk
  if riskparams == 'BP':
    condition1 = 'high blood pressure'; yourrisk = BPrisk
  elif futriskparams == 'BP':
    condition1 = 'high blood pressure'; yourrisk = futBPrisk
  if riskparams == 'CHOL':
    condition1 = 'high cholesterol'; yourrisk = CHOLrisk
  elif futriskparams == 'CHOL':
    condition1 = 'high cholesterol'; yourrisk = futCHOLrisk
  if riskparams == 'DIAB':
    condition1 = 'diabetes'; yourrisk = DIABrisk
  elif futriskparams == 'DIAB':
    condition1 = 'diabetes'; yourrisk = futDIABrisk

  session['yourrisk'] = yourrisk
  session['condition1'] = condition1

  if BPrisk >= 0.68 and CHOLrisk >= 0.68:
    futout = 'you will be at high risk for both high blood pressure and high cholesterol'
  if BPrisk >= 0.68 and DIABrisk >= 0.68:
    futout = 'you will be at high risk for both high blood pressure and diabetes'
  if CHOLrisk >= 0.68 and DIABrisk >= 0.68:
    futout = 'you will be at high risk for both high choleserol and diabetes'

  # if they still haven't crossed a risk probability threshold, tell them they're fine!
  if futout == '':
    futout = 'you will still be in good health!'

  return render_template("output1.html", myoutput = myoutput, futout=futout, BPrisk=BPrisk, future=future, risk1=risk1, risk2=risk2, 
                futrisk1=futrisk1, futrisk2=futrisk2, session = session)

@app.route("/risk")
def simple():
  import random
  from matplotlib.figure import Figure
  import seaborn as sns
  sns.set(style="darkgrid", palette="muted", color_codes=True)

  Xvect1 = session.get('xvect1', None)
  condition1 = session.get('condition1', None)
  riskparams = session.get('riskparams', None)
  risk1 = session.get('risk1', None)
  risk2 = session.get('risk2', None)
  futrisk1 = session.get('futrisk1', None)
  futrisk2 = session.get('futrisk2', None)
  
  # fetch population data for the probabilities
  query = "SELECT * FROM demo4 WHERE ridageyr='%s'" % (Xvect1[0]-5)
  query_results=pd.read_sql_query(query,con)
  
  toprisk = 'bpproba'
  probas = []
  for i in range(0,query_results.shape[0]):
      probas.append(dict(index1=query_results.iloc[i][toprisk], index2=query_results.iloc[i][toprisk]))

  probaslist = []; ct = 0
  for each in probas:
    if probas[ct]['index2'] > 0.0:
      probaslist.append(probas[ct]['index2']); ct+=1

  x = probaslist
  normalize = np.mean(x)
  ct=0
  for each in x:
    x[ct] = each/(normalize); ct+=1

  # fetch the population data for the risk factors for this individual
  query2 = "SELECT * FROM demo5 WHERE ridageyr='%s'" % (Xvect1[0]-5)
  query_results2=pd.read_sql_query(query2,con)
  
  dotted1 = 0.45
  dotted2 = 0.5
  ylimit1 = 0
  ylimit2 = 0
  bins1 = 0
  bins2 = 0
  if risk1 == 'your eating habits':
    riskfactor1 = 'dbd900'
    risk1title = 'Frequency of meals eaten out'
  if risk1 == 'your exercise habits':
    riskfactor1 = 'padvig2'
    risk1title = 'Monthly Exercise'
  if risk1 == 'your tv/computer use':
    riskfactor1 = 'padtvcomp'
    risk1title = 'Daily tv/computer use'
  if risk1 == 'your smoking':
    riskfactor1 = 'smdamt'
    risk1title = 'Lifetime Cigarette Usage'
    dotted1 = 0.8
    ylimit1 = 1.7
    bins1 = 16.0
  if risk1 == 'your drinking':
    riskfactor1 = 'alcpoly'
    risk1title = 'Alcohol Consumption'

  if risk2 == 'your eating habits':
    riskfactor2 = 'dbd900'
    risk2title = 'Frequency of meals eaten out'
  if risk2 == 'your exercise habits':
    riskfactor2 = 'padvig2'
    risk2title = 'Monthly Exercise'
  if risk2 == 'your tv/computer use':
    riskfactor2 = 'padtvcomp'
    risk2title = 'Daily tv/computer use'
  if risk2 == 'your smoking':
    riskfactor2 = 'smdamt'
    risk2title = 'Lifetime Cigarette Usage'
    dotted2 = 0.8
    ylimit2 = 1.7
    bins2 = 16.0
  if risk2 == 'your drinking':
    riskfactor2 = 'alcpoly'
    risk2title = 'Alcohol Consumption'
  
  riskvals = []
  for i in range(0,query_results2.shape[0]):
      riskvals.append(dict(index1=query_results2.iloc[i][riskfactor1], index2=query_results2.iloc[i][riskfactor2]))

  risk1list = []; ct = 0
  for each in riskvals: 
    if riskvals[ct]['index1'] > -1.0:
      risk1list.append(riskvals[ct]['index1'])
    ct+=1

  x2 = risk1list
  normalize2 = np.mean(x2)
  ct=0
  for each in x2:
    x2[ct] = (each)/(normalize2)
    ct+=1

  risk2list = []; ct = 0
  for each in riskvals:
    if riskvals[ct]['index2'] > -1.0:
      risk2list.append(riskvals[ct]['index2']); 
    ct+=1

  x3 = risk2list
  normalize3 = np.mean(x3)

  ct=0
  for each in x3:
    x3[ct] = (each)/(normalize3)
    ct+=1

  x2 = np.array(x2)
  x2 = x2[(x2<8)]
  x3 = np.array(x3)
  x3 = x3[(x3<8)]

  yourrisk = session.get('yourrisk', None)
  yourrisk1 = session.get('yourrisk1', None)
  yourrisk2 = session.get('yourrisk2', None) 
  yourrisk = yourrisk/normalize
  yourrisk = yourrisk - .1
  yourrisk1 = yourrisk1/normalize2
  yourrisk2 = yourrisk2/normalize3

  sns.set_context("paper")
  plt.figure(figsize=(7, 4))
  c = plt.plot([yourrisk, yourrisk], [0,1], 'k--', linewidth=2, label='=  you')
  plt.ylabel('Density of population', fontsize=12)
  plt.xlabel('Relative risk (1.0 is average at your age)', fontsize=12)
  plt.legend(fontsize=10)
  plt.tight_layout(pad=1.5, w_pad=4, h_pad=1)
  histplot = sns.distplot(x, color = '#ad6969', kde=False, norm_hist=True)
  sns.despine()
  fig = histplot.get_figure()
  fig.patch.set_facecolor('blue')
  fig.patch.set_alpha(0.5)
  fig.set_alpha(0.24)

  i=np.random.randint(99999999)
  fig.savefig((("flaskexample/static/images/img%s.png") % i), format='png')
  plt.close()

  sns.set_context("paper")
  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  c = plt.plot([yourrisk1, yourrisk1], [0,dotted1], 'r--', linewidth=2, label='=  you')
  plt.ylabel('Density of population', fontsize=10)
  plt.xlabel('Relative rate (1.0 is average at your age)', fontsize=10)
  plt.legend(fontsize=10)
  if ylimit1 > 0.0:
    plt.ylim(0,ylimit1)
  plt.title(risk1title, fontsize=11)
  if bins1 > 0:
    histplot = sns.distplot(x2, bins=bins1, color = '#ad6969', kde=False, norm_hist=True)
  else:
    histplot = sns.distplot(x2, color = '#ad6969', kde=False, norm_hist=True)
  sns.despine()
  fig = histplot.get_figure()

  plt.subplot(1, 2, 2)
  c = plt.plot([yourrisk2, yourrisk2], [0,dotted2], 'r--', linewidth=2, label='=  you')
  plt.ylabel('Density of population', fontsize=10)
  plt.xlabel('Relative rate (1.0 is average at your age)', fontsize=10)
  plt.legend(fontsize=10)
  plt.title(risk2title, fontsize=11)
  if ylimit2 > 0.0:
    plt.ylim(0,ylimit2)
  plt.tight_layout(pad=1.5, w_pad=5, h_pad=2)
  if bins2 > 0:
    histplot = sns.distplot(x3, bins=bins2, color = '#ad6969', kde=False, norm_hist=True)
  else:
    histplot = sns.distplot(x3, color = '#ad6969', kde=False, norm_hist=True)
  sns.despine()
  fig = histplot.get_figure()
  fig.patch.set_facecolor('blue')
  fig.patch.set_alpha(0.5)
  fig.set_alpha(0.24)

  j=np.random.randint(99999999)
  fig.savefig((("flaskexample/static/images/img%s.png") % j), format='png')
  
  return render_template('matplotlib.html', i=i, j=j, condition1=condition1)    


app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'