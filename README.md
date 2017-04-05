# NHANES_modeling
__predicting health conditions based on lifestyle habits__

This is a exploratory project to see what kind of health conditions we can predict based on self-reported metrics.  One outcome from this project was a web-app to predict hypertension, high cholesterol, and diabetes based on a 20-question survey (found here: www.cardiometrics.life), but the dataset (the ongoing NHANES survey conducted by the CDC (available here: https://wwwn.cdc.gov/Nchs/Nhanes/ContinuousNhanes/Default.aspx)) is ripe for further analysis.  Potential additional targets include osteoporosis, anemia, or arthritis, and further exploration of potential features could improve modelling of the heart-disease-related conditions (building separate models for different age-groups is a promising path forward).

The of data is extracted and cleaned in the __data_extract_clean_recode__ jupyter notebook, and explored/used to construct predictive models in the __predictive_modeling__ jupyter, coded in python with pandas and using scikit-learn for modeling.  For the website, Flask with bootstrap was used, and a postgreSQL database was used to store data for some visualizations on the website.  That code is contained in the folder CARDIOMETRICS.

Feel free to use this as a starting point for further analysis!

