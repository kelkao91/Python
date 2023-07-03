#!/usr/bin/env python
# coding: utf-8

# ### I. Data Analysis and Visualization

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('BC_car_crash.csv')


# To learn more about this data set, we first conducted some preliminary analysis by previewing the data using df.head(), and calling the df.info() function for a summary of the columns including any null values. We also used the df.describe() function for more summary statistics and the df.shape() to find out the number of columns and rows.

# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


df.info()


# 

# #1. For our first question, we are going to analyze how the months of the year affect car crash severity. Specifically, we wanted to find out which months have the highest number of casualty crashes (i.e., crashes that resulted in at least one injury or death).

# In[ ]:


#1.a)
casualty_crash_only = df[df['Crash Severity'] == 'CASUALTY CRASH'] 
#created a data frame with only the casualty crashes


# In[ ]:


#1.a)
#Create dataframe group by month with count of crashes
casualty_month = casualty_crash_only.groupby(['Month Of Year'])['Month Of Year'].count()

#Order the dataframe by month 
#Reference: https://stackoverflow.com/questions/40816144/pandas-series-sort-by-month-index

months_order = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL','MAY','JUNE', 'JULY', 'AUGUST','SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
casualty_month.index = pd.CategoricalIndex(casualty_month.index, categories=months_order, ordered=True)
casualty_month = casualty_month.sort_index()
casualty_month


# In[ ]:





# In[ ]:


#1.a)
casualty_month.plot.bar()
plt.xlabel('Month')
plt.ylabel('# of Severe Car Crashes')
plt.title('Number of Severe Car Crashes Per Month')
plt.xticks(rotation=45)  
plt.show()


# The graph above illustrates the number of severe (i.e., casualty) crashes per month. It appears that the months of September, October, November, December and January have the highest counts of severe crashes. This could be due to the weather conditions in the fall and winter. For instance, rain, snow, wind, and ice can create dangerous driving conditions. 

# #1.b) We also wanted to calculate the proportion of casualty crashes per month. 

# In[ ]:


#1.b)
df['Casualty_Crash'] = (df["Crash Severity"] == "CASUALTY CRASH") 

CC_proportion = df.groupby('Month Of Year', as_index = False)['Casualty_Crash'].mean()

months_order1 = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL','MAY','JUNE', 'JULY', 'AUGUST','SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
CC_proportion.index = pd.CategoricalIndex(CC_proportion['Month Of Year'], categories=months_order1, ordered=True)
CC_proportionFinal = CC_proportion.sort_index().reset_index(drop=True)

display(CC_proportionFinal)

y = df.groupby('Month Of Year')['Casualty_Crash'].mean()
x = CC_proportionFinal['Month Of Year']
plt.xticks(rotation=45)  
#reference: https://www.pythonpool.com/matplotlib-xticks/
plt.bar(x,y)
plt.title("Proportion of Casualty Car Crashes Per Month")
plt.xlabel('Month')
plt.ylabel('Proportion')
plt.show()


# In the graph above, we have the proportion of casualty car crashes out of the total car crashes per month. It appears that March, October, November, and December have the highest proportion of casualty crashes. October-December generally experiences more dangerous driving conditions due to weather which may explain why the proportion of severe car crashes is higher. A higher proportion of casualty car crashes in March is an interesting observation given that in 1.a), we found that the number of severe crashes in March is relatively low.

# #1.c) We also wanted to see how many severe crashes there are in different time categories

# In[ ]:


casualty_time = casualty_crash_only.groupby(['Time Category'])['Time Category'].count()
display(casualty_time)

casualty_time.plot.bar()
plt.xlabel('Time')
plt.ylabel('# of Severe Car Crashes')
plt.title('Number of Severe Car Crashes Per Time Category')
plt.xticks(rotation=45)  
plt.show()


# The graph above illustrates that the most frequent time period for severe car crashes is between 3-6pm. This time period is generally considered rush hour with an increased number of vehicles on the road which could be a contributing factor.

# #2. For our second question, we wanted to see how the location of the crash impacts crash severity. Specially, we will be looking at which streets (i.e., Street Full Name column) have the most casualty crashes.

# In[ ]:


#2.
casualty_crash_only['Street Full Name'].value_counts()


# In[ ]:


casualty_crash_only


# In[ ]:


#2.
casualty_crash_only['Street Full Name'].value_counts().idxmax()


# According to the analysis above, it appears that HWY 1 has the highest number of casualty crashes followed by HWY 97, and Lougheed Hwy. This result is not surprising as HWY 1 is the largest highway in BC connecting to the rest of Canada, which means there are likely more vehicles using this road. 

# However, it is worth noting that the number of crashes on HWY 1 is likely even higher as the data set appears to describe the street names inconsistently. For instance, the data set also includes street names like "HIGHWAY 1" or "HW 1" which likely refers to the same highway. The function below shows that there are 32 casualty crashes that were reported as "HIGHWAY 1" instead of "HWY 1". For the purposes of this analysis, we will simply refer to Highway 1 as "HWY 1".

# In[ ]:


#2.
casualty_crash_only[casualty_crash_only['Street Full Name'] == 'HIGHWAY 1'].shape[0]


# #3. Since HWY 1 is such a large stretch of road, we want to find out which section of this highway is the most dangerous. We will analyze the "Road Location Description" columns to see which areas of the highway have the highest number of casualty crashes. 

# In[ ]:


#3.

HWY1_only = casualty_crash_only[casualty_crash_only['Street Full Name'] == 'HWY 1'] 
#created a data frame with only casualty crashes on HWY 1
HWY1_only.info()
#no null values


# In[ ]:


HWY1_only['Road Location Description'].value_counts()


# As illustrated above, many of the entries do not have a specific description as to what part of HWY 1 the crashes occurred but rather just states "HWY 1 " or "UNKNOWN". However, out of the location descriptions we do have, "264 ST & TRANS-CANADA HWY" has the highest count of casualty crashes followed by "BRUNETTE AVE & TRANS-CANADA HWY", and "232 ST & TRANS-CANADA HWY".

# It is interesting to see that the highest number of casualty crashes on HWY 1 occurs in Greater Vancouver as these sections of the highway are located in Langley and Burnaby. One hypothesis is that crashes are more severe in these areas due to the higher amount of traffic or perhaps even the design of the road is dangerous. One real-world application is to take a closer look to see if there are flaws in the designs of these on/off ramps and reconstruct them to reduce the likelihood of severe car crashes.

# In[ ]:


df_1 = _deepnote_execute_sql('SELECT *\nFROM \'BC_car_crash.csv\'', 'SQL_DEEPNOTE_DATAFRAME_SQL', audit_sql_comment='', sql_cache_mode='cache_disabled')
df_1


# #4. Next, we wanted to observe how crash configuration affected crash severity.

# In[ ]:


#4.a) 

df.groupby('Derived Crash Configuration')['Casualty_Crash'].mean()


# The analysis above shows that "REAR END" collisions have the highest proportion of casualty crashes out of all the crash configurations at 0.42.

# In[ ]:


#4.b)

casualty_crash_only['Derived Crash Configuration'].value_counts()


# "REAR END" collisions have also resulted in the highest number of casualty crashes. 

# In[ ]:


#4.c)

casualty_crash_only.groupby('Derived Crash Configuration')['Total Victims'].sum()


# It appears that "REAR END" collisions have also resulted in the highest number of total victims (i.e., people injured or killed in the crashes), at a combined total of 180348. The three analysis we've conducted on crash configurations have led to the conclusion that there is a high chance that the "REAR END" crash configuration has a detrimental impact on casualty crashes; our research shows that it has the highest number and proportion of casualty crashes and highest number of total victims out of all the other crash configurations.

# #5 Next, we would like to find out the involvement within casualty crashes such as intersection crashes and involvement with animals, pedestrians, heavy vehicles, parked vehicles, cyclists...etc.

# In[ ]:


#5
df.groupby(['Crash Severity','Intersection Crash'])['Intersection Crash'].count()


# In[ ]:


casualty_crash_only['Intersection Crash'].value_counts('Yes')


# We can see that Intersection Crashes occurred in more than half of Casualty Crashes, whereas most Property-Damage-Only Crashes did not involve intersection crashes. It can be assumed that crashes happened at intersections are usually more deadly, which matches the Canada government's statistics of "30% of fatalities and 40% of serious injuries occur at intersections". (source: https://www.cacp.ca/index.html?asst_id=2143) 

# In[ ]:


#https://stackoverflow.com/questions/32589829/how-to-get-value-counts-for-multiple-columns-at-once-in-pandas-dataframe
involvement_casualty = casualty_crash_only[['Heavy Veh Flag','Animal Flag','Cyclist Flag','Motorcycle Flag','Parked Vehicle Flag','Parking Lot Flag','Pedestrian Flag']].apply(pd.Series.value_counts)
involvement_casualty = involvement_casualty.transpose()
involvement_casualty


# As shown above, Parking Lot, Heavy Vehicles, Parked Vehicles, and Pedestrians have significantly more involvement within all casualty crashes. Some of the most common types of crashes happening at parking lots can be hitting a parked vehicle when parking, colliding with another car when exiting the parking lot, and two cars backing into one another.  

# In[ ]:


intersection_casualty = casualty_crash_only[casualty_crash_only['Intersection Crash'] == 'Yes']
intersection_involvement = intersection_casualty[['Heavy Veh Flag','Animal Flag','Cyclist Flag','Motorcycle Flag','Pedestrian Flag']].apply(pd.Series.value_counts)
intersection_involvement = intersection_involvement.transpose()
intersection_involvement


# In[ ]:


#Create dataframe group by month with count of crashes
intersection_casualty_month = intersection_casualty.groupby(['Month Of Year'])['Month Of Year'].count()

#Order the dataframe by month 
#Reference: https://stackoverflow.com/questions/40816144/pandas-series-sort-by-month-index

months_order = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL','MAY','JUNE', 'JULY', 'AUGUST','SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
intersection_casualty_month.index = pd.CategoricalIndex(intersection_casualty_month.index, categories=months_order, ordered=True)
intersection_casualty_month = intersection_casualty_month.sort_index()
intersection_casualty_month


# In[ ]:


intersection_casualty_month.plot.bar()
plt.xlabel('Month')
plt.ylabel('# of Casualty Crashes at Intersections')
plt.title('Number of Casualty Crashes at Intersections Per Month')
plt.xticks(rotation=45)  
plt.show()


# ### II. Machine Learning 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina' # display figures in retina quality")
from sklearn.cluster import KMeans


# As we are curious about the most important factors contributing to casualty crashes, we will be using the data set that includes only casualty car crashes. Some columns are dropped for faster and more concise results because almost all variables in the original data set are categorical instead of numerical values. 

# In[ ]:


ml = casualty_crash_only

#dropping these columns because of redundancy or will create too many columns if use encoding 
ml = casualty_crash_only.drop(columns=['Year','Crash Severity','Street Full Name','Region','Municipality Name','Road Location Description'])
ml.head()


# In[ ]:


ml.info()


# As indicated above, most column types are objects that contain categorical variables, which will be difficult to use in machine learning. Therefore, we will be converting categorical variables into binary variables by using 1/0 or dummy variables.

# In[ ]:


#Transform names of Month of Year and Day of Week into numbers
#reference: https://sparkbyexamples.com/pandas/pandas-map-function-explained/
month_convert = {'JANUARY':1, 'FEBRUARY':2, 'MARCH':3, 'APRIL':4,'MAY':5,'JUNE':6, 'JULY':7, 'AUGUST':8,'SEPTEMBER':9, 'OCTOBER':10, 'NOVEMBER':11, 'DECEMBER':12}
ml['Month Of Year'] = ml['Month Of Year'].map(month_convert)
day_convert = {'MONDAY':1, 'TUESDAY':2, 'WEDNESDAY':3, 'THURSDAY':4,'FRIDAY':5,'SATURDAY':6, 'SUNDAY':7}
ml['Day Of Week'] = ml['Day Of Week'].map(day_convert)

#Transform Yes/No into 1 or 0, with 1 representing "Yes"
#reference: #https://stackoverflow.com/questions/29960733/how-to-convert-true-false-values-in-dataframe-as-1-for-true-and-0-for-false

ml[['Intersection Crash']] = (ml[['Intersection Crash']] == 'Yes').astype(int)
ml[['Heavy Veh Flag']] = (ml[['Heavy Veh Flag']] == 'Yes').astype(int)
ml[['Animal Flag']] = (ml[['Animal Flag']] == 'Yes').astype(int)
ml[['Cyclist Flag']] = (ml[['Cyclist Flag']] == 'Yes').astype(int)
ml[['Motorcycle Flag']] = (ml[['Motorcycle Flag']] == 'Yes').astype(int)
ml[['Parked Vehicle Flag']] = (ml[['Parked Vehicle Flag']] == 'Yes').astype(int)
ml[['Parking Lot Flag']] = (ml[['Parking Lot Flag']] == 'Yes').astype(int)
ml[['Pedestrian Flag']] = (ml[['Pedestrian Flag']] == 'Yes').astype(int)

#Use dummy variables to encode categorical variables
#referece: https://stackoverflow.com/questions/37265312/how-to-create-dummies-for-certain-columns-with-pandas-get-dummies

ohe = pd.get_dummies(ml, prefix=['Time', 'Configuration'], columns=['Time Category','Derived Crash Configuration'])
                              
ohe


# In[ ]:


ohe.info()


# In[ ]:


#Month & Total Victims? can't really put all columns i think?
X = ohe.iloc[:,[0,11]].values

RSS = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 337)
    kmeans.fit(X)
    RSS.append(kmeans.inertia_)


# In[ ]:


plt.plot(range(1, 11), RSS, c='orange')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('RSS')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 337)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], c = 'cyan', label = 'Cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 500, c = 'orange', marker = "*")

plt.xlabel('Month')
plt.ylabel('Total Victims')

plt.legend()
plt.show()


# We wanted to find which factors have the most impact on the number of victims for each severe crash.

# In[44]:


#reference: https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Target column: Total Victim
X = ohe.iloc[:,ohe.columns != 'Total Victims']  #independent columns
y = ohe.iloc[:,11]    #target column
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Factors','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score')) 


# In[ ]:


X = ohe.iloc[:, ohe.columns != 'Total Victims'].values
y = ohe.iloc[:, 11].values #target column: total victims

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


pip install --upgrade --force-reinstall scikit-learn


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 5, random_state = 0)
classifier.fit(X_train, y_train)


predicted_y_test = classifier.predict(X_test)
predicted_y_test


error_rate = np.mean(predicted_y_test != y_test)
print(error_rate)


# In[ ]:


predicted_y_train = classifier.predict(X_train)
error_rate = np.mean(predicted_y_train != y_train)
print(error_rate)


# In[ ]:


predicted_y_test = classifier.predict(X_test)
error_rate = np.mean(predicted_y_test != y_test)
print(error_rate)


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=ea6abd78-f08c-4a01-8902-05862cd49159' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
