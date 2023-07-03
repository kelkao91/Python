#!/usr/bin/env python
# coding: utf-8

# ### Part A. Nobel Laureates

# In[14]:


#1. Read the data in your notebook and display the first 8 rows. How many columns and rows are there in this dataset? What are the column names?

import pandas as pd

nobel = pd.read_csv('nobel.csv')
nobel8 = nobel.head(8)

display(nobel8)
display(nobel.shape)
display(nobel.columns)


# First output shows the first 8 rows of the dataset.
# 
# Second output shows the number of columns and rows as a tuple of (rows, columns). In this dataset, there are 18 columns and 989 rows.
# 
# Third output is a list of the columns' names: 'year', 'category', 'prize', 'motivation', 'price_share', 'laureate_id','laureate_type', 'full_name', 'birth_date', 'birth_city', 'birth_country', 'gender', 'organization_name', 'organization_city', 'organization_country', 'death_date', 'death_city', 'death_country'.

# In[15]:


#2. Looking at all winners in this data, which gender and which country is the most commonly represented?

genderCount = nobel['gender'].value_counts()
print(genderCount.head(1))

countryCount = nobel['birth_country'].value_counts()
print(countryCount.head(1))


# First output shows the most commonly represented gender of the winners in the data, which is male, and the number of winners of that gender - 898. 
# 
# Second output shows the most commonly represented country of the winners in the data, which is USA, and the number of winners from that country - 296.

# In[16]:


#3. Calculate the proportions of winners from that country in each decade. Make some plots to visualize your results.

nobel['USALaureate'] = (nobel["birth_country"] == "USA")
nobel["decade"] =(nobel['year']//10)*10

USAWinnerProp = nobel.groupby('decade', as_index = False)['USALaureate'].mean()
display(USAWinnerProp)

import matplotlib.pyplot as plt

y = nobel.groupby('decade')['USALaureate'].mean()
x = USAWinnerProp['decade']
plt.bar(x, y, width=5, color="pink", edgecolor = 'black')
plt.xlabel('Decade')
plt.ylabel('Proportion of USA Winners')
plt.title("Proportion of Winners from USA in Each Decade")
plt.show()


# First output shows the proportions of winners from USA in each decade (i.e., number of USA winners out of the total number of winners in each decade).
# 
# Second output shows the same data in bar chart form. Upon visualizing my results, it's clearer that the proportion of winners from USA reached the highest in the 2000s, at 0.439024.

# In[17]:


#4. Is there any gender imbalance in this data? How significant is that? Calculate the proportion of female laureates in each decade. Visualize your results and discuss.

indivLaureate = nobel[nobel["laureate_type"] == 'Individual']

femaleLaureate = indivLaureate['gender'] == 'female'

totalFemaleLaureateTotal = femaleLaureate.sum()
totalindivLaureate = indivLaureate['full_name'].count()
femaleLaureateProp = totalFemaleLaureateTotal/totalindivLaureate
display(femaleLaureateProp)

indivLaureate['femaleLaureate'] = (indivLaureate['gender'] == 'female')

femaleLaureatePropDecade = indivLaureate.groupby('decade', as_index = False)['femaleLaureate'].mean()
display(femaleLaureatePropDecade)

y = indivLaureate.groupby('decade')['femaleLaureate'].mean()
x = femaleLaureatePropDecade['decade']
plt.bar(x, y, width=5, color="pink", edgecolor='black')
plt.xlabel('Decade')
plt.ylabel('Proportion of Female Winners')
plt.title("Proportion of Female Winners in Each Decade")
plt.show()


# First output shows the proportion of female laureates in the data. A little over 6% of the individual laureates are female, showing that there is great gender imbalance in the data.
# 
# Second output shows the proportion of female laureates in each decade.
# 
# Third output visualizes the same data. The proportion of female laureates reached the lowest in the 1950s at 0 and the highest in the 2020s at 0.194444. The general upwards trend of the proportion of female laureates starting from the 1970s could potentially be explained by the rise in gender equality and more opportunities granted to potential female laureates.

# In[18]:


#5. For the gender imbalance that you found in question 4, is it better or worse within specific prize categories? 

indivLaureateChem = indivLaureate[indivLaureate["category"] == 'Chemistry']
indivLaureateLit = indivLaureate[indivLaureate["category"] == 'Literature']
indivLaureatePeace = indivLaureate[indivLaureate["category"] == 'Peace']
indivLaureatePhy = indivLaureate[indivLaureate["category"] == 'Physics']
indivLaureatePhyMed = indivLaureate[indivLaureate["category"] == 'Physiology or Medicine']
indivLaureateEconSci = indivLaureate[indivLaureate["category"] == 'Economic Sciences']

femalePropChem = indivLaureateChem['femaleLaureate'].mean()
display(femalePropChem)

femalePropLit = indivLaureateLit['femaleLaureate'].mean()
display(femalePropLit)

femalePropPeace = indivLaureatePeace['femaleLaureate'].mean()
display(femalePropPeace)

femalePropPhy = indivLaureatePhy['femaleLaureate'].mean()
display(femalePropPhy)

femalePropPhyMed = indivLaureatePhyMed['femaleLaureate'].mean()
display(femalePropPhyMed)

femalePropEconSci = indivLaureateEconSci['femaleLaureate'].mean()
display(femalePropEconSci)


# Output shows the proportion of female winners in each category in the following order: Chemistry, Literature, Peace, Physics, Physiology or Medicine, and Economic Sciences. The proportion greatly differs within each category, showing that the gender imbalance is worse in Chemistry, Physics, Physiology or Medicine, and Economic Sciences, but a little better in Literature and Peace (when comparing to the overall proportion which is 0.06360792492179354).

# In[19]:


#5. Visualize your results for each category and discuss. Which of them has the largest gender imbalance?

femaleLaureatePropCategory = indivLaureate.groupby('category', as_index = False)['femaleLaureate'].mean()

y = indivLaureate.groupby('category')['femaleLaureate'].mean()
x = femaleLaureatePropCategory['category']
plt.bar(x, y, width=0.5, color="pink", edgecolor='black')
plt.xlabel('Category')
plt.ylabel('Proportion of Female Winners')
plt.title("Proportion of Female Winners in Each Category")
plt.xticks(rotation=45)
plt.show()


# Output compares the proportion of female laureates in each category. Physics has the largest gender imbalance, with the proportion of female laureates sitting at 0.018.

# In[22]:


#5. Which has shown some positive trend over the decades?

plt.plot(femaleLaureatePropDecade['decade'],indivLaureateChem.groupby('decade')['femaleLaureate'].mean())
plt.xlabel('Decade')
plt.ylabel('Proportion of Female Chemistry Winners')
plt.title("Proportion of Female Winners in Chemistry Over the Decades")
plt.show()

plt.plot(femaleLaureatePropDecade['decade'],indivLaureateLit.groupby('decade')['femaleLaureate'].mean())
plt.xlabel('Decade')
plt.ylabel('Proportion of Female Literature Winners')
plt.title("Proportion of Female Winners in Literature Over the Decades")
plt.show()

plt.plot(femaleLaureatePropDecade['decade'],indivLaureatePeace.groupby('decade')['femaleLaureate'].mean())
plt.xlabel('Decade')
plt.ylabel('Proportion of Female Peace Winners')
plt.title("Proportion of Female Winners in Peace Over the Decades")
plt.show()

plt.plot(femaleLaureatePropDecade['decade'],indivLaureatePhy.groupby('decade')['femaleLaureate'].mean())
plt.xlabel('Decade')
plt.ylabel('Proportion of Female Physics Winners')
plt.title("Proportion of Female Winners in Physics Over the Decades")
plt.show()

plt.plot(femaleLaureatePropDecade['decade'],indivLaureatePhyMed.groupby('decade')['femaleLaureate'].mean())
plt.xlabel('Decade')
plt.ylabel('Proportion of Female Physiology or Medicine Winners')
plt.title("Proportion of Female Winners in Physiology or Medicine Over the Decades")
plt.show()

plt.plot(indivLaureateEconSci['decade'].unique(),indivLaureateEconSci.groupby('decade')['femaleLaureate'].mean())
plt.xlabel('Decade')
plt.ylabel('Proportion of Female Economic Sciences Winners')
plt.title("Proportion of Female Winners in Economic Sciences Over the Decades")
plt.show()


# Outputs visualize the proportion of female winners in each category over the decades. There is a positive trend in recent decades in Chemistry, Literature and Physics (and in previous decades, Peace but the proportion has declined in recent decade) while the proportion of female winners in other categories fluctuate greatly over the decades.

# In[ ]:


#6. Are there any people who have won the Nobel Prize more than once? Who are they?

indivLaureate = nobel[nobel["laureate_type"] == 'Individual']
multiNobelLaureates = indivLaureate['full_name'].value_counts()

counter = 0
for nameResults in multiNobelLaureates:
    if nameResults > 1:
        counter += 1

print(multiNobelLaureates.head(counter))


# Left side shows the full names of multi-winning Nobel laureates who are K. Barry Sharpless, Frederick Sanger, John Bardeen, Linus Carl Pauling, and Marie Curie, née Sklodowska. 
# 
# The right side shows the number of times they each won, which is twice for all of them.

# In[ ]:


#7. Who are the oldest and youngest people ever to have won a Nobel Prize? How old were the winners generally when they got the prize? Show the summary statistics, and plot the distribution of the age of winners.

#reference: https://towardsdatascience.com/working-with-datetime-in-pandas-dataframe-663f7af6c587

indivLaureate['birth_date'] = pd.to_datetime(indivLaureate['birth_date'])
indivLaureate['age'] = indivLaureate['year'] - indivLaureate['birth_date'].dt.year

maxAge = indivLaureate['age'].max()
oldestLaureate = indivLaureate[indivLaureate['age'] == maxAge]
display(oldestLaureate[['full_name','age']])

minAge = indivLaureate['age'].min()
youngestLaureate = indivLaureate[indivLaureate['age'] == minAge]
display(youngestLaureate[['full_name','age']])

meanAge = indivLaureate['age'].mean()
display(meanAge)

summaryStat = indivLaureate['age'].describe()
print(summaryStat)

import numpy as np

x = np.array(indivLaureate['age'])

plt.hist(x, bins = 100, color = 'pink', edgecolor = 'black')
plt.title('Distribution of Age of Winners')
plt.xlabel('Age')
plt.ylabel('# of Winners of that Age')
plt.show()


# First output shows the full name and age of the oldest person to win, who is John B. Goodenough at age 97.
# 
# Second output shows the full name and age of the youngest person to win, who is Malala Yousafzai at age 17. 
# 
# Third output shows the mean of the age of the winners when they got the prize, which is around 60. 
# 
# Fourth output shows the summary statistics of the age of winners. 
# 
# Final output plots the age distribution of winners, the x-axis referring to the age and the y-axis referring to the number of winners who were of that age when they won.

# In[ ]:


#8. For your results in question 7, does the average age of winners differ across each category?

indivLaureateChem = indivLaureate[indivLaureate["category"] == 'Chemistry']
indivLaureateLit = indivLaureate[indivLaureate["category"] == 'Literature']
indivLaureatePeace = indivLaureate[indivLaureate["category"] == 'Peace']
indivLaureatePhy = indivLaureate[indivLaureate["category"] == 'Physics']
indivLaureatePhyMed = indivLaureate[indivLaureate["category"] == 'Physiology or Medicine']
indivLaureateEconSci = indivLaureate[indivLaureate["category"] == 'Economic Sciences']

meanAgeChem = indivLaureateChem['age'].mean()
display(meanAgeChem)
meanAgeLit = indivLaureateLit['age'].mean()
display(meanAgeLit)
meanAgePeace = indivLaureatePeace['age'].mean()
display(meanAgePeace)
meanAgePhy = indivLaureatePhy['age'].mean()
display(meanAgePhy)
meanAgePhyMed = indivLaureatePhyMed['age'].mean()
display(meanAgePhyMed)
meanAgeEconSci = indivLaureateEconSci['age'].mean()
display(meanAgeEconSci)


# The average age of winners does differ across each category; the outputs show the different average ages of winners for Chemistry, Literature, Peace, Physics, Physiology or Medicine, and Economic Sciences respectively. The average ages of winners for Chemistry, Physics, and Physiology or Medicine are lower than the overall average age, while for Literature, Peace, and Economic Sciences, the average ages of winners are higher.
# 
# Economic Sciences has the largest average age of winners and Physics has the lowest.

# In[ ]:


#8. For each category, show the summary statistics and plot the distribution of the age of winners. CHEMISTRY:

summaryStatChem = indivLaureateChem['age'].describe()
print(summaryStatChem)

x = np.array(indivLaureateChem['age'])

plt.hist(x, bins = 100, color = 'green', edgecolor = 'black')
plt.title('Distribution of Age of Chemistry Winners')
plt.xlabel('Age')
plt.ylabel('# of Winners of that Age')
plt.show()


# Outputs show the summary statistics and plot the distribution of the age of winners for Chemistry.

# In[ ]:


#8. LITERATURE:

summaryStatLit = indivLaureateLit['age'].describe()
print(summaryStatLit)

x = np.array(indivLaureateLit['age'])

plt.hist(x, bins = 100, color = 'yellow', edgecolor = 'black')
plt.title('Distribution of Age of Literature Winners')
plt.xlabel('Age')
plt.ylabel('# of Winners of that Age')
plt.show()


# Outputs show the summary statistics and plot the distribution of the age of winners for Literature.

# In[ ]:


#8.PEACE:

summaryStatPeace = indivLaureatePeace['age'].describe()
print(summaryStatPeace)

x = np.array(indivLaureatePeace['age'])

plt.hist(x, bins = 100, color = 'blue', edgecolor = 'black')
plt.title('Distribution of Age of Peace Winners')
plt.xlabel('Age')
plt.ylabel('# of Winners of that Age')
plt.show()


# Outputs show the summary statistics and plot the distribution of the age of winners for Peace.

# In[ ]:


#8. PHYSICS:

summaryStatPhy = indivLaureatePhy['age'].describe()
print(summaryStatPhy)

x = np.array(indivLaureatePhy['age'])

plt.hist(x, bins = 100, color = 'red', edgecolor = 'black')
plt.title('Distribution of Age of Physics Winners')
plt.xlabel('Age')
plt.ylabel('# of Winners of that Age')
plt.show()


# Outputs show the summary statistics and plot the distribution of the age of winners for Physics.

# In[ ]:


#8. PHYSIOLOGY OR MEDICINE:

summaryStatPhyMed = indivLaureatePhyMed['age'].describe()
print(summaryStatPhyMed)

x = np.array(indivLaureatePhyMed['age'])

plt.hist(x, bins = 100, color = 'pink', edgecolor = 'black')
plt.title('Distribution of Age of Physiology or Medicine Winners')
plt.xlabel('Age')
plt.ylabel('# of Winners of that Age')
plt.show()


# Outputs show the summary statistics and plot the distribution of the age of winners for Physiology or Medicine.

# In[ ]:


#8. ECONOMIC SCIENCES:

summaryStatEconSci = indivLaureateEconSci['age'].describe()
print(summaryStatEconSci)

x = np.array(indivLaureateEconSci['age'])

plt.hist(x, bins = 100, color = 'gray', edgecolor = 'black')
plt.title('Distribution of Age of Economic Sciences Winners')
plt.xlabel('Age')
plt.ylabel('# of Winners of that Age')
plt.show()


# Outputs show the summary statistics and plot the distribution of the age of winners for Economic Sciences.

# In[ ]:


#9 Make some plots to visualize the time trend of the average age of winners in each specific
#category per decade. What do you find?


# In[ ]:


#9 Chemistry - average age of winners in chemistry per decade
import matplotlib.pyplot as plt

nobel_chem = indivLaureate[indivLaureate['category'] == 'Chemistry']

nobel_chem['avg_age_decade'] = nobel_chem.groupby(['decade'])['age'].transform('mean')

plt.plot(nobel_chem['decade'], nobel_chem['avg_age_decade'])
plt.xlabel('Decade')
plt.ylabel('Average age')
plt.title('Chemistry - Average age of winners over time')
plt.show()

#reference: https://stackoverflow.com/questions/33445009/pandas-new-column-from-groupby-averages


# In the graph above, the average age of Nobel winners in the chemistry category generally increases each decade meaning that the average age of chemistry winners has increased over time.

# In[ ]:


# 9 Economic Sciences

nobel_economic_sciences = indivLaureate[indivLaureate['category'] == 'Economic Sciences']
nobel_economic_sciences['avg_age_decade_ES'] = nobel_economic_sciences.groupby(['decade'])['age'].transform('mean')

plt.plot(nobel_economic_sciences['decade'], nobel_economic_sciences['avg_age_decade_ES'])
plt.xlabel('Decade')
plt.ylabel('Average age')
plt.title('Economic Sciences - Average age of winners over time')
plt.show()


# In the graph above, the average age of Nobel winners in the economic sciences category fluctuates with a higher average winning age in the 1960's, a drop in the 1990's, and then an increasing average winning age until 2020.

# In[ ]:


#9 Literature
nobel_literature = indivLaureate[indivLaureate['category'] == 'Literature']
nobel_literature['avg_age_decade_L'] = nobel_literature.groupby(['decade'])['age'].transform('mean')


plt.plot(nobel_literature['decade'], nobel_literature['avg_age_decade_L'])
plt.xlabel('Decade')
plt.ylabel('Average age')
plt.title('Literature - Average age of winners over time')
plt.show()


# In the graph above, while the average age of Nobel winners fluctuates slightly in the literature category, the average age of winners generally increases over the decades meaning that the average age of literature winners has increased over time.

# In[ ]:


#9 Peace

nobel_peace = indivLaureate[indivLaureate['category'] == 'Peace']
nobel_peace['avg_age_decade_P'] = nobel_peace.groupby(['decade'])['age'].transform('mean')


plt.plot(nobel_peace['decade'], nobel_peace['avg_age_decade_P'])
plt.xlabel('Decade')
plt.ylabel('Average age')
plt.title('Peace - Average age of winners over time')
plt.show()


# In the graph above, the average age of Nobel winners in the peace category fluctuates. However, the average age of winners in this category overall declines over time. 

# In[ ]:


#9 Physics

nobel_physics = indivLaureate[indivLaureate['category'] == 'Physics']
nobel_physics['avg_age_decade_phy'] = nobel_physics.groupby(['decade'])['age'].transform('mean')


plt.plot(nobel_physics['decade'], nobel_physics['avg_age_decade_phy'])
plt.xlabel('Decade')
plt.ylabel('Average age')
plt.title('Physics - Average age of winners over time')
plt.show()


# In the graph above, the average age of Nobel winners in the physics category generally increases each decade meaning that the average age of physics winners has increased over time.

# In[ ]:


#9 Physiology or medicine

nobel_med = indivLaureate[indivLaureate['category'] == 'Physiology or Medicine']
nobel_med['avg_age_decade_med'] = nobel_med.groupby(['decade'])['age'].transform('mean')


plt.plot(nobel_med['decade'], nobel_med['avg_age_decade_med'])
plt.xlabel('Decade')
plt.ylabel('Average age')
plt.title('Physiology or Medicine - Average age of winners over time')
plt.show()


# In the graph above, the average age of Nobel winners in the physiology or medicine category generally increases each decade meaning that the average age of winners in this category has increased over time.

# In[ ]:


#10 Repeat question 9 but with lifespan instead of age. 


# In[ ]:


indivLaureate.isna().sum()


# In order to calculate life span, we need to subtract the birth date from the death date. Calling the isna() function, we see there are 301 missing values for the death date which we will need to drop before continuing.

# In[ ]:



#drop missing values from death_date and birth_date

nobel_dropped = indivLaureate.dropna(subset=['death_date', 'birth_date',])


#Adding lifespan column

nobel_dropped['death_date'] = pd.to_datetime(nobel_dropped['death_date'])

nobel_dropped['lifespan'] = nobel_dropped['death_date'].dt.year - nobel_dropped['birth_date'].dt.year


# In[ ]:





# In[ ]:


#10 Chemistry
nobel_chem10 = nobel_dropped[nobel_dropped['category'] == 'Chemistry']

nobel_chem10['avg_ls_decade'] = nobel_chem10.groupby(['decade'])['lifespan'].transform('mean')

plt.plot(nobel_chem10['decade'], nobel_chem10['avg_ls_decade'])
plt.xlabel('Decade')
plt.ylabel('Average lifespan')
plt.title('Chemistry - Average lifespan of winners over time')
plt.show()


# In the graph above, the average lifespan of Nobel winners in the chemistry category generally increases each decade meaning that the average lifespan of winners in this category has increased over time.

# In[ ]:


#10 Economic Sciences

nobel_es10 = nobel_dropped[nobel_dropped['category'] == 'Economic Sciences']

nobel_es10['avg_ls_decade'] = nobel_es10.groupby(['decade'])['lifespan'].transform('mean')

plt.plot(nobel_es10['decade'], nobel_es10['avg_ls_decade'])
plt.xlabel('Decade')
plt.ylabel('Average lifespan')
plt.title('Economic Sciences - Average lifespan of winners over time')
plt.show()


# The graph above demonstrates an interesting trend where the lifespan increases from 1960-1980 and begins to decline into 2010. However, this fluctuation is not that drastic given that the range in lifespan is small.

# In[ ]:


#10 Literature

nobel_lit10 = nobel_dropped[nobel_dropped['category'] == 'Literature']

nobel_lit10['avg_ls_decade'] = nobel_lit10.groupby(['decade'])['lifespan'].transform('mean')

plt.plot(nobel_lit10['decade'], nobel_lit10['avg_ls_decade'])
plt.xlabel('Decade')
plt.ylabel('Average lifespan')
plt.title('Literature - Average lifespan of winners over time')
plt.show()


# In[ ]:


#10 Peace

nobel_peace10 = nobel_dropped[nobel_dropped['category'] == 'Peace']

nobel_peace10['avg_ls_decade'] = nobel_peace10.groupby(['decade'])['lifespan'].transform('mean')

plt.plot(nobel_peace10['decade'], nobel_peace10['avg_ls_decade'])
plt.xlabel('Decade')
plt.ylabel('Average lifespan')
plt.title('Peace - Average lifespan of winners over time')
plt.show()


# The graph demonstrates that the average lifespan of nobel winners in the peace category fluctuate quite a bit over the decades without a clear trend

# In[ ]:


#10 Physics

nobel_phy10 = nobel_dropped[nobel_dropped['category'] == 'Physics']

nobel_phy10['avg_ls_decade'] = nobel_phy10.groupby(['decade'])['lifespan'].transform('mean')

plt.plot(nobel_phy10['decade'], nobel_phy10['avg_ls_decade'])
plt.xlabel('Decade')
plt.ylabel('Average lifespan')
plt.title('Physics- Average lifespan of winners over time')
plt.show()


# The graph above shows that while there are fluctuations, the average lifespan of nobel winners in the physics category tends to increase over time.

# In[ ]:


#10 Physiology or Medicine

nobel_med10 = nobel_dropped[nobel_dropped['category'] == 'Physiology or Medicine']

nobel_med10['avg_ls_decade'] = nobel_med10.groupby(['decade'])['lifespan'].transform('mean')

plt.plot(nobel_med10['decade'], nobel_med10['avg_ls_decade'])
plt.xlabel('Decade')
plt.ylabel('Average lifespan')
plt.title('Medicine - Average lifespan of winners over time')
plt.show()


# The graph above demonstrates that the average lifespan of medicine nobel winners tends to increase over time with the exception of a sharp decrease in the 2000s.

# In[ ]:





# ### Part B. COVID-19

# In[4]:


#11
import pandas as pd
covid = pd.read_csv('covid.csv')
covid.head(8)


# In[ ]:


#11
covid.shape #12554 rows, 5 columns


# covid.shape returns a tuple with the number of rows as the first value and the number of columns as the second value. 

# In[ ]:


#11
covid.columns #column names


# covid.columns returns the column names which are 'Reported_Date', 'HA', 'Sex', 'Age_Group', and 'Classification_Reported'

# In[6]:


#12. Create a new column Month to represent the month of the Reported_Date in the data. 
#Print the first 8 rows of updated dataframe

#reference: https://blog.hubspot.com/website/pandas-split-string

date_split = covid['Reported_Date'].str.split('-', expand = True)
covid[['Year', 'Month', 'Day']] = date_split
covid = covid.drop(['Year', 'Day'], axis = 1)
print(covid.head(8))


# In[7]:


#13 Create a new dictionary which contains the months as keys and the number of cases for corresponding months as values. 

#refereces: https://pandas.pydata.org/docs/reference/api/pandas.Series.to_dict.html

#Show the dictionary.
covidbymonth_dict = covid.groupby(['Month'])['Month'].count().to_dict()
display(covidbymonth_dict)

#What is the largest number of cases in the dictionary?
max(covidbymonth_dict.values())


# #13 Explanation: To count the number of cases for each month, data needs to first be grouped by "Month" to organize the data by month, then count the occurrences of corresponding month within the data for number of cases. to_dict() is used at the end to transform the resulting data type Series into Dictionary. The largest number of cases is 3392. max() is used to find the maximum number, and .values(0 is to refer to values within the dictionary instead of keys.

# In[8]:


#14 Make some plots to visualize the time trend of the number of cases in every month. What do you find?
import matplotlib.pyplot as plt
x = covidbymonth_dict.keys()
y = covidbymonth_dict.values()

#Line Plot
plt.plot(x,y)
plt.xlabel('Month')
plt.ylabel('Number of Cases')
plt.title('Number of Cases per Month')
plt.show()


# #14 Explanation: values for x and y are taken from the dictionary created in #13. For x, .keys() is used to return the month (key) from the dictionary, while values() is used to return number of cases each month (value) from it. Line chart and bar chart are plotted to visualize the growth and decline of cases.

# #14 Analysis: It is evident that cases of covid started to appear around mid to end of February. Although number of cases was declining through May to June, it began to rise significantly from July, and reached its peak in September.

# In[9]:


#15 Is there any gender imbalance in this data? Visualize the time trend of the number of cases for each gender and discuss.

#references: https://www.geeksforgeeks.org/applying-lambda-functions-to-pandas-dataframe/

#group data by each gender and month
covid_M_dict = covid.groupby(['Month'])['Sex'].apply(lambda x: (x=='M').sum()).to_dict()
covid_F_dict = covid.groupby(['Month'])['Sex'].apply(lambda x: (x=='F').sum()).to_dict()
covid_U_dict = covid.groupby(['Month'])['Sex'].apply(lambda x: (x=='U').sum()).to_dict()

#assign values fo plotting
x = covid_M_dict.keys()
m = covid_M_dict.values()
f = covid_F_dict.values()
u = covid_U_dict.values()

#Line Plot
plt.plot(x,m, label = 'Male', color = 'green')
plt.plot(x,f, label = 'Female', color = 'orange')
plt.plot(x,u, label = 'Undefined', color = 'purple')
plt.title('Number of Cases for Each Gender per Month')
plt.legend()
plt.show()


# #15 Explanation: For simpleness and effectiveness to plot the graphics for each gender, data is first filtered by gender of M, F, U to create data sets for each gender. Then, new variables are defined to represent values carries by each gender's data set. Line plot and bar graphs are used to visualize the difference among number of cases each month from each gender.

# #15 Analysis: Through visualization as stacked bars, it is noticeable that female takes up the largest portion of cases each month compared to male and undefined.

# In[ ]:


#16 Create new dictionaries which contain the months as keys and the number of cases for corresponding months per each gender as values. 

covid_M_dict = covid.groupby(['Month'])['Sex'].apply(lambda x: (x=='M').sum()).to_dict()
covid_F_dict = covid.groupby(['Month'])['Sex'].apply(lambda x: (x=='F').sum()).to_dict()
covid_U_dict = covid.groupby(['Month'])['Sex'].apply(lambda x: (x=='U').sum()).to_dict()

#Show the dictionaries
display(covid_M_dict)
display(covid_F_dict)
display(covid_U_dict)

# Find the month with the smallest number of female cases.
#reference: https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
print(list(covid_F_dict.keys())[list(covid_F_dict.values()).index(min(covid_F_dict.values()))])


# #16 Explanation: list() is used to return the corresponding key based on the minimum value from the dictionary values.

# In[10]:


#17 Is there any imbalance among different regions in this data? Make some plots to visualize the difference among the regions in terms of reported cases in every month, and discuss your results.

#references: https://www.geeksforgeeks.org/applying-lambda-functions-to-pandas-dataframe/

#Number of cases each month in each region 
out = covid.groupby(['Month'])['HA'].apply(lambda x: (x=='Out of Canada').sum())
coastal = covid.groupby(['Month'])['HA'].apply(lambda x: (x=='Vancouver Coastal').sum())
interior = covid.groupby(['Month'])['HA'].apply(lambda x: (x=='Interior').sum())
fraser = covid.groupby(['Month'])['HA'].apply(lambda x: (x=='Fraser').sum())
northern = covid.groupby(['Month'])['HA'].apply(lambda x: (x=='Northern').sum())
island = covid.groupby(['Month'])['HA'].apply(lambda x: (x=='Vancouver Island').sum())

#Assign variables
o = out.to_dict()
c = coastal.to_dict()
i = interior.to_dict()
f = fraser.to_dict()
n = northern.to_dict()
i = island.to_dict()
x = covid_M_dict.keys()

#Line Plot
plt.plot(x,o.values(), label = 'Out of Canada', color = 'green')
plt.plot(x,c.values(), label = 'Vancouver Coastal', color = 'orange')
plt.plot(x,i.values(), label = 'Interior', color = 'purple')
plt.plot(x,f.values(), label = 'Fraser', color = 'blue')
plt.plot(x,n.values(), label = 'Northern', color = 'red')
plt.plot(x,i.values(), label = 'Vancouver Island', color = 'yellow')
plt.title('Number of Cases from each region per Month')
plt.legend()
plt.show()


# #17 Explanation: New data series are first to summarize number of cases each month in each region. Values are returned from the series during plotting. Line plot and bar graphs are used to visualize the difference among number of cases each month from each region.

# #17 Analysis: In the beginning, Vancouver Coastal has the steepest increase in cases from February to March. However, number of cases in Fraser started to surpass from mid-March, while cases in Vancouver Coastal declined. After mid-March, Fraser remained as the region with the largest number of cases, and Vancouver Coastal has the second largest. 

# In[ ]:



#18 Calculate the cumulative reported cases in every month for each region. Print the first 8 rows. 

#references: https://www.statology.org/pandas-cumulative-sum-by-group/

region = covid.groupby(['Month','HA'])['HA'].count().reset_index(name="Number of Cases")
region['Cumulative Cases'] = region.groupby(['HA'])['Number of Cases'].cumsum()
region=region.drop(['Number of Cases'], axis=1)
region.head(8)


# #18-1 Explanation: A new data frame "region" is created to return number of cases each month in each region for easiness to do cumulative aggregation. A new column "Cumulative Cases" is created to indicate the number of cases from that the region has accumulated until that month, thus the use of group by 'HA' then cumsum. 

# In[ ]:


#18 Visualize the difference among the regions in terms of cumulative reported cases in every month, and discuss your results.

#references: https://www.geeksforgeeks.org/applying-lambda-functions-to-pandas-dataframe/

#Assgin variables for number of cumulative cases in each region 
o_cum = out.cumsum().to_dict()
c_cum = coastal.cumsum().to_dict()
i_cum = interior.cumsum().to_dict()
f_cum = fraser.cumsum().to_dict()
n_cum = northern.cumsum().to_dict()
i_cum = island.cumsum().to_dict()
x = covid_M_dict.keys()

#Line Plot
plt.plot(x,o_cum.values(), label = 'Out of Canada', color = 'green')
plt.plot(x,c_cum.values(), label = 'Vancouver Coastal', color = 'orange')
plt.plot(x,i_cum.values(), label = 'Interior', color = 'purple')
plt.plot(x,f_cum.values(), label = 'Fraser', color = 'blue')
plt.plot(x,n_cum.values(), label = 'Northern', color = 'red')
plt.plot(x,i_cum.values(), label = 'Vancouver Island', color = 'yellow')
plt.title('Number of Cumulative Cases in each region each Month')
plt.legend()
plt.show()


# #18-2 Explanation: By using the data series created in #17 for number of cases in each region each month, cumsum() is applied to the series to result in number of cumulative cases in every month for each region. Line plot is used to visualize the differences of cumulative cases.

# #18-2 Analysis: Vancouver Coastal had the all-time largest number of cumulative cases in February and March, and Fraser had it from April and onwards. 

# 
# #19. Open question: what else can you find from this data? 

# In[25]:


age_cases = covid.groupby(['Age_Group'])['Age_Group'].count().reset_index(name="Cases")
plt.bar(age_cases['Age_Group'],age_cases['Cases'],color='orange')
plt.xlabel('Age Group')
plt.ylabel('Number of Cases')
plt.title('Total Number of Cases in Each Age Group')
plt.xticks(rotation=45)
plt.show()


# Based on previous graphs, we can tell that the number of cases through out all regions rapidly grew since July and onwards. It may be assumed that cases rose in summer as people went outside more for activities and gatherings, especially when children and students were on school vacations. 

# Also, Fraser and Vancouver Coastal regions had similar slopes for number of increasing cases. However, Fraser had far more cases than Vancouver Coastal, which could be resulted from Fraser's larger population.

# An additional graph is made in this question to further investigate and visualize the total number of  cases in each age group. It is evident that age groups of 20-29 had the highest number of cases, while 30-39 had the second highest and 40-49 being the third. 

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=ea6abd78-f08c-4a01-8902-05862cd49159' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
