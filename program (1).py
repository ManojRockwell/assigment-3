#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import  exp
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
# Read the dataset
dataframe = pd.read_csv("API_19_DS2_en_csv_v2_5361599.csv",skiprows=4)

# Filter relevant columns and rows
required_c = ["Country Name", "Indicator Name",'1960','1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014',"2015", "2016", "2017", "2018", "2019", "2020"]
required_r = dataframe[dataframe["Indicator Name"] == "Mortality rate, under-5 (per 1,000 live births)"]
dataframe_filtered = required_r[required_c]
dataframe_filtered


# In[2]:


dataframe1 = dataframe
dataframe2 = dataframe1.drop(columns=["Country Code", "Indicator Code"], axis=1)
dataframe2
dataframe2.isnull().sum()
dataframe3 = dataframe2.fillna(0)
dataframe3

dataframe4 = dataframe3
dataframe4.set_index("Country Name", inplace=True)
dataframe4 = dataframe4.loc["United Kingdom"]
dataframe4 = dataframe4.reset_index(level="Country Name")
dataframe4.groupby(["Indicator Name"])
dataframe4 = dataframe4.drop(["Unnamed: 66", '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969',
                    '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979',
                    '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989',
                    '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', 'Country Name'], axis=1)

dataframe4
dataframe4 = dataframe4.set_index('Indicator Name')
dataframe4 = dataframe4.transpose()
dataframe4


# In[ ]:


""" Tools to support clustering: correlation heatmap, normaliser and scale
(cluster centres) back to original scale, check for mismatching entries """


def map_corr(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)

    The function does not have a plt.show() at the end so that the user
    can savethe figure.
    """

    import matplotlib.pyplot as plt  # ensure pyplot imported

    corr = df.corr()
    plt.figure(figsize=(size, size))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='Pastel2')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.colorbar()
    plt.show()


# In[ ]:


dataframe4 = dataframe4.drop(['Urban population (% of total population)', 'Urban population', 'Urban population growth (annual %)', 
                                'Population, total', 'Population growth (annual %)', 'Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)', 
                                'Prevalence of underweight, weight for age (% of children under 5)', 'Community health workers (per 1,000 people)', 
                                'Mortality rate, under-5 (per 1,000 live births)', 'Primary completion rate, total (% of relevant age group)', 
                                'School enrollment, primary and secondary (gross), gender parity index (GPI)', 'Agriculture, forestry, and fishing, value added (% of GDP)', 
                                'CPIA public sector management and institutions cluster average (1=low to 6=high)', 
                                'Ease of doing business rank (1=most business-friendly regulations)', 'Terrestrial and marine protected areas (% of total territorial area)', 
                                'Marine protected areas (% of territorial waters)', 'Terrestrial protected areas (% of total land area)', 
                                'Annual freshwater withdrawals, total (% of internal resources)', 'Annual freshwater withdrawals, total (billion cubic meters)', 
                                'Population in urban agglomerations of more than 1 million (% of total population)', 
                                'Population living in areas where elevation is below 5 meters (% of total population)', 
                                'Urban population living in areas where elevation is below 5 meters (% of total population)', 
                                'Rural population living in areas where elevation is below 5 meters (% of total population)', 
                                'Droughts, floods, extreme temperatures (% of population, average 1990-2009)', 'GHG net emissions/removals by LUCF (Mt of CO2 equivalent)', 
                                'Disaster risk reduction progress score (1-5 scale; 5=best)', 'SF6 gas emissions (thousand metric tons of CO2 equivalent)', 
                                'PFC gas emissions (thousand metric tons of CO2 equivalent)', 'Nitrous oxide emissions (% change from 1990)', 
                                'Nitrous oxide emissions (thousand metric tons of CO2 equivalent)', 'Methane emissions (% change from 1990)', 
                                'Methane emissions (kt of CO2 equivalent)', 'HFC gas emissions (thousand metric tons of CO2 equivalent)', 
                                'Total greenhouse gas emissions (% change from 1990)', 'Total greenhouse gas emissions (kt of CO2 equivalent)', 
                                'Other greenhouse gas emissions (% change from 1990)', 'Other greenhouse gas emissions, HFC, PFC and SF6 (thousand metric tons of CO2 equivalent)', 
                                'CO2 emissions from solid fuel consumption (% of total)', 'CO2 emissions from solid fuel consumption (kt)', 
                                'CO2 emissions (kg per 2017 PPP $ of GDP)', 'CO2 emissions (kg per PPP $ of GDP)', 'CO2 emissions (metric tons per capita)', 
                                'CO2 emissions from liquid fuel consumption (% of total)', 'CO2 emissions from liquid fuel consumption (kt)', 'CO2 emissions (kt)', 
                                'CO2 emissions (kg per 2015 US$ of GDP)', 'CO2 emissions from gaseous fuel consumption (% of total)', 
                                'CO2 emissions from gaseous fuel consumption (kt)', 'CO2 intensity (kg per kg of oil equivalent energy use)', 
                                'Energy use (kg of oil equivalent per capita)', 'Electric power consumption (kWh per capita)', 
                                'Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)', 'Renewable energy consumption (% of total final energy consumption)', 
                                'Electricity production from renewable sources, excluding hydroelectric (% of total)', 
                                'Electricity production from renewable sources, excluding hydroelectric (kWh)', 'Renewable electricity output (% of total electricity output)', 
                                'Access to electricity (% of population)', 
                                'Foreign direct investment, net inflows (% of GDP)', 'Cereal yield (kg per hectare)', 'Average precipitation in depth (mm per year)', 
                                'Agricultural irrigated land (% of total agricultural land)', 'Forest area (% of land area)', 'Forest area (sq. km)', 
                                'Land area where elevation is below 5 meters (% of total land area)', 'Urban land area where elevation is below 5 meters (% of total land area)', 
                                'Urban land area where elevation is below 5 meters (sq. km)', 'Rural land area where elevation is below 5 meters (% of total land area)', 
                                'Rural land area where elevation is below 5 meters (sq. km)', 'Arable land (% of land area)', 'Agricultural land (% of land area)', 
                                'Agricultural land (sq. km)'], axis=1)


# In[ ]:


map_corr(dataframe4)


# In[ ]:


required_c1 = ['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014',"2015", "2016", "2017", "2018", "2019", "2020"]
dataframe5 = dataframe_filtered[required_c1]
dataframe5


# In[ ]:


dataframe5 = dataframe5.fillna(0)

#K means model with 5 clusters
clustering = KMeans(n_clusters=5)
#training the model
cluster = clustering.fit_predict(dataframe5)
#creating a blank dataframe
dataframe6 = pd.DataFrame()
#storing the value of clusters generated from the k-means model into the column "Clusters" of this dataframe
dataframe6["Clusters"] = cluster
#checking unique cluster values
val = dataframe6["Clusters"].unique()
val


# In[ ]:


indexing = list(dataframe_filtered.index.values)
#setting the index of data_ dataframe as that of the original_version1 so that they can be merged later on
dataframe6["index"]=indexing
dataframe6.set_index(["index"], drop=True)


# In[ ]:


dataframe_filtered["index"]=indexing
dataframe_filtered.set_index(["index"], drop=True)


# In[ ]:


#merging both the datasets
dataframe_filtered = pd.merge(dataframe_filtered, dataframe6)
#dropping the unnecessary column index
dataframe_filtered.drop(columns=["index"], inplace=True)
#Number of values in each cluster
dataframe_filtered.Clusters.value_counts()


# In[ ]:


#computing the centroids for each cluster
value_of_centroid = clustering.cluster_centers_
dataframe4 = []
dataframe3 = []
#getting the centroid value of only 2017 and 2021
for i in value_of_centroid:
    for j in range(len(i)):
        x = i[57]
        y = i[60]
    dataframe4.append(i[57])
    dataframe3.append(i[60])


# In[ ]:


#defining the colors in a list
colors = ['#fc0303', '#fa05f2', '#9b05ff', '#0533ed', "#029925"]
#mapping the colors according to unique values in the column Clusters
dataframe_filtered['c'] = dataframe_filtered.Clusters.map(
    {0: colors[0], 1: colors[1], 2: colors[2], 3: colors[3], 4: colors[4]})
#initiating a figure
fig, ax = plt.subplots(1, figsize=(15, 8))
#plotting a scatter plot of data
plt.scatter(dataframe_filtered["2000"], dataframe_filtered["2020"], c=dataframe_filtered.c, alpha=0.7, s=40)
#plotting a scatter plot of centroids
plt.scatter(dataframe4, dataframe3, marker='^', facecolor=colors, edgecolor="black", s=100)
#getting the legend for data
legend = [Line2D([0], [0], marker='o', color='w', label='Cluster or C{}'.format(i + 1),
                          markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]
#getting the legend for centroids
legend_of_centroid = [Line2D([0], [0], marker='^', color='w', label='Centroid of C{}'.format(i + 1),
                          markerfacecolor=mcolor, markeredgecolor="black", markersize=10) for i, mcolor in
                   enumerate(colors)]
#final legend elements
legend.extend(legend_of_centroid)
#setting the legend
plt.legend(handles=legend, loc='upper right', title="Clusters", fontsize=10, bbox_to_anchor=(1.15, 1))
#setting xlabel, ylabel and title
plt.xlabel("2000 data in %", fontsize='18')
plt.ylabel("2020 data in %", fontsize='18')
plt.title("Advance K-means clustering on the basis of Mortality rate, under-5 (per 1,000 live births)", fontsize='20')


# In[ ]:


""" Module errors. It provides the function err_ranges which calculates upper
and lower limits of the confidence interval. """

import numpy as np


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   


# In[ ]:


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from errors import err_ranges  # assuming the errors.py module is in the same directory

# Define the model function
def exponential_growth(t, a, b):
    return a * np.exp(b * t)

# Generate some example data
t = np.linspace(0, 10, 100)
y_true = exponential_growth(t, 1, 0.5)
y_noisy = y_true + np.random.normal(0, 0.1, size=len(t))

# Fit the model to the data
params, cov = curve_fit(exponential_growth, t, y_noisy)

# Get the lower and upper limits of the confidence interval using err_ranges
sigma = np.sqrt(np.diag(cov))
lower, upper = err_ranges(t, exponential_growth, params, sigma)

# Make a plot showing the data, fitted curve, and confidence interval
plt.subplots(1, figsize=(14, 8))
plt.plot(t, y_noisy, 'o', label='Data')
plt.plot(t, exponential_growth(t, *params), '-', label='Fit')
plt.fill_between(t, lower, upper, alpha=0.2, label='Confidence Interval')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Make a prediction for the value at t=20 and its confidence interval
t_new = np.array([20])
y_pred = exponential_growth(t_new, *params)
lower_new, upper_new = err_ranges(t_new, exponential_growth, params, sigma)
print(f'Predicted value at t=20: {y_pred[0]:.2f} ({lower_new[0]:.2f}, {upper_new[0]:.2f})')



# In[ ]:





# In[ ]:




