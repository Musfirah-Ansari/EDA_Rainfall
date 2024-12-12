# EDA_Rainfall

#libraries/dependencies
import numpy as np
import pandas as pd #for forming data into tabular form(rows/coln) instead of structured

#for graphical visuaalization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.pipeline import Pipeline
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#warnings.filterwarnings('ignore')

# DATA CLEANING____

#input - features
#output - Target variables/labels
#strip - trim(removing leading/trailing spaces but not in between sapces)
#object is string here
#if more than half values are missing in the column, then drop that column

#STEP1 - drop irrelevant column
#STEP2 - remove missing values
#for continous value, mean and median. For discrete(categorical)-mode 
#1-value drop(if 1 missing out of 1000), coln drop(if 800 missing out of 1000), see the dist of that coln, if normal(means its numeical not categorical) then place the avg of it in the missing value place, if not normal(skewd) then take median(take ascending then tak mid val) and replace with median
#continous data and discrete data(categorical), and for categorical data, we'll take mode(most frequent val)
#Outliers: Value that disturbs the overall calculation of data
#But sometimes, its necessary to keep the outliers depending on the data
#Distribution: y axis(counts-no of values), x-axis(values)
#standarddization will be applied only at the numerical data not on categorical, thats why separate the numerical & categorical data initially. if whole data is numerical then there is no need.
# decide between numerical & categorical , see the nature of feature first
#histogram for distribution(numerical)
#count plot for categorical
#for outliers we use box plot
#for filling null values one by one .fillna manual way
#function for imputation, import it then apply, if missing values columns are many

#Label encoding: converting objects into numbers, yes,no into 1 or 0 through .map function but has limitation of doing one by one colmn not all at the same time, .replace can also be used but has limitation of converting data type of all val of that colmn through for loop
#inplace = true means storing the same in data variable
#IQR = final(Q3) - initial(Q1)
#lower bound: LB<data: Q1-(Q3 * 1.5)
#upper bound: UB>data: Q3-(Q1 * 1.5)
#.groupby.mean for target value (output variable) for estimation
#where (condition, value if true, val if false)
#Correlation heatmap e.g clod is positively co related with rainfall
#problem is multicollinearity, multiple columns saying the same thing e.g clouds, wind speed, humidity are saying that rainfall will happen, so there's no need for all 3, you can use the 1 and remove the rest.
#class imbalance issue: when output variable has more yes and less no
#1 way to correct it is under sampling, make the classes equal, very impure
#2nd way is over sampling, ver very impure
# SMOTE: A more advanced over-sampling method that generates synthetic data points for the minority class, reducing the risk of overfitting compared to simple duplication.
#fitting and transformation, fitting is making a formula and transformation is inserting the data in to that formula.
#x is input. y is output, axis=1 means dropping col not row
#In the end, if numerical then do the standarddization, if categorical then one hot encoding


