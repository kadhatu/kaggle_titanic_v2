# coding: utf-8
# submssion accuracy gets public score of 0.81339

# Loading a bunch of stuff
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, \
                    ensemble, discriminant_analysis, gaussian_process
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix


# Loading dataset
train_df = pd.read_csv("/Users/yangxinchen/Documents/Python/\
  python_wheel/machine_learning/kaggle_titanic/train.csv")
test_df = pd.read_csv("/Users/yangxinchen/Documents/Python/\
  python_wheel/machine_learning/kaggle_titanic/test.csv")
data_df = train_df.append(test_df)


# Create feature 'Title' from 'Name' col
data_df['Title'] = data_df['Name']
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    
# mapping 'Title'
mapping = {'Mlle':'Miss', 'Major':'Mr', 'Col':'Mr', 'Sir':'Mr', 'Don':'Mr', 'Mme':'Miss',
          'Jonkheer':'Mr', 'Lady':'Mrs', 'Capt':'Mr', 'Countess':'Mrs',
          'Ms':'Miss', 'Dona':'Mrs'}

data_df.replace({'Title':mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']


# set missing 'Age' grouped by title and use median value
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute


# split train and test dataset
train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]


# drop 'Title'
data_df.drop('Title', axis = 1, inplace=True)


# Adding Family_Size (Parch + SibSp)
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']

train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]


# Adding Family_Survival
data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data_df[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 
'PassengerId', 'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print ("Number of passengers with family survival information:",
      data_df.loc[data_df['Family_Survival']!=0.5].shape[0])

train_df['Family_Survival'] = data_df['Family_Survival'][:891]
test_df['Family_Survival'] = data_df['Family_Survival'][891:]


# Number of passenger with family/group survival information
for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] ==0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if smax == 1.0:
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif smin == 0.0:
                    data_df.loc[data_df['PassengerId'] ==passID, 'Family_Survival'] = 0
print("Number of passenger with family/group survival information:" + \
      str(data_df[data_df['Family_Survival']!=0.5].shape[0]))


# Making FARE BINS
data_df['Fare'].fillna(data_df['Fare'].median(), inplace=True)

data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)
label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])

train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]

train_df.drop(['Fare'], 1, inplace=True)
test_df.drop(['Fare'], 1, inplace=True)


# Make AGE BINS
data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)
label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])

train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]

train_df.drop(['Age'], 1, inplace=True)
test_df.drop(['Age'], 1, inplace=True)


# Mapping SEX and cleaning data(dropping garbage)
train_df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
test_df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
              'Embarked'], axis = 1, inplace=True)
test_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
              'Embarked'], axis = 1, inplace=True)


# Began Training

X = train_df.drop('Survived', 1)
y = train_df['Survived']
X_test = test_df.copy()


# Scaling features
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)


# Grid Search CV
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1, 50, 5))
hyperparams = {'algorithm':algorithm, 'weights':weights, 
              'leaf_size':leaf_size, 'n_neighbors':n_neighbors}

gd = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=hyperparams, 
                  verbose=True, cv=10, scoring='roc_auc')
gd.fit(X, y)

print(gd.best_score_)
print(gd.best_estimator_)

gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X_test)


# Using another K
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
                          metric_params=None, n_jobs=1, n_neighbors=6, p=2,
                          weights='uniform')
knn.fit(X, y)
y_pred = knn.predict(X_test)


# Making submission
temp = pd.DataFrame(pd.read_csv('/Users/yangxinchen/Documents/Python/\
    python_wheel/machine_learning/kaggle_titanic/test.csv')['PassengerId'])
temp['Survived'] = y_pred

temp.to_csv("/Users/yangxinchen/Documents/Python/python_wheel/\
    machine_learning/kaggle_titanic_top4per/submission_v2", index=False)

