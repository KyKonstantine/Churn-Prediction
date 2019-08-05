# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:57:58 2019

@author: Konstantinos & Giorgos
"""

# Import libraries and classifiers 
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

''' Data Preprocess'''

# Import training and test set
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

# Overview of data
print(df_train.info())
df_train.head()
print(df_test.info())
df_test.head()

# Chech for null values
df_train.isnull().sum()
df_test.isnull().sum()

# Check if it's an unbalanced dataset
df_train.groupby('churn').size()

# Check unique values of State and Area_code
print(df_train.area_code.unique())
print(df_train.state.unique())

# Save column ID for later use and drop it drom test set
ids=df_test.id
df_test.drop(['id'],axis=1,inplace=True)

# Change categorical values into numerical for both datasets
df_train.churn.replace(('yes', 'no'), (1, 0), inplace=True)
df_train.international_plan.replace(('yes', 'no'), (1, 0), inplace=True)
df_train.voice_mail_plan.replace(('yes', 'no'), (1, 0), inplace=True)
df_test.international_plan.replace(('yes', 'no'), (1, 0), inplace=True)
df_test.voice_mail_plan.replace(('yes', 'no'), (1, 0), inplace=True)

# Confirm changes
print(df_train.head())
print(df_test.head())


# Create two bins for State based on the proportion of churners
df1=df_train.groupby(by='state')[['state','churn']].agg(lambda x: x.sum()/ x.count()).reset_index()
np.mean(df1.churn.values)
group1=(df1.loc[df1.churn>0.14]).state.values
group2=(df1.loc[df1.churn<=0.14]).state.values


# Convert State and Area_code into dummies 
def preprocess(df):
    df_dummies=pd.get_dummies(df.area_code) 
    df=pd.concat([df_dummies,df],axis=1)
    
    for i in group1:
        for j in group2:
            df.state.replace((i,j), ('group1','group2'), inplace=True)
    
    df_dummies2=pd.get_dummies(df.state)
    df=pd.concat([df_dummies2,df],axis=1)
    
    df.drop(['state','area_code'],inplace=True,axis=1)
    return df
	

df_train=preprocess(df_train)
df_test=preprocess(df_test)


# Confirm changes
print(df_train.head())
print(df_test.head())


''' Feature Engineering '''

# Use RandomForestClassifier to check features importances
train_array=df_train.drop(['churn'],axis=1).values
target_array=df_train.churn.values

clf=RandomForestClassifier()
clf.fit(train_array,target_array)

importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]
names = [(df_train.drop(['churn'],axis=1)).columns[i] for i in indices]
plt.figure()
plt.title("Feature Importance")
plt.bar(range(train_array.shape[1]), importances[indices])
plt.xticks(range(train_array.shape[1]), names, rotation=90)
plt.show()


# Dropping least important features
drops = ['group1','group2','area_code_408','area_code_415','area_code_510']
df_train.drop(drops,inplace=True,axis=1)
df_test.drop(drops,inplace=True,axis=1)

# Confirm changes
print(df_train.columns)
print(df_test.columns)



# Create new features in both datasets
# Create feature that counts the total number of minutes without International calls
total_mins=df_train.total_day_minutes + df_train.total_eve_minutes + df_train.total_night_minutes
total_mins1=df_test.total_day_minutes + df_test.total_eve_minutes + df_test.total_night_minutes
# Create feature that counts the total number of calls without International calls
total_calls=df_train.total_day_calls + df_train.total_eve_calls + df_train.total_night_calls 
total_calls1=df_test.total_day_calls + df_test.total_eve_calls + df_test.total_night_calls
# Create feature that counts the total charge with International calls
total_charge=df_train.total_day_charge + df_train.total_eve_charge + df_train.total_night_charge + df_train.total_intl_charge
total_charge1=df_test.total_day_charge + df_test.total_eve_charge + df_test.total_night_charge + df_test.total_intl_charge
# Droppingthe features used to create the new features 
df_train.drop(['total_eve_minutes','total_day_minutes','total_night_minutes','total_day_calls','total_eve_calls','total_night_calls','total_day_charge','total_eve_charge','total_night_charge','total_intl_charge'],inplace=True,axis=1)
df_test.drop(['total_eve_minutes','total_day_minutes','total_night_minutes','total_day_calls','total_eve_calls','total_night_calls','total_day_charge','total_eve_charge','total_night_charge','total_intl_charge'],inplace=True,axis=1)
# Merging datasets with new features
df_train=pd.concat([df_train,total_mins,total_calls,total_charge],axis=1)
df_test=pd.concat([df_test,total_mins1,total_calls1,total_charge1],axis=1)


# Confirm changes
print(df_train.head())
print(df_test.head())


''' Compare classifiers with 20-fold CV '''

categorical_features_indices =[1,2]

classifiers=[GradientBoostingClassifier(),RandomForestClassifier(),
             XGBClassifier(),CatBoostClassifier(cat_features=categorical_features_indices,verbose=False)]


from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=20 , random_state=1)

from sklearn.model_selection import cross_val_score
accuracies=[]
for clf in classifiers:
    scores = cross_val_score(clf,df_train.drop(['churn'],axis=1).values,df_train.churn.values,scoring='accuracy',verbose=1,cv=sss)
    accuracies.append(np.mean(scores)) 

print(accuracies)

# Plot accuracies according to each classifier
classifier_names=['GradientBoostingClassifier','RandomForestClassifier',
             'XGBClassifier','CatBoostClassifier']

y_pos = np.arange(len(classifier_names))
plt.barh(y_pos, accuracies,color='teal')
plt.title('Classifier mean accuracy comparison')
plt.yticks(y_pos, classifier_names)
plt.xlim([0.96,0.99])
plt.show()



''' Kaggle predictions for each classifier '''

# Random Forest Classifier

clf=RandomForestClassifier(n_estimators=150)

train_array=df_train.drop(['churn'],axis=1).values
target_array=df_train.churn.values
test_array=df_test.values

# Train classifier and make predictions
clf.fit(train_array,target_array)
y_predict=pd.DataFrame(clf.predict(test_array))

# Export predictions according to sample submission
y_predict.replace((1, 0), ('yes', 'no'), inplace=True)
df_submit=pd.concat([ids,y_predict],axis=1)
df_submit.columns=['id','churn']
df_submit.to_csv('SubmissionRF2.csv',index=False)


# Gradient Boosting Classifier  

clf=GradientBoostingClassifier()

train_array=df_train.drop(['churn'],axis=1).values
target_array=df_train.churn.values
test_array=df_test.values

# Train classifier and make predictions
clf.fit(train_array,target_array)
y_predict=pd.DataFrame(clf.predict(test_array))

# Export predictions according to sample submission
y_predict.replace((1, 0), ('yes', 'no'), inplace=True)
df_submit=pd.concat([ids,y_predict],axis=1)
df_submit.columns=['id','churn']
df_submit.to_csv('SubmissionGBC.csv',index=False)


# XGBoost Classifier

clf=XGBClassifier(silent=True)

train_array=df_train.drop(['churn'],axis=1).values
target_array=df_train.churn.values
test_array=df_test.values

# Train classifier and make predictions 
clf.fit(train_array,target_array)
y_predict=pd.DataFrame(clf.predict(test_array))

# Export predictions according to sample submission
y_predict.replace((1, 0), ('yes', 'no'), inplace=True)
df_submit=pd.concat([ids,y_predict],axis=1)
df_submit.columns=['id','churn']
df_submit.to_csv('SubmissionXGB.csv',index=False)


# CatBoost Classifier

categorical_features_indices =[1,2]

clf=CatBoostClassifier(cat_features=categorical_features_indices,verbose=False)

train_array=df_train.drop(['churn'],axis=1).values
target_array=df_train.churn.values
test_array=df_test.values

# Train classifier and make predictions
clf.fit(train_array,target_array)
y_predict=pd.DataFrame(clf.predict(test_array))

# Export predictions according to sample submission
y_predict.replace((1, 0), ('yes', 'no'), inplace=True)
df_submit=pd.concat([ids,y_predict],axis=1)
df_submit.columns=['id','churn']
df_submit.to_csv('SubmissionCat.csv',index=False)



''' Voting Classifier '''

from sklearn.ensemble import  VotingClassifier

class CatBoostClassifierInt(CatBoostClassifier):
    def predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, thread_count=1, verbose=None):
        predictions = self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose)

        # This line is the only change I did
        return np.asarray(predictions, dtype=np.int64).ravel()



classifiers = [("classifier1", GradientBoostingClassifier()),
               ("classifier2", XGBClassifier()),
               ("classifier3", RandomForestClassifier()),
               ("classifier4", CatBoostClassifierInt(cat_features=categorical_features_indices,verbose=False))]

# Soft Voting
clf = VotingClassifier(classifiers, voting='soft')

train_array=df_train.drop(['churn'],axis=1).values
target_array=df_train.churn.values
test_array=df_test.values

# Train classifier and make predictions
clf.fit(train_array,target_array)
y_predict=pd.DataFrame(clf.predict(test_array))

# Export predictions according to sample submission
y_predict.replace((1, 0), ('yes', 'no'), inplace=True)
df_submit=pd.concat([ids,y_predict],axis=1)
df_submit.columns=['id','churn']
df_submit.to_csv('SubmissionVotS.csv',index=False)


# Hard Voting
clf = VotingClassifier(classifiers, voting='hard')

train_array=df_train.drop(['churn'],axis=1).values
target_array=df_train.churn.values
test_array=df_test.values

# Train classifier and make predictions
clf.fit(train_array,target_array)
y_predict=pd.DataFrame(clf.predict(test_array))

# Export predictions according to sample submission
y_predict.replace((1, 0), ('yes', 'no'), inplace=True)
df_submit=pd.concat([ids,y_predict],axis=1)
df_submit.columns=['id','churn']
df_submit.to_csv('SubmissionVotH.csv',index=False)