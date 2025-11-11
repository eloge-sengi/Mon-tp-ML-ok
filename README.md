# Mon-tp-ML-ok

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#chargement de dataset 
titanic=sns.load_dataset('titanic')

titanic.info()

titanic['survived'].value_counts()

titanic.loc[titanic['sex']=='male']

titanic['age'].mean()

titanic['age'].mode()

titanic.isnull().sum()

#Nettoyage de données
titanic['age'].fillna(titanic['age'].mean(),inplace=True)

titanic['embarked'].fillna(titanic['embarked'].mode()[0],inplace=True)

titanic.drop(columns=['deck','embark_town','alive'],inplace=True)

titanic.drop(columns=['class','who','adult_male'],inplace=True)

titanic['sex']

titanic['sex']=titanic['sex'].map({'male':0,'female':1})

titanic['embarked']

sns.countplot(x='survived',data=titanic)
plt.title('Répartition des survivants')
plt.show()

sns.histplot(data=titanic,x='age',hue='survived',kde=True,bins=30)
plt.title('Age selon la survie')
plt.show()

sns.boxplot(x='pclass',y='age',hue='survived',data=titanic)
plt.title('Age par classe et survie')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(titanic.corr(), annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show

sns.histplot(titanic['age'], kde=True)
plt.title("Distribution de l'age des passagers")
plt.show()
