#                                                              TRAIN CLASSIFIER:

import pandas as pd

df = pd.read_csv("cleaned_news.csv")

DV = "fake_news" #name of the column that has all the 1's and 0's 
#we don't need ^^, so we drop it: 
X = df.drop([DV], axis = 1) #indpeendent variable set 
y = df[DV] #dependent variable set

#train on 75% of the dataset and test on the remaining 25%. This goes back to the idea of overfitting
#want to make sure our model performs well on data it has never seen before.

from sklearn.model_selection import train_test_split
#Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
#Other parameters can be train size (TEST is auto split to 0.25 if neither is specified, so in this case, we could have put nothing)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

#making a key,value dictionary for word counts:
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_features = 5000) #CHANGE THIS TO SEE IF WE CAN INCREASE ACCURACY ! 
X_train_counts = count_vect.fit_transform(X_train["text"])   
X_test = count_vect.transform(X_test["text"]) 

from sklearn.naive_bayes import MultinomialNB
Naive = MultinomialNB()
Naive.fit(X_train_counts, y_train)

from sklearn.naive_bayes import BernoulliNB
Naive2 = BernoulliNB()
Naive2.fit(X_train_counts, y_train) 

##accuracy stats:
from sklearn.metrics import accuracy_score
#Multinomial
predictions_NB = Naive.predict(X_test)
print("Accuracy Score:",accuracy_score(predictions_NB, y_test)*100)

#Bernoulli
predictions_NB2 = Naive2.predict(X_test)
print("Accuracy Score 2:",accuracy_score(predictions_NB2, y_test)*100)


# _____________________________________ SAVING MODELS_______________________________________________________________

import pickle
f = open('Multinomial.pickle', 'wb')
pickle.dump(Naive, f)
f.close()

f2 = open('Bernoulli.pickle', 'wb')
pickle.dump(Naive2, f2)
f2.close()

f3 = open('count.pickle', 'wb')
pickle.dump(count_vect, f3)
f3.close()
