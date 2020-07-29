
#maps every unique word to the number of times it appears in the data
#CountVectorizer is used to convert the words into vectors that we can fit to our model.
#something that could be changed: Naive Bayes ASSUMES conditional independence between every pair of features. 
#^^ MEANS FOR US: we are assuming that the words in a news article have no impact on each other; 
#we are only examining the probability of seeing each word given a fake or real news story.

import pandas as pd

#                                                              IMPORT AND CLEAN:

#loading the data sets: 
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

#LOOK INT0: Dataframes !!

#flagging - now there is a new column of ones and zeroes 
true["fake_news"] = 0 
fake["fake_news"] = 1


#only unique elements:
true["subject"].unique() #-----> makes an array of unique subjects, in this case: ['politicsNews', 'worldNews']
fake["subject"].unique()

"""
#head fnc: returns the first n rows for the object based on position
print(true.head())
print(fake.head())
"""

#extracting text in pandas
just_text = true["text"]

#only for true cuz false doesn't have the reuters --- 
#i dont REALLY get this so look more into REGEX!
just_text = just_text.str.extractall(r"^.*? - (?P<text>.*)")  
just_text = just_text.droplevel(1)
true = true.assign(text=just_text["text"])   #puts what we did with just_text into the column titled text in the true file 

#combining and saving data: 
#  #making a new concatenated dataframe(set) called df
df = pd.concat([fake, true], axis = 0)
df = df.drop(["subject", "date", "title"], axis = 1) #we did DROP, not CONCAT !! 
df = df.dropna(axis = 0) #don't have to specify which column since the only one that CAN have null values here is the text column, drop any of these nulls.
#MIGHT HAVE TO CHANGE ABOVE FOR DIFFERENT DATA SETS !!^^^ 

#saving df as a new csv file 
clean_text = df.to_csv("./cleaned_news.csv", index=False) #index = False prevents two indeces- gotta figure this out !

