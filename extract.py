import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
#Loading Models: 

import pickle
f = open('Multinomial.pickle', 'rb')
Multi_Bayes = pickle.load(f)

f2 = open('Bernoulli.pickle', 'rb')
Bern_Bayes = pickle.load(f2)

f3 = open('count.pickle', 'rb')
word_vec = pickle.load(f3)

##getting HTML from user: 

userIn = input("Please copy a URL: ")
url = userIn
req = urllib.request.urlopen(url)

from bs4 import BeautifulSoup
parsed = BeautifulSoup(req, 'lxml')


# article prediction


def classifier(text):
   
    words = word_vec.transform([text]) 
    
    predict = Multi_Bayes.predict(words)
    predict2 = Bern_Bayes.predict(words)

    if predict[0] and predict2[0]:
        return "Fake News"

    #elif (predict[0] and not predict2[0]) or (predict2[0] and not predict[0]): //fix later  
        #return "Unsure"
    else:
        return "Real News"


for paragraph in parsed.find_all('p'):
    #make list object containing text here: 
    textIn = paragraph.text

print(classifier(textIn))

f.close()
f2.close()
f3.close()

