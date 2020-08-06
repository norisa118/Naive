# main.py 
from flask import Flask,render_template,url_for,request
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
import pickle


app = Flask(__name__)

@app.route("/") # this route is the default one
def home():
    return  render_template("home.html")

@app.route("/naive", methods=['POST'])
def naive():

    #importing models
    import pickle
    f = open('Multinomial.pickle', 'rb')
    Multi_Bayes = pickle.load(f)

    f2 = open('Bernoulli.pickle', 'rb')
    Bern_Bayes = pickle.load(f2)

    f3 = open('count.pickle', 'rb')
    word_vec = pickle.load(f3)

    #getting URL, parsing into text, then returning output 

    if request.method == 'POST':
        url = request.form['url']
        data = url
        req = urllib.request.urlopen(data)

        from bs4 import BeautifulSoup
        parsed = BeautifulSoup(req, 'lxml')

    # article prediction

    def classifier(text):
    
        words = word_vec.transform([text]) 
        
        predict = Multi_Bayes.predict(words)
        predict2 = Bern_Bayes.predict(words)

        ac1 = Multi_Bayes.predict_proba(words)
        ac2 = Bern_Bayes.predict_proba(words)


        if predict[0] and predict2[0]:
            return (1, ac1, ac2)
        #elif (predict[0] and not predict2[0]) or (predict2[0] and not predict[0]): //fix later  
            #return "Unsure"
        else:
            return (0, ac1,ac2)


    for paragraph in parsed.find_all('p'):
        #make list object containing text here: 
        textIn = paragraph.text

    my_prediction = classifier(textIn)

    f.close()
    f2.close()
    f3.close()

    return render_template('naive.html', prediction = my_prediction)

if  __name__  ==  "__main__":
    app.run(debug=True)