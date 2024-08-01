import flask
from flask import render_template, request
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocessing(text):
    text = text.lower()

    text = re.sub(r'[^A-Za-z]',' ', text)

    token = word_tokenize(text)

    words = [i for i in token if i not in stopwords.words("english")]

    processed = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(processed)


app = flask.Flask(__name__)
classifier = joblib.load(r'Models\Textclassifier_logreg.pkl')
preprocess = joblib.load(r"Models\Tfidf.pkl")

@app.route('/', methods=['POST','GET'])
def home():
    if request.method == 'POST':
        text = request.form.get('textinput')
        text = preprocessing(text)
        text = preprocess.transform([text])
        final = classifier.predict(text)
        return render_template('home.html',result=final[-1])
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)