import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import text2emotion as te
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request, jsonify
import re
import string
from nltk.corpus import stopwords


app = Flask(__name__)
app.static_folder = 'static'

lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
#Data cleanup for emotion detection
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back to a single string
    preprocessed_text = " ".join(tokens)

    return preprocessed_text
#for music recommendation 
import requests

api_key='6b78d62d953f2e279be790c92e8c92c1'
limit= '5'

def get_tracks(tag):
    url = f"http://ws.audioscrobbler.com/2.0/?method=tag.getTopTracks&tag={tag}&limit={limit}&api_key={api_key}&format=json"
    response= requests.get(url)
    data= response.json()
    tracks= data['tracks']['track']
    
    return(tracks)

    

@app.route("/")
def home():
   max_emo = "neutral"
   return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    reply = chatbot_response(userText)
    emo_userText= preprocess_text(userText)
    emo = te.get_emotion(emo_userText)
    max_emo = max(emo, key=emo.get)

    audio= get_tracks(max_emo)
    
    response = {"reply": reply, "max_emo": max_emo, "audio":audio}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
