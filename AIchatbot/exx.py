import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from text2emotion import get_emotion

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

# Example usage
input_text = "“My love for you shall grow and not perish. I will live to love you more and more.)"
preprocessed_text = preprocess_text(input_text)
emotion = get_emotion(preprocessed_text)
print(emotion)
