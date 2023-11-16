import nltk 
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
import random



lemmatizer= WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore_words=['!','?']

data_file= open('data.json').read()
intents= json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w,intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes= sorted(list(set(classes)))

print(len(documents)," Documents")
print(len(classes),"   Classes", classes)
print(len(words)," words",words)

pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))

training=[]
output_empty= [0]*len(classes)

for doc in documents:
    bag=[]
    pattern_words=doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1)if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)
print("model created")


import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Load the test data
test_x = np.array(train_x)  
test_y = np.array(train_y)  

# Evaluate the model
loss, accuracy = model.evaluate(test_x, test_y)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
