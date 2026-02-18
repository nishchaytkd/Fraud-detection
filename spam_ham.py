import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df= pd.read_csv('spam.csv', encoding='latin1')
# print(df.head())

df= df[['v1','v2']]
df.columns= ['target', 'text']

# print(df.head())

df['label'] = df['target'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()

    return text

df['text'] = df['text'].apply(clean_text)

# print(df.head())

vectorizer= TfidfVectorizer()

X=vectorizer.fit_transform(df['text'])

y=df['label']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
# print("n\Model Training complate ")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


cm= confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.title("Confusion metrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
# plt.show()

def predict_spam(text):
    message = clean_text(text)
    message_vector= vectorizer.transform([message])
    prediction= model.predict(message_vector)

    if prediction[0]== 1:
        return "spam"
    else:
        return "Not spam"

# print("Custome Predition: ")
write= input("Enter you message here: ")
print(predict_spam(write))

import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
