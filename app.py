
import string
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import streamlit as st


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open("vector.pk", "rb"))
model = pickle.load(open("model.pk", "rb"))

st.title("Spam Detection")

input_msg = st.text_area("Enter the mail")

if st.button('Predict'):

    # 1. preprocess
    transformed_msg = transform_text(input_msg)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_msg])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
