import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#load model
model = load_model("next_word_lstm.h5")

#load the tokenizer
with open("tokenizer.pickle","rb") as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model , tokenizer,text,max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length: # in case length >= max seq length then 
        token_list = token_list[-(max_sequence_length-1):]
    token_list = pad_sequences([token_list], maxlen= max_sequence_length, padding= "pre")
    predicted = model.predict(token_list , verbose =0)
    # print(predicted)
    # multiple values predicted based on probability 
    # take max one
    predicted_word_index = np.argmax(predicted, axis =1)
    print(predicted_word_index)
    for word, index in tokenizer.word_index.items():
        #get word corr to the index in dictionary
        if index == predicted_word_index:
            return word
    return None

st.title("Next word prediction with lstm and early stopping")
input_text = st.text_input("enter the sequence of words","to be or not to")
if st.button("Predict next word"):
    max_sequence_length = model.input_shape[1]
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    st.write("next word ",next_word)
