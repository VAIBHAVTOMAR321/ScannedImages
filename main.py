import cv2
import easyocr
import streamlit as st
import os
import matplotlib.pyplot as plt
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

st.title("Extracting Data From Images")

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)

im_1_path = filename

def recognize_text(img_path):
    '''loads an image and recognizes text.'''

    reader = easyocr.Reader(['en'])
    return reader.readtext(img_path)

result = recognize_text(im_1_path)


img_1 = cv2.imread(im_1_path)
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
fig = plt.figure()
plt.imshow(img_1)
plt.axis("off")
st.pyplot(fig)

def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary

def summerize_ocr_text(img_path):
    '''loads an image, recognizes text, and overlays the text on the image.'''

    # recognize text
    result = recognize_text(img_path)
    list1=[]
    # if OCR prob is over 0.5, overlay bounding box and text
    for (bbox, text, prob) in result:
       if prob >= 0.5:
            # display
            list1.append(text)

    # Result Extracted
    result1=' '.join(list1)
    # print(result1)
    st.header('Extracted Result')
    st.write(result1)
    st.header('Summarized Text')
    st.write(summarize(result1,0.5))

# button for using Overlayed Text
run=st.button('RUN')
if run:
    summerize_ocr_text(im_1_path)