from logging import warning
from keras_utils import create_inference_model
import warnings
from tensorflow.keras.preprocessing.sequence import pad_sequences
from better_profanity import profanity
from preprocesstext import preProcessText
import numpy as np
import pickle
import pandas as pd
import json 
from nltk.corpus import wordnet
warnings.filterwarnings('ignore')

VAD = pd.read_excel(r'C:\Users\Mrulay\OneDrive - University of Windsor\uWindsor\COMP 8700 - Intro to AI\Project\data\chatbot\NRC-VAD-Lexicon-Aug2018Release\VAD_Database.xlsx')
vadWords = VAD['word'].to_list()
V = VAD['V'].to_list()


with open(r'C:\Users\Mrulay\OneDrive - University of Windsor\uWindsor\COMP 8700 - Intro to AI\Project\data\chatbot\vocab.json') as json_file:
    vocab = json.load(json_file)
    
inv_vocab = {w:v for v, w in vocab.items()}

VOCAB_SIZE = len(vocab)

with open('models\emotionDetection\linearSVC.pkl', 'rb') as fid:
    emotion_detector = pickle.load(fid)

with open(r'C:\Users\Mrulay\OneDrive - University of Windsor\uWindsor\COMP 8700 - Intro to AI\Project\code\emotionDetection\tfidf.pk', 'rb') as f:
    tfidf = pickle.load(f)


def str_to_tokens(sentence:str):
    words = preProcessText(sentence)
    tokens_list = list()
    for word in words:
        try:
            tokens_list.append(vocab[word]) 
        except:
            tokens_list.append(vocab['<OUT>'])
    return pad_sequences([tokens_list], 18, padding='post', truncating='post')

def easify_sent(sentence:str):
    dec_trans_list = sentence.split()

    counts = []
    for word in dec_trans_list:
        try: 
            counts.append(vocab[word])
        except:
            counts.append(1)
    imp_word = inv_vocab[counts[np.argmax(counts)]]
    syn = list()
    if imp_word in vadWords:
        currV = V[vadWords.index(imp_word)]
        for synset in wordnet.synsets(imp_word):
            for lemma in synset.lemmas():
                syn.append(lemma.name())
        for word in syn:
            if word in vadWords:
                newV = V[vadWords.index(word)]
            if newV > currV:
                newWord = vadWords[V.index(newV)]
                currV = newV
        if newWord:
            dec_trans_list[dec_trans_list.index(imp_word)] = newWord
            decoded_translation = " ".join(dec_trans_list)
            return decoded_translation
        else:
            return sentence
    else:
        return sentence



enc_model , dec_model = create_inference_model()

emotions = []
for _ in range(10):
    syn = list()
    if _==0:
        print("Hi! How are you feeling today?")
    else:
        print("Tell me a little more...")
    text_input = (input('Enter question: '))
    text = preProcessText(text_input)
    text = tfidf.transform([text]).toarray()
    emotions.append(emotion_detector.predict(text)[0])
    text_input = str_to_tokens(text_input)
    states_values = enc_model.predict(text_input)
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = vocab['<SOS>']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = inv_vocab[sampled_word_index] + ' '
        
        if sampled_word == 'end' or len(decoded_translation.split())>18:
            stop_condition = True
        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word  
        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 18:
            stop_condition = True 

        empty_target_seq = np.zeros((1,1))  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ]

    decoded_translation = easify_sent(decoded_translation)
    decoded_translation = profanity.censor(decoded_translation)
    print(decoded_translation)

