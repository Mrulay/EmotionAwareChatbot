from keras_utils import create_inference_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocesstext import preProcessText
import numpy as np
import json 

with open(r'C:\Users\Mrulay\OneDrive - University of Windsor\uWindsor\COMP 8700 - Intro to AI\Project\data\chatbot\vocab.json') as json_file:
    vocab = json.load(json_file)
    
inv_vocab = {w:v for v, w in vocab.items()}

VOCAB_SIZE = len(vocab)

def str_to_tokens(sentence:str):
    words = preProcessText(sentence)
    tokens_list = list()
    for word in words:
        try:
            tokens_list.append(vocab[word]) 
        except:
            tokens_list.append(vocab['<OUT>'])
    return pad_sequences([tokens_list], 18, padding='post', truncating='post')

enc_model , dec_model = create_inference_model()

for _ in range(10):
    states_values = enc_model.predict( str_to_tokens( input( 'Enter question : ' ) ) )
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
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    print(decoded_translation)