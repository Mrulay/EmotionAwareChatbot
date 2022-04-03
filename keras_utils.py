import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import  Input
import json

with open(r'C:\Users\Mrulay\OneDrive - University of Windsor\uWindsor\COMP 8700 - Intro to AI\Project\data\chatbot\vocab.json') as json_file:
    vocab = json.load(json_file)
    
VOCAB_SIZE = len(vocab)

def create_inference_model():
    encoder_inputs = Input(shape=(18,))
    encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200 , mask_zero=True )(encoder_inputs)
    encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
    encoder_states = [state_h , state_c]

    decoder_inputs = Input(shape=(18,))
    decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200 , mask_zero=True)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE , activation=tf.keras.activations.softmax) 
    output = decoder_dense ( decoder_outputs )

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
    
    w_encoder_embeddings = np.load('models\chatbot\layers\embedding_2.npz', allow_pickle=True)
    w_decoder_embeddings = np.load('models\chatbot\layers\embedding_3.npz', allow_pickle=True)
    w_encoder_lstm = np.load('models\chatbot\layers\lstm_2.npz', allow_pickle=True)
    w_decoder_lstm = np.load('models\chatbot\layers\lstm_3.npz', allow_pickle=True)
    w_dense = np.load('models\chatbot\layers\dense_1.npz', allow_pickle=True)
    
    model.layers[2].set_weights(w_encoder_embeddings['arr_0'])
    model.layers[3].set_weights(w_decoder_embeddings['arr_0'])
    model.layers[4].set_weights(w_encoder_lstm['arr_0'])
    model.layers[5].set_weights(w_decoder_lstm['arr_0'])
    model.layers[6].set_weights(w_dense['arr_0'])
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model