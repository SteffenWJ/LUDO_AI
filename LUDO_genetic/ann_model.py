import numpy as np
import tensorflow as tf
import keras as ks

# https://github.com/chiatsekuo/Genetic_Algorithm/blob/master/genetic_algorithm_code.py

class ANN_network:
    def __init__(self, state_inputs, layer_1 = 8, layer_2 = 8, weights = None):
        
        if weights == None:
            """Create a numpy array with random values of the size of the input layer"""
            self.weights = np.random.rand(state_inputs)
        
        self.model = tf.keras.models.Sequential()
        self.layer_1 = ks.layers.Dense(layer_1, activation='relu',input_shape =(state_inputs,))
        self.layer_2 = layer_2
        self.model = self.create_model()

#def ann_network(inputs, outputs, first_layer = 8, secound_layer = 8):
#    """Create a neural network with 2 hiden layers with the given number of inputs and outputs."""
#    # Create the model
#    model = tf.keras.models.Sequential()
#    model.add(tf.keras.layers.Dense(first_layer, activation='relu', input_shape=(inputs,)))
#    model.add(tf.keras.layers.Dense(secound_layer, activation='relu'))
#    model.add(tf.keras.layers.Dense(outputs, activation='softmax'))
#    model.compile(optimizer='adam',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
#    return model




