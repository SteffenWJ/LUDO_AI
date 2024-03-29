import numpy as np
import tensorflow as tf
import keras as ks
from keras.utils import plot_model
from keras.backend import clear_session


import gc

class ANN_network:
    #Creates the ANN model using tensorflow and keras to make creating the model easier for training my the genetic alghorithm:
    #To DO is to set the weights more randomly
    def get_the_weights(self):
        return self.model.layers[0].get_weights()[0], self.model.layers[1].get_weights()[0], self.model.layers[2].get_weights()[0], self.model.get_weights()
    def get_all_the_weights(self):
        return self.model.get_weights()
    def get_model_view(self):
        return self.model.summary()

    
    def plot_model_all(self):
        plot_model(self.model, to_file='network_structure.png', show_shapes=True, show_layer_names=True,)
    
    #Should maybe be made into a call function
    def use_model(self,input, size = 36):
        #Takes the input of the game and returns a 1x4 predition of what piece to move.
        input_state = np.reshape(input, (1, size)) #Reshapes the input to fit the model the size is to make it easier to test later.
        output = self.model.predict(input_state, verbose=0)
        pawn_values = np.argsort(output)[0][::-1]
        return pawn_values #Verbose is turned off to make it easier to read the output.
    
    def clear_the_session():
        clear_session()
        
    def __init__(self, state_inputs = 36, layer_1 = 64, layer_2 = 64, weights = None): #was 21
        
        self.model = tf.keras.models.Sequential()
        self.layer_1 = ks.layers.Dense(layer_1, activation='sigmoid', name = "Layer_1",input_shape =(state_inputs,),use_bias=False)
        self.layer_2 = ks.layers.Dense(layer_2, activation='sigmoid', name = "Layer_2",use_bias=False)
        self.output_layer = ks.layers.Dense(4, activation='softmax', name = "Output",use_bias=False) #Removed the bias as we dont want it to influence the output of which piece to move.
        
        self.model.add(self.layer_1)
        self.model.add(self.layer_2)
        self.model.add(self.output_layer)
        
        if weights != None:
            self.model.set_weights(weights)
        
        #print("Model built")