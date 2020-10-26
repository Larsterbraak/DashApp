import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

# Discriminator network for GAN in latent space in Tensorflow 2.x
class Discriminator(Model):
    def __init__(self, tensorboard_folder_path, hparams, 
                 hidden_dim, dropout=0.1):
        super(Discriminator, self).__init__()
        self.LSTM1 = Bidirectional(LSTM(units=7,
                                        return_sequences=True,
                                        kernel_initializer = 'he_uniform',
                                        dropout = 0,
                                        recurrent_dropout = 0,
                                        input_shape=(20,hidden_dim),
                                        name='LSTM1'))
        self.Dropout1 = Dropout(dropout)
        self.LSTM2 = Bidirectional(LSTM(units=4,
                                        return_sequences=False,
                                        kernel_initializer = 'he_uniform',
                                        dropout = 0,
                                        recurrent_dropout = 0,
                                        name='LSTM2'))
        self.Dropout2 = Dropout(dropout)
        self.Dense1 = Dense(units=1,
                            activation=None,
                            name='Dense1')
        self.graph_has_been_written=False
        self.i = 0
        self.tensorboard_folder_path = tensorboard_folder_path
        
    def call(self, x, training=True, **kwargs): # Implement training = False when testing 
        x = self.LSTM1(x)
        x = self.Dropout1(x, training)
        x = self.LSTM2(x)
        x = self.Dropout2(x, training)
        x = self.Dense1(x)
        
        # Print the graph in TensorBoard
        if not self.graph_has_been_written and self.i == 0:
            #model_graph = x.graph
            #writer = tf.compat.v1.summary.FileWriter(logdir=self.tensorboard_folder_path,
            #                                         graph=model_graph)
            #writer.flush()
            self.graph_has_been_written = True
            print("Wrote eager graph to:", self.tensorboard_folder_path)
        
        self.i = self.i + 1 # Log the number of calls to the discriminator model 
        return x
    
    def predict(self, x, training=False, **kwargs):
        x = self.LSTM1(x) # Perform the normal training steps
        x = self.Dropout1(x, training)
        x = self.LSTM2(x)
        x = self.Dropout2(x, training)
        x = self.Dense1(x)
        
        x = tf.math.sigmoid(x)        
        return x
        
        