import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Destroys the current graph and session and change all layers dtype to float64
tf.keras.backend.clear_session()
#policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
#tf.keras.mixed_precision.experimental.set_policy(policy) 

# Enable XLA
#tf.config.optimizer.set_jit(True)

# Embedder network in Tensorflow 2.x
class Embedder(Model):
    def __init__(self, tensorboard_folder_path, hparams, 
                 hidden_dim, dimensionality, dropout=0.1):
        super(Embedder, self).__init__()
        self.LSTM1 = LSTM(units=10, 
                          return_sequences=True,
                          input_shape=(20,dimensionality),
                          kernel_initializer = 'he_uniform',
                          dropout = dropout,
                          recurrent_dropout = 0,
                          name = 'LSTM1')
        self.Dropout1 = Dropout(dropout)
        self.Dense1 = Dense(units=hidden_dim, # [4 x 4] weight grads + [4,1] bias grads
                            activation='sigmoid', # To ensure [0, 1]
                            name = 'Dense1')
        self.graph_has_been_written=False
        self.tensorboard_folder_path = tensorboard_folder_path

    def call(self, x, training=True, **kwargs): # Implement training = False when testing
        x = self.LSTM1(x)
        x = self.Dropout1(x, training)
        x = self.Dense1(x)
        return x