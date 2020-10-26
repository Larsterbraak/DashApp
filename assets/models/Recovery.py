from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Recovery network in Tensorflow 2.x
class Recovery(Model):
    def __init__(self, tensorboard_folder_path, hparams, 
                 hidden_dim, dimensionality, dropout=0.1):
        super(Recovery, self).__init__()
        self.LSTM1 = LSTM(units=hidden_dim, 
                          return_sequences=True,
                          kernel_initializer = 'he_uniform',
                          dropout = 0,
                          recurrent_dropout = 0,
                          input_shape=(20,hidden_dim), 
                          name = 'LSTM1')
        self.Dropout1 = Dropout(dropout)
        #self.LSTM2 = LSTM(units=10,
        #                  return_sequences=True,
        #                  kernel_initializer = 'he_uniform',
        #                  dropout = 0,
        #                  recurrent_dropout = 0,
        #                  name = 'LSTM2')
        #self.Dropout2 = Dropout(dropout)
        self.Dense1 = Dense(units=dimensionality, 
                            activation='sigmoid', 
                            name = 'Dense1')
        self.graph_has_been_written=False
        self.tensorboard_folder_path = tensorboard_folder_path
        
    def call(self, x, training=True, **kwargs): # Implement training = False when testing
        x = self.LSTM1(x)
        x = self.Dropout1(x, training)
        #x = self.LSTM2(x)
        #x = self.Dropout2(x, training)
        x = self.Dense1(x)  
        return x