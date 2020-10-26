import tensorflow as tf
import numpy as np

def add_hist(train_vars, epoch):
    for i in train_vars:
        name = i.name.split(":")[0]
        value = i.value()
        tf.summary.histogram(name, value, step=epoch)
            
# Random vector generation Z
def RandomGenerator(batch_size, z_dim):
    Z_minibatch = list()
    
    for i in range(batch_size): 
        Z_minibatch.append(np.random.uniform(0., 1, [z_dim[0], z_dim[1]]))
        
    # Reshape the random matrices
    Z_minibatch = np.reshape(Z_minibatch, (batch_size, z_dim[0], z_dim[1]))
    return Z_minibatch
            

