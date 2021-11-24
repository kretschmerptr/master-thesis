#Import packages and modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from helpers import *
from helpers_real_data import *
from CTLSTM_Class import *

# Setting for real data
filename = 'nasdaq100'  # 'sp500', 'nasdaq100'
select_colname = 'NDX'   # 'Adj Close', 'NDX'
add_colname = ['log_return', 'date_index']
train_size = 0.9
quantiles = [2.5, 97.5]
mark_type = 'cont' # 'cont', 'discrete'
seq_length = 50
seed = 42
epochs = 3
num_features = 2
batch_size = 16
hidden_size = 64
num_features = 2
reg_coeff = 0
mark_distr = 'exp' # 'exp', 'void'
learning_rate = 1e-2
alpha = 0.05

# Load and preprocess the data
data = data_loader(filename, select_colname)
data.prepare_for_models(add_colname,quantiles, mark_type, train_size, seq_length, seed)
    
# Define the training and test set
tf_train = data.tf_seqs_train
tf_val = data.tf_seqs_val
tf_val_seq = data.tf_seq_val
    
# Get the training and test batches
train_batches = create_batches(tf_train, batch_size) 
val_batches = create_batches(tf_val, batch_size)

# Define the Continuous-time LSTM neural network and the optimizer to train the NN
model = CTLSTM(hidden_size = hidden_size, num_features = num_features,
               mark_distr = mark_distr, reg_coeff = reg_coeff) 
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Training:
for epoch in range(epochs):
   for (batch_index, batch) in enumerate(train_batches):

       # Apply model to the batch
        with tf.GradientTape() as tape:
            (output_gate, cell_state, cell_state_bar, decay_layer,
                     intensity) = model(batch)
            neg_log_likelihood = model.neg_log_likelihood(batch)
         
        # Calculate the gradient of the negative log-likelihood wrt the model parameters
        grads = tape.gradient(neg_log_likelihood, model.trainable_variables)
        # Apply the gradients and adjust the model parameters using the Adam optimizer
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Print loss
        if batch_index % 10 == 0:
          template = 'Ep. {}, Batch {}: log_ll_time {:.6f}, log_ll_mark {:.6f}'
          # print(template.format(epoch, batch_index, neg_log_likelihood))
          print(template.format(epoch, batch_index,
                                model.log_ll_time_event,
                                model.log_ll_mark_event))
        
        # Validate
        if (batch_index % 50 == 0 and batch_index > 0):
            if data_type == 'real':
                (output_gate, cell_state, cell_state_bar,
                     decay_layer, intensity) = model(tf.expand_dims(tf_val_seq, axis = 0))
                neg_log_likelihood = model.neg_log_likelihood(tf.expand_dims(tf_val_seq,
                                                                             axis = 0))
                
                val_template = 'Val.: Ep. {}, Batch {}: log_ll_time {:.6f}, log_ll_mark {:.6f}'
                print(val_template.format(epoch, batch_index, 
                                          model.log_ll_time_event,
                                          model.log_ll_mark_event))
                
                # Save the good models - Values set by experience
                #if model.log_ll_mark_event >= 7.43:
                if model.log_ll_time_event >= 1.47:
                    filename = 'nasdaq_setting_{}_reg_{}_epoch_{}_batch_{}_only_time'.format(hidden_size, reg_coeff,
                                               epoch, batch_index)
                    save_model_weights(model,filename)
            else:
                val_loss_list = []
                for (val_batch_index, val_batch) in enumerate(val_batches):
                    (output_gate, cell_state, cell_state_bar,
                         decay_layer, intensity) = model(val_batch)
                    neg_log_likelihood = model.neg_log_likelihood(val_batch)
                    val_loss_list.append(neg_log_likelihood)
                
                val_loss_event = tf.reduce_sum(tf.stack(val_loss_list)) / \
                    (len(val_loss_list) * batch_size * intensity.shape[1])
                    
                val_template = 'Epoch {}, Batch {}, Val. loss per event {}'
                print(val_template.format(0, batch_index, val_loss_event))
  

# Reinitalize model using saved model weights     
filename = 'nasdaq_setting_64_opt_2'
model.reinit(train_batches[0][0,:,:],filename)

# Test fit on real data
output_gate, cell_state, cell_state_bar, decay_layer, intensity = model(tf.expand_dims(tf_val_seq, axis = 0))
grid, intensity_grid, mark_grid = model.intensity_seq(tf_val_seq)
model.plot_CTLSTM_intensity(grid, intensity_grid)
model.plot_CTLSTM_mark(grid, mark_grid)
model.test_fit(tf_val_seq,alpha)
sim_event_seq = model.simulate_seq(100)
model.plot_sim_seq(sim_event_seq, T=100)


















