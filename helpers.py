'''Help functions for the Continuous-Time LSTM'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_data(filename, num_features):
    '''Function to load data and return main data as tensorflow data.'''
    np_event_times = np.load('data/' + filename + '_' + 'event_times.npy',allow_pickle=True) 
    np_event_marks = np.load('data/' + filename + '_' + 'event_marks.npy',allow_pickle=True)
    np_intensity_times = np.load('data/' + filename + '_' + 'intensity_times.npy',allow_pickle=True)
    np_intensity_val = np.load('data/' + filename + '_' + 'intensity_val.npy',allow_pickle=True)
    
    np_event_times_pad, np_event_marks_pad = zero_pad_sequences(np_event_times,
                                                                np_event_marks,
                                                                num_features)
    
    seq_length = np_event_times_pad.shape[1]   # num_events + 1
    num_sequences = np_event_times_pad.shape[0]
    
    np_times_marks = np.empty((num_sequences, seq_length,1 + num_features))
    np_times_marks[:,:,0] = np_event_times_pad
    if num_features == 1:
        np_times_marks[:,:,1] = np_event_marks_pad
    else:
        np_times_marks[:,:,1:(num_features + 1)] = np_event_marks_pad
    
    # [batch_size, seq_length, 1+ num_features]
    data = tf.convert_to_tensor(np_times_marks, tf.float32)
    #data = tf.data.Dataset.from_tensor_slices(np_times_marks.astype('float32'))
    
    return(data, np_intensity_times, np_intensity_val)
    
    
def zero_pad_sequences(np_event_times, np_event_marks, num_features):
    '''Function to zero pad the event times and mark sequences by adding zero columns
    at column index 0.'''
    np_event_times_pad = np.concatenate((np.zeros((np_event_times.shape[0],1)),
                                         np_event_times),axis = 1)
    if num_features == 1:
        np_event_marks_pad = np.concatenate((np.zeros((np_event_marks.shape[0],
                                                       num_features)),np_event_marks),
                                                        axis = 1)
    else:
        np_event_marks_pad = np.concatenate((np.zeros((np_event_marks.shape[0],1,
                                                       num_features)),np_event_marks),
                                                        axis = 1)
        
    return(np_event_times_pad, np_event_marks_pad)

def train_val_test_split(data, train_size):
    '''Function to split the data into training and validation set.'''
    num_train_seqs = int(data.shape[0] * train_size)

    data_train = data[:num_train_seqs,:,:]
    data_val = data[num_train_seqs:,:,:]
    
    return(data_train, data_val, num_train_seqs)
    
def seqs_to_seq(tf_val):
    np_val = tf_val[0].numpy()
    for seqs_idx in range(1,tf_val.shape[0]):
        np_add_val = tf_val[seqs_idx][1:,:].numpy()
        np_add_val[:,0] = np_add_val[:,0] + np_val[-1,0]
        np_val = np.concatenate((np_val, np_add_val))
        
    tf_val_seq = tf.convert_to_tensor(np_val, tf.float32)
    return(tf_val_seq)
    
    
def create_batches(data, batch_size):
    '''Function to create a list of batches with exactly batch size of batch_size.'''
    batches = []
    i = 0
    while i <= data.shape[0] - batch_size :
        batches.append(data[i:(i + batch_size),:,:])
        i += batch_size
    return batches

def save_model_weights(model, filename):
    '''Function to save model weights in a file named filename.'''
    filepath = 'saved_models/' +  filename
    model_weights = model.get_weights()
    np.save(filepath, model_weights)
            
def plot_multivar_intensity(np_intensity_times, np_intensity_val, seq_index):
    '''Function to plot the intensity functions of a multivariate exponentially
    decaying Hawkes process.'''
    dim = len(np_intensity_val[seq_index])
    #plt.figure()
    for i in range(dim):
        plt.subplot(dim,1,i+1)
        plt.plot(np_intensity_times[seq_index], np_intensity_val[seq_index][i])
        

        
if __name__ == '__main__':
    filename = 'sim_hp_format_1'
    num_features = 1
    batch_size = 32
    
    data, np_intensity_times, np_intensity_val = load_data(filename, num_features)
    data_batches = create_batches(data, batch_size) 
    # batch_0 = data_batches[0]
