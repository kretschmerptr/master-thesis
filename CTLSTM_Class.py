import tensorflow as tf 
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

from helpers import *
from helpers_test_fit import *

class CTLSTM(keras.models.Model):
    def __init__(self, hidden_size, num_features, mark_distr, reg_coeff):
        super(CTLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.mark_distr = mark_distr
        self.reg_coeff = reg_coeff
        
        self.lin_rec_layer = keras.layers.Dense(7*self.hidden_size,
                                                input_shape = (self.num_features + self.hidden_size,))#,trainable = False)
        self.lin_int_layer = keras.layers.Dense(self.num_features,
                                                input_shape = (self.hidden_size,))#, trainable = False) #, use_bias = False)
        self.lin_mark_layer = keras.layers.Dense(self.num_features,
                                                input_shape = (self.hidden_size,)) #, use_bias = False)

    def init_states(self, batch_size):
        '''Function to initialize the hidden state (h(t_i)), the cell state decay (c(t_i)),
         and cell state bar (\bar{c}_t_i) for the initial time step t_0.'''
        hidden_state_t = tf.zeros((batch_size, self.hidden_size))
        cell_state_decay_t = tf.zeros((batch_size, self.hidden_size))
        cell_state_bar_t = tf.zeros((batch_size, self.hidden_size))
        
        return(hidden_state_t, cell_state_decay_t, cell_state_bar_t)
        
    def init_mark_layer_bias(self, batch):
        '''Function to manually initialize the bias of the lin_mark_layer.'''
        # Only needed when we model lambda and not 1 /lambda
        self.lin_mark_layer.build((self.hidden_size,))
        np_mark_layer_bias = np.zeros(self.num_features)
        for event_index in range(self.num_features):
            bool_mask = tf.not_equal(batch[:,1:,1 + event_index],0)
            marks_event_type = tf.boolean_mask(batch[:,1:, 1 + event_index], bool_mask)
            if self.mark_distr == 'exp':
                np_mark_layer_bias[event_index] = 1 / (tf.reduce_mean(marks_event_type))
        
        keras.backend.set_value(self.lin_mark_layer.weights[1], np_mark_layer_bias)
                
        
    def recurrent_step(self, event_marks_t,hidden_state_prev_t, cell_state_decay_prev_t,
                   cell_state_bar_prev_t):
        '''Function to implement the recurrent step as given by (5a) - (6c).'''
        # event_marks_t: [batch_size, num_features]
        # hidden_state_t: [batch_size,hidden_size]
        # concat_marks_hidden_t: [batch_size, num_features + hidden_size]
        # all in (), i.e. e.g.  input_gate [batch_size, hidden_size]
        # return: all [batch_size, hidden_size]
        
        # TODO: Softplus to scaled softplus
        concat_marks_hidden_t = tf.concat([event_marks_t, hidden_state_prev_t], axis=1)
        
        (input_gate, forget_gate, add_layer, output_gate, input_gate_bar, forget_gate_bar,
         decay_layer) = tf.split(self.lin_rec_layer(concat_marks_hidden_t),7,1)
        
        input_gate = tf.nn.sigmoid(input_gate)              # (5a)
        forget_gate = tf.nn.sigmoid(forget_gate)            # (5b)
        add_layer = tf.nn.tanh(add_layer)                   # (5c)
        output_gate = tf.nn.sigmoid(output_gate)            # (5d)
        input_gate_bar = tf.nn.sigmoid(input_gate_bar)      # (5e)
        forget_gate_bar = tf.nn.sigmoid(forget_gate_bar)    # (5f)
        decay_layer = tf.nn.softplus(decay_layer)           # (6c)
        
        cell_state_t = forget_gate * cell_state_decay_prev_t + input_gate * add_layer   # (6a)
        cell_state_bar_t = forget_gate_bar * cell_state_bar_prev_t + input_gate_bar * add_layer   # (6b)
        
        return(cell_state_t, cell_state_bar_t, decay_layer, output_gate)
    
    
    def decay_step(self, cell_state_t, cell_state_bar_t, decay_layer,
                   event_times_t_next_t):
        '''Function to implement the decay step as given by (7)'''
        # cell_state_decay_t: [batch_size, hidden_size]
        #       c(t_i) of the next time step
        
        interevent_times_t = event_times_t_next_t[:,1] - event_times_t_next_t[:,0]
        interevent_times_t = tf.expand_dims(interevent_times_t, axis = 1)
        
        cell_state_decay_t = cell_state_bar_t + (cell_state_t - cell_state_bar_t) * \
                                tf.exp(- decay_layer * interevent_times_t)
        
        return(cell_state_decay_t)
    
    def update_hidden_state(self, cell_state_decay_t, output_gate):
        '''Function to update the hidden state as given by (4b).'''
        # hidden_state_t : [batch_size, hidden_size]
        #       h(t_i) of the next time step
        hidden_state_t = output_gate * tf.nn.tanh(cell_state_decay_t)
        
        return(hidden_state_t)
        
    def update_intensity(self, hidden_state_t):
        '''Function to update the intensity function as given by (4a).'''
        # intensity_t : [batch_size, num_features]
        #       lambda(t_i) of the next time step
        
        intensity_t = tf.nn.softplus(self.lin_int_layer(hidden_state_t))
        
        return(intensity_t)
        
    def update_mark_params(self, hidden_state_t):
        '''Function to update the parameters of the mark distribution.'''
        mark_params_t = tf.nn.softplus(self.lin_mark_layer(hidden_state_t))
        
        return(mark_params_t)
        
    def mark_pdf_cdf(self, seq_idx, seq):
        '''Function to evaluate the mark pdf at the values given by seq.'''
        
        if self.mark_distr == 'exp':
            # Mark parameter = rate of exponential distribution, i.e. 1/lambda
            pdf = (1 / self.mark_params[seq_idx]) * \
                    tf.math.exp(- (1 / self.mark_params[seq_idx]) * seq)
            cdf = 1 - tf.math.exp(- (1 / self.mark_params[seq_idx]) * seq)
        
        '''if self.mark_distr == 'exp':
            pdf = self.mark_params[seq_idx] * \
                    tf.math.exp(-self.mark_params[seq_idx] * seq)
            cdf = 1 - tf.math.exp(- self.mark_params[seq_idx] * seq)'''
        return(pdf, cdf)
           
    def call(self, batch):
        '''Function to compute the forward step as given by (4) - (7).'''
        # batch : [batch_size, seq_length, num_features]
        batch_size = batch.shape[0]
        seq_length = batch.shape[1]
        
        output_gate_list = []
        cell_state_list = []
        cell_state_bar_list = []
        decay_layer_list = []
        intensity_list = []
        
        if self.mark_distr != 'void':
            mark_params_list = []
        
        (hidden_state_t, cell_state_decay_t,
             cell_state_bar_t) = self.init_states(batch_size)
        
        # Only needed when we model lambda, and not 1 / lambda
        '''if not self.lin_mark_layer.weights:
            self.init_mark_layer_bias(batch)'''
            
        for t in range(0,seq_length):
            (cell_state_next_t, cell_state_bar_next_t, decay_layer_next_t,
             output_gate_next_t) = self.recurrent_step(batch[:,t,1:], hidden_state_t,
                        cell_state_decay_t, cell_state_bar_t)
            
            if t < seq_length -1 : 
                cell_state_decay_next_t = self.decay_step(cell_state_next_t, 
                                                     cell_state_bar_next_t,
                                                     decay_layer_next_t,
                                                     batch[:,t:(t+2),0])
                
                hidden_state_next_t = self.update_hidden_state(cell_state_decay_next_t,
                                                          output_gate_next_t)
                intensity_next_t = self.update_intensity(hidden_state_next_t)
                intensity_list.append(intensity_next_t)
                if self.mark_distr != 'void':
                    mark_params_next_t = self.update_mark_params(hidden_state_next_t)
                    mark_params_list.append(mark_params_next_t)
                
                
            cell_state_bar_list.append(cell_state_bar_next_t)
            cell_state_list.append(cell_state_next_t)
            decay_layer_list.append(decay_layer_next_t)
            output_gate_list.append(output_gate_next_t)
            
            hidden_state_t = hidden_state_next_t
            cell_state_decay_t = cell_state_decay_next_t
            cell_state_bar_t = cell_state_bar_next_t
            
        # hidden_state: [batch_size,seq_length - 1, hidden_size]
        #   contains [h(t_1),...,h(t_n)]
        # intensity: [batch_size, seq_length - 1, num_features]
        #   contains [lambda(t_1),....,lambda(t_n)]
        # all the others have [batch_size, seq_length, hidden_size]
        #   as they contain the corresponding values for index 1,...,n+1
        #   n+1 as theses can be calculated using (t_n, k_n)
        
        self.intensity = tf.transpose(tf.stack(intensity_list),[1,0,2])  
        self.cell_state_bar = tf.transpose(tf.stack(cell_state_bar_list),[1,0,2])  
        self.cell_state = tf.transpose(tf.stack(cell_state_list),[1,0,2])
        self.decay_layer = tf.transpose(tf.stack(decay_layer_list), [1,0,2])
        self.output_gate = tf.transpose(tf.stack(output_gate_list),[1,0,2])
        if self.mark_distr != 'void':
            self.mark_params = tf.transpose(tf.stack(mark_params_list),[1,0,2])
        
        # TODO: brauch ich den return?
        return(self.output_gate, self.cell_state, self.cell_state_bar, self.decay_layer, self.intensity)

    def intensity_one_t(self, t, seq, cell_state_seq, cell_state_bar_seq,
                        decay_layer_seq, output_gate_seq):
        '''Function to evaluate the insitensity function at one time point given by t.'''
        # seq: [seq_length, 1(time) + num_features]
        # cell_state_seq: [seq_length, hidden_size]
        
        # seq[index_prev_element,0] < t <= seq[index_prev_element + 1,0]
        index_prev_element =  tf.reduce_max(tf.where(t > seq[:,0]))
        
        prev_event_time_curr_time = tf.expand_dims(tf.stack([seq[index_prev_element,0],
                                                            t],axis = 0), axis = 0)
        
        cell_state_decay_t = self.decay_step(cell_state_seq[index_prev_element,:],
                                        cell_state_bar_seq[index_prev_element,:],
                                        decay_layer_seq[index_prev_element,:],
                                        prev_event_time_curr_time)
        
        hidden_state_t = self.update_hidden_state(cell_state_decay_t,
                                             output_gate_seq[index_prev_element,:])
            
        intensity_t = self.update_intensity(hidden_state_t)
        
        return(intensity_t)

    def neg_log_likelihood(self, batch):
        '''Function to compute the log-likelihood of a batch as given by (8) using
        a Monte Carlo estimator for the integral.'''
        seq_length = batch.shape[1] - 1
        
        # Part 1
        sum_ll_list = []
        for seq_index in range(batch.shape[0]):
            bool_mask = tf.not_equal(batch[seq_index,1:,1:],0)
            sum_ll_list.append(tf.reduce_sum(tf.math.log(tf.boolean_mask(self.intensity[seq_index,:,:],
                                                         bool_mask))))
            
        log_ll = tf.reduce_sum(tf.convert_to_tensor(sum_ll_list))
        
        # Part 2
        mc_compensator_list = []
        for seq_index in range(batch.shape[0]):
            #seq = tf.expand_dims(batch[seq_index,:,:],axis = 0)
            seq = batch[seq_index,:,:]
            T = seq[-1,0]
            time_grid = tf.random.uniform((1,int(seq_length / 2)), minval = seq[0,0] + 1e-6, maxval = T)
            time_grid = tf.sort(time_grid)
            
            intensity_seq_list = []
            for t_index in range(time_grid.shape[1]):
                intensity_t = self.intensity_one_t(time_grid[0,t_index],
                                              seq,
                                              self.cell_state[seq_index,:,:],
                                              self.cell_state_bar[seq_index,:,:],
                                              self.decay_layer[seq_index,:,:],
                                              self.output_gate[seq_index,:,:])
                intensity_seq_list.append(intensity_t)
            
            mc_compensator_seq = tf.reduce_sum(tf.convert_to_tensor(intensity_seq_list)) * (T / int(seq_length / 2))
            mc_compensator_list.append(mc_compensator_seq)
     
        mc_compensator = tf.reduce_sum(tf.stack(mc_compensator_list))
        log_ll = tf.subtract(log_ll, mc_compensator)
        self.log_ll_time = log_ll
        self.log_ll_time_event = log_ll / (seq_length * batch.shape[0])
        
        # Part 3
        if self.mark_distr != 'void':
            sum_mark_list = []
            for seq_index in range(batch.shape[0]):
                bool_mask = tf.not_equal(batch[seq_index, 1:, 1:],0)
                pdf,_ = self.mark_pdf_cdf(seq_index, batch[seq_index, 1:, 1:])
                sum_mark_list.append(tf.reduce_sum(tf.math.log(tf.boolean_mask(pdf,
                                                                bool_mask))))
            
            log_ll_mark = tf.reduce_sum(tf.convert_to_tensor(sum_mark_list))
            self.log_ll_mark = log_ll_mark
            self.log_ll_mark_event = log_ll_mark / (seq_length * batch.shape[0])
            log_ll += log_ll_mark
            #log_ll = log_ll_mark
        
        loss = - log_ll
        #loss += self.reg_coeff * tf.nn.l2_loss(self.lin_rec_layer.kernel) + \
        #        self.reg_coeff * tf.nn.l2_loss(self.lin_int_layer.kernel)
        if self.mark_distr != 'void':
            #loss += self.reg_coeff * tf.nn.l2_loss(self.lin_mark_layer.kernel) TODO
            loss += self.reg_coeff * tf.nn.l2_loss(self.lin_rec_layer.kernel)
            
        return(loss)

    def residual_process_time(self, event_seq):
        '''Function to calculate the time residual_process.'''
        # TODO: Sicherstellen, dass alle Elemente is res_times_type_list mon. wachsend sind
        res_times_list = []
        for event_index in range(self.num_features):
            bool_mask = tf.not_equal(event_seq[:,(1 + event_index)],0)
            event_times_type = tf.boolean_mask(event_seq[:,0], bool_mask)
            end_time = tf.reduce_max(event_times_type)
            time_grid = tf.random.uniform((1,500 * len(event_times_type)),
                                          minval = 1e-6,
                                          maxval = end_time)
            time_grid = tf.sort(time_grid)
            intensity_seq_list = []
            for t_index in range(time_grid.shape[1]):
                intensity_t = self.intensity_one_t(time_grid[0,t_index],
                                                  event_seq,
                                                  self.cell_state[0,:,:],
                                                  self.cell_state_bar[0,:,:],
                                                  self.decay_layer[0,:,:],
                                                  self.output_gate[0,:,:])
                intensity_seq_list.append(intensity_t[0,event_index])
            intensity_seq = tf.stack(intensity_seq_list)
            
            res_times_type_list = []
            for time in event_times_type:
                prev_sim_times = time_grid[(time_grid <= time)].shape[0]
                res_time = tf.reduce_sum(intensity_seq[:prev_sim_times]) * (time / prev_sim_times)
                res_times_type_list.append(res_time)
            
            res_times_list.append(tf.stack(res_times_type_list))
            
        return(res_times_list)
        
    def residual_process_mark(self, event_seq):
        _, cdf = self.mark_pdf_cdf(0,event_seq[1:,1:])
        bool_mask = tf.not_equal(event_seq[1:, 1:],0)
        res_event_marks = []
        for event_type in range(self.num_features):
            res_event_marks.append(tf.boolean_mask(cdf[:,event_type],
                                                   bool_mask[:,event_type]))
        
        return(res_event_marks)
        
    def test_fit(self, event_seq, alpha):
        (output_gate, cell_state, cell_state_bar, decay_layer,
             intensity) = self(tf.expand_dims(event_seq, axis = 0))
        neg_log_likelihood = self.neg_log_likelihood(tf.expand_dims(event_seq,
                                                                     axis = 0))
                
        #Time
        res_times_list = self.residual_process_time(event_seq)
        res_event_times = [res_times_list[idx].numpy() for idx in range(len(res_times_list))]
        res_interevent_times = calc_interevent_times(res_event_times)
        qq_plot(res_interevent_times, 'exp')
        
        res_counting_process(res_event_times, alpha)
        
        acf(res_interevent_times, alpha)
        
        p_value_exp = ks_test(res_interevent_times, 'expon')
        self.test_results_time = {'p value KS Test Exp' : p_value_exp}
        
        #Mark
        if self.mark_distr != 'void':
            res_marks_list = self.residual_process_mark(event_seq)
            res_event_marks = [res_marks_list[idx].numpy() for idx in range(len(res_marks_list))]
            
            #uniform_mark(res_event_marks)
            
            qq_plot(res_event_marks, 'uniform')

            #acf(res_event_marks, alpha)
            
            p_value_uni = ks_test(res_event_marks, 'uniform')
        
            self.test_results_mark = {'p value KS Test Uniform' : p_value_uni}
        

    def intensity_seq(self, seq):
        '''Function to calculate the conditional intensity function for one event 
        sequence.'''
        # seq: [seq_length, 1 (time) + num_features]
        # starts with lambda(t_0 +)
        # approx. dt = 0.01
        grid = tf.linspace(seq[0,0] + 1e-9, seq[-1,0] + 1e-9, 10000)
        concat_grid_event_times = tf.concat([grid, seq[1:,0]],axis=0)
        concat_grid_event_times, _ = tf.unique(tf.sort(concat_grid_event_times, axis=0))
        
        event_index = 0
        intensity_grid_event_times_list =  []
        if self.mark_distr != 'void':
            mark_grid_list = []
            
        for curr_time_index in range(len(concat_grid_event_times)):
            # do not do event_index += 1 if last element was already bigger than the last
            # event time t_n, there is no t_n+1
            if ((event_index < seq.shape[0] - 1) and 
                concat_grid_event_times[curr_time_index] > seq[(event_index + 1),0]):
                event_index += 1
            
    
            prev_event_time_curr_time = tf.expand_dims(tf.stack([seq[event_index,0],
                                                                 concat_grid_event_times[curr_time_index]],axis = 0),axis = 0)
            cell_state_decay_t = self.decay_step(self.cell_state[:,event_index,:],
                                            self.cell_state_bar[:,event_index,:],
                                            self.decay_layer[:,event_index,:],
                                            prev_event_time_curr_time)
            
            hidden_state_t = self.update_hidden_state(cell_state_decay_t,
                                                 self.output_gate[:,event_index,:])
            
            intensity_t = self.update_intensity(hidden_state_t)
            intensity_grid_event_times_list.append(intensity_t)
            
            if self.mark_distr != 'void':
                mark_params_t = self.update_mark_params(hidden_state_t)
                mark_grid_list.append(mark_params_t)
            
        intensity_grid_event_times = tf.squeeze(tf.convert_to_tensor(intensity_grid_event_times_list))
        
        if self.mark_distr == 'void':    
            return(concat_grid_event_times, intensity_grid_event_times)
        else:
            mark_grid = tf.squeeze(tf.convert_to_tensor(mark_grid_list))
            return(concat_grid_event_times, intensity_grid_event_times, mark_grid)
        
    def plot_CTLSTM_intensity(self, grid, intensity_grid):
        '''Function to plot the conditional intensity function of the CTLSTM model.'''
        if len(intensity_grid.shape) == 1:
            #plt.figure()
            plt.plot(grid, intensity_grid)
        else:
            dim = int(intensity_grid.shape[1])
            #plt.figure()
            for i in range(dim):
                #plt.subplot(dim,1,i + 1)
                plt.rcParams.update({'figure.autolayout': True})
                plt.figure(figsize = (12,3.5))
                plt.plot(grid, intensity_grid[:,i])
                plt.tight_layout()
                plt.xlabel('Time')
                plt.ylabel('Intensity')
                plt.title(i)


    def plot_CTLSTM_mark(self, grid, mark_grid):
        '''Function to plot the estimated Mark parameter.'''
        if len(mark_grid.shape) == 1:
            #plt.figure()
            plt.plot(grid, mark_grid)
        else:
            dim = int(mark_grid.shape[1])
            #plt.figure()
            for i in range(dim):
                #plt.subplot(dim,1,i + 1)
                plt.rcParams.update({'figure.autolayout': True})
                plt.figure(figsize = (12,3.5))
                plt.plot(grid, mark_grid[:,i])
                plt.tight_layout()
                plt.xlabel('Time')
                plt.title(i)
       
    def simulate_seq(self, T):
        '''Function to simulate a Neural Hawkes process until time T.
        
        Remark:
        ========
        Function has to be called first, else e.g. lin_int_layer.weights will be [].
        
        '''
        curr_event_time = tf.constant(0, tf.float32)
        curr_event = tf.zeros((1,1 + self.num_features),dtype = tf.float32)
        
        event_list = [curr_event]
        
        (hidden_state_t, cell_state_decay_t,
         cell_state_bar_t) = self.init_states(1)
        
        #while curr_event_time < T:
        while True:
            (cell_state_next_t, cell_state_bar_next_t, decay_layer_next_t,
                 output_gate_next_t) = self.recurrent_step(tf.expand_dims(curr_event[0,1:],axis=0),
                                                     hidden_state_t, cell_state_decay_t,
                                                     cell_state_bar_t)
            
            bound_cell_state=output_gate_next_t * tf.nn.tanh(cell_state_next_t)
            bound_cell_state_bar=output_gate_next_t * tf.nn.tanh(cell_state_bar_next_t)
            
            next_event_time_list = []
            for event_type in range(self.num_features):
                comp_bound_list = []
                for hidden_dim in range(self.hidden_size):
                    weight = self.lin_int_layer.weights[0][hidden_dim,event_type]
                    comp_bound = tf.reduce_max([weight * bound_cell_state[0,hidden_dim],
                                               weight * bound_cell_state_bar[0, hidden_dim]])
                    comp_bound_list.append(comp_bound)
                lambda_star = tf.nn.softplus(tf.reduce_sum(tf.stack(comp_bound_list)))
                
                curr_time = curr_event_time
                
                while True:
                    u = tf.random.uniform([],minval = 1e-10, maxval = 1)
                    interevent_time = tf.random.gamma([],alpha = 1, beta = lambda_star)
                    curr_time += interevent_time
                    lambda_curr_time = self.intensity_one_t(tf.squeeze(curr_time),
                                                      curr_event,
                                                      cell_state_next_t,
                                                      cell_state_bar_next_t,
                                                      decay_layer_next_t,
                                                      output_gate_next_t)
                    
                    if u*lambda_star <= lambda_curr_time[0, event_type]:
                        break
                    
                next_event_time_list.append(curr_time)
                
            next_event_time = tf.reduce_min(tf.convert_to_tensor(next_event_time_list))
            
            if next_event_time > T:
                break
            
            next_event_mark_index = tf.math.argmin(tf.convert_to_tensor(next_event_time_list))
            next_event_mark = np.zeros((1,self.num_features))
            
            time_decay = tf.expand_dims(tf.stack([curr_event_time, next_event_time],
                                                 axis = 0), axis = 0)
            
            cell_state_decay_next_t = self.decay_step(cell_state_next_t, 
                                                 cell_state_bar_next_t,
                                                 decay_layer_next_t,
                                                 time_decay)
                
            hidden_state_next_t = self.update_hidden_state(cell_state_decay_next_t,
                                                      output_gate_next_t)
            
            if self.mark_distr == 'void':
                next_event_mark[0,next_event_mark_index.numpy()] = 1
                next_event_mark = tf.convert_to_tensor(next_event_mark,tf.float32)
            elif self.mark_distr == 'exp':
                mark_params_next_t = self.update_mark_params(hidden_state_next_t)[0,next_event_mark_index]
                # TODO: Currently only exponential distribution supported
                next_event_mark[0, next_event_mark_index.numpy()] = tf.random.gamma([],alpha = 1, 
                                beta = 1 / mark_params_next_t)
                next_event_mark = tf.convert_to_tensor(next_event_mark,tf.float32)
                
                
            next_event = tf.concat([tf.constant(next_event_time,shape=(1,1),dtype=tf.float32),
                                    next_event_mark], axis = 1)
        
            event_list.append(next_event)
            
            hidden_state_t = hidden_state_next_t
            cell_state_decay_t = cell_state_decay_next_t
            cell_state_bar_t = cell_state_bar_next_t
            
            curr_event = next_event
            curr_event_time = next_event_time
        
        sim_event_seq = tf.squeeze(tf.stack(event_list))
        
        return(sim_event_seq)
        
    def plot_sim_seq(self,sim_event_seq, T):
        #fig, axs = plt.subplots(2, 1,sharey = 'col')
        return_max = np.max([sim_event_seq[:,1].numpy(), 
                             sim_event_seq[:,2].numpy()])
        
        for index in [1,2]:
            plt.rcParams.update({'figure.autolayout': True})
            fig, axs = plt.subplots(figsize = (12, 3.5))
            axs.stem(sim_event_seq[:,0].numpy(), sim_event_seq[:,index].numpy(),
                     markerfmt=' ')
            plt.axhline(y=0, color = 'k')
            axs.set_ylim([0, return_max + 0.0002])
            axs.set_xlim([-1, T + 1])
            plt.xlabel('Time')
            plt.ylabel('Excess log-return')
            plt.title(index - 1)
        
    def reinit(self, seq, filename):
        '''Function to reinitialize the model using pretrained weights.'''
        #seq = train_batches[0][0,:,:]
        (output_gate, cell_state, cell_state_bar, decay_layer,
             intensity) = self(tf.expand_dims(seq, axis = 0))
        filepath = 'saved_models/' + filename  + '.npy'
        saved_weights = np.load(filepath, allow_pickle = True)
        self.set_weights(saved_weights)
        

if __name__ == '__main__': 
    '''Univariate Setting'''
    hidden_size = 4
    num_features = 1
    filename = 'sim_hp_format_1'
    mark_distr = 'void'
    batch_size = 8
    
    model = CTLSTM(hidden_size = hidden_size, num_features = num_features,
                   mark_distr = mark_distr)

    data, np_intensity_times, np_intensity_val = load_data(filename, num_features)
    data_batches = create_batches(data, batch_size)     
    
    batch = data_batches[0]
    seq = batch[0,:,:]
    
    # Seq
    seq  = train_batches[0][1,:,:]
    output_gate, cell_state, cell_state_bar, decay_layer, intensity = model(tf.expand_dims(seq, axis = 0))
    grid, intensity_grid = model.intensity_seq(seq)
    model.plot_CTLSTM_intensity(grid, intensity_grid)
    plt.plot(np_intensity_times[0],np_intensity_val[0])
    neg_log_likelihood = model.neg_log_likelihood(tf.expand_dims(seq, axis = 0))    
    
    # Batch
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    with tf.GradientTape() as tape:
        output_gate, cell_state, cell_state_bar, decay_layer, intensity = model(batch)
        neg_log_likelihood = model.neg_log_likelihood(batch)
        
    grads = tape.gradient(neg_log_likelihood, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Simulate one event sequence
    sim_event_seq = model.simulate_seq(50)
    
    '''Bivariate Setting'''
    filename = 'sim_hp_format_3'
    hidden_size = 64
    num_features = 2
    batch_size = 16
    mark_distr = 'void'
    
    model = CTLSTM(hidden_size = hidden_size, num_features = num_features,
                   mark_distr = mark_distr)
    
    data, np_intensity_times, np_intensity_val = load_data(filename, num_features)
    data_batches = create_batches(data, batch_size)   
    
    batch = data_batches[0]
    seq = batch[0,:,:]
    
    # One sequence
    seq = train_batches[0][2,:,:]
    output_gate, cell_state, cell_state_bar, decay_layer, intensity = model(tf.expand_dims(seq, axis = 0))
    grid, intensity_grid, mark_grid = model.intensity_seq(seq)
    #plt.figure()
    model.plot_CTLSTM_intensity(grid, intensity_grid)
    model.plot_CTLSTM_mark(grid, mark_grid)
    #plt.figure()
    plot_multivar_intensity(np_intensity_times, np_intensity_val,2)
    
    # One batch with SGD step
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    with tf.GradientTape() as tape:
        output_gate, cell_state, cell_state_bar, decay_layer, intensity = model(batch)
        neg_log_likelihood = model.neg_log_likelihood(batch)
        
    grads = tape.gradient(neg_log_likelihood, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Simulate one sequence
    sim_event_seq = model.simulate_seq(T = 2)
    grid, intensity_grid = model.intensity_seq(sim_event_seq)
    model.plot_CTLSTM_intensity(grid, intensity_grid)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    