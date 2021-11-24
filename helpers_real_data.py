'''Help functions to process real data'''
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class data_loader:
    def __init__(self,filename, select_colname):
        if filename == 'nasdaq100':
            filepath = 'data/nasdaq100/full/full_non_padding.csv'
        elif filename == 'sp500':
            filepath = 'data/sp500_1980_2019.csv'
        self.df = pd.read_csv(filepath)[select_colname]
        if isinstance(self.df, pd.Series):
            self.df = self.df.to_frame()
        self.select_colname = select_colname
        self.filename = filename
       
    def add_cols(self,add_colname):
        self.add_colname = add_colname
        if 'log_return' in add_colname:
            #if len(self.select_colname) == 1:
            self.df['log_return'] = np.log(self.df[self.select_colname]) - \
                                    np.log(self.df[self.select_colname].shift(1))
            '''else:
                df_log = np.log(self.df[self.select_colname]) - \
                            np.log(self.df[self.select_colname].shift(1))
                self.log_cols = ['log_return_' + string for string in self.select_colname]
                df_log.columns = self.log_cols
                self.df.reset_index(drop=True, inplace=True)
                df_log.reset_index(drop=True, inplace=True)
                self.df = pd.concat([self.df, df_log], axis=1)'''
                            
        if ('date_index' in add_colname) and (self.filename == 'nasdaq100'):
            self.df = self.df.dropna()
            self.df['date_index'] = np.arange(1,self.df.shape[0] + 1) / 389
        elif 'date_index' in add_colname:
            self.df['date_index'] = np.arange(1,self.df.shape[0] + 1)
            self.df = self.df.dropna()
    
        
    def thin_data(self, quantiles):
        '''Only consider data which is lower than the choosen small quantile and bigger
        than the larger choosen quantile.'''
        self.quantile_values = np.percentile(self.df['log_return'].values, quantiles)
        self.df_thin = self.df.copy()
        self.df_thin = self.df_thin[(self.df_thin['log_return'] <= self.quantile_values[0]) |
                        (self.df_thin['log_return'] >= self.quantile_values[1])]
        '''else:
            self.quantile_values = np.percentile(self.df[self.log_cols],
                                                                 quantiles, axis = 0)
            self.df_thin = self.df.copy()
            bool_mask = (self.df_thin[self.log_cols] <= self.quantile_values).any(axis = 1)
            self.df_thin = self.df_thin[bool_mask]'''
        
    def discrete_mark_format(self):
        '''Transform the log returns into a one hot encoding'''
        self.df_mark = self.df_thin.copy()
        self.df_mark = self.df_mark[self.add_colname]
        self.df_mark.loc[self.df_mark['log_return'] < 0, 'log_return'] = -1
        self.df_mark.loc[self.df_mark['log_return'] > 0, 'log_return'] = 1
        self.df_mark = pd.get_dummies(self.df_mark,
                                               columns=['log_return'])
        self.df_mark.rename(index=str,
            columns={"log_return_-1.0": "log_return_neg", "log_return_1.0": "log_return_pos"},
            inplace = True)
        
    def cont_mark_format(self):
        self.df_mark = self.df_thin.copy()
        self.df_mark = self.df_mark[self.add_colname]
        self.df_mark['log_return_neg'] = 0
        self.df_mark['log_return_pos'] = 0
        self.df_mark.loc[self.df_mark['log_return'] > 0,
                              'log_return_pos'] = self.df_mark['log_return'] - \
                                                     self.quantile_values[1]
        self.df_mark.loc[self.df_mark['log_return'] < 0,
                              'log_return_neg'] = np.abs(self.df_mark['log_return']) -\
                                                     np.abs(self.quantile_values[0])
        self.df_mark.drop('log_return', axis = 1, inplace = True)
        
    def mark_format(self, mark_type):
        if mark_type == 'discrete':
            self.discrete_mark_format()
        elif mark_type == 'cont':
            self.cont_mark_format()
    
    def train_val_split(self, train_size):
        num_train_events = int(self.df_mark.shape[0] * train_size)
        
        self.df_mark_train = self.df_mark[:num_train_events].copy()
        self.df_mark_val = self.df_mark[num_train_events:].copy()
        self.df_mark_val['date_index'] = self.df_mark_val['date_index'] - \
                                self.df_mark['date_index'].iloc[num_train_events - 1]
    
    def train_val_seq_for_CTLSTM(self, train_val_type):
        if train_val_type == 'train':
            np_data = self.df_mark_train.values
        elif train_val_type == 'val':
            np_data = self.df_mark_val.values
        # TODO: hard coded
        np_data = np.concatenate((np.zeros((1,3)), np_data))
        
        if train_val_type == 'train':
            self.tf_seq_train = tf.convert_to_tensor(np_data, tf.float32)
        elif train_val_type == 'val':
            self.tf_seq_val = tf.convert_to_tensor(np_data, tf.float32)
          
    def create_times_list(self, train_val_type):
        if train_val_type == 'train':
            df = self.df_mark_train
        elif train_val_type == 'val':
            df = self.df_mark_val
            
        np_neg_excess = df[df['log_return_neg'] != 0]['date_index'].values
        np_pos_excess = df[df['log_return_pos'] != 0]['date_index'].values
        
        if train_val_type == 'train':
            self.times_list_train = [np_neg_excess, np_pos_excess]
        elif train_val_type == 'val':
            self.times_list_val = [np_neg_excess, np_pos_excess]
        
    def create_marks_list(self, train_val_type):
        if train_val_type == 'train':
            df = self.df_mark_train
        elif train_val_type == 'val':
            df = self.df_mark_val
            
        np_neg_excess = df[df['log_return_neg'] != 0]['log_return_neg'].values
        np_pos_excess = df[df['log_return_pos'] != 0]['log_return_pos'].values
        
        if train_val_type == 'train':
            self.marks_list_train = [np_neg_excess, np_pos_excess]
        elif train_val_type == 'val':
            self.marks_list_val = [np_neg_excess, np_pos_excess]
        
    def create_seqs(self, seq_length, train_val_type):
        '''Split event seq into several subsequences of length seq_length.'''
        if train_val_type == 'train':
            df = self.df_mark_train
        elif train_val_type == 'val':
            df = self.df_mark_val
            
        n_rows_data = int(df.shape[0] - seq_length + 1)
        np_seqs = np.zeros((n_rows_data, seq_length + 1, 3))   # zero imputation
        for row_index in range(n_rows_data):
            if (row_index > 0):
                data_date_index = df['date_index'][row_index :(row_index + seq_length)] - \
                                    df['date_index'][(row_index - 1) : row_index].values
            else:
                data_date_index = df['date_index'][row_index :(row_index + seq_length)]
                
            np_seqs[row_index,1:,0] = data_date_index
            np_seqs[row_index,1:,1] = df['log_return_neg'][row_index :(row_index + seq_length)]
            np_seqs[row_index,1:,2] = df['log_return_pos'][row_index :(row_index + seq_length)]
            
        if train_val_type == 'train':
            self.np_seqs_train = np_seqs
        elif train_val_type == 'val':
            self.np_seqs_val = np_seqs
    
    
    def create_seqs_lists(self, train_val_type):
        if train_val_type == 'train':
            np_data = self.np_seqs_train
        elif train_val_type == 'val':
            np_data = self.np_seqs_val
            
        seqs_times_list = []
        seqs_marks_list = []
        for idx in range(np_data.shape[0]):
            seq = np_data[idx,:,:]
            times_neg_excess = seq[:,0][seq[:,1] != 0]
            times_pos_excess = seq[:,0][seq[:,2] != 0]
            marks_neg_excess = seq[:,1][seq[:,1] != 0]
            marks_pos_excess = seq[:,2][seq[:,2] != 0]
            
            seq_times = [times_neg_excess, times_pos_excess]
            seq_marks = [marks_neg_excess, marks_pos_excess]
            
            seqs_times_list.append(seq_times)
            seqs_marks_list.append(seq_marks)
        
        if train_val_type == 'train':
            self.seqs_times_list_train = seqs_times_list
            self.seqs_marks_list_train = seqs_marks_list
        elif train_val_type == 'val':
            self.seqs_times_list_val = seqs_times_list
            self.seqs_marks_list_val = seqs_marks_list
    
    def shuffle_rows(self, seed, train_val_type):
        '''Shuffle the np_ary along axis 0.'''
        if train_val_type == 'train':
            np.random.seed(seed)
            np.random.shuffle(self.np_seqs_train)
        elif train_val_type == 'val':
            np.random.seed(seed)
            np.random.shuffle(self.np_seqs_val)
        
    def np_to_tensor(self, train_val_type):
        if train_val_type == 'train':
            tf_seqs = tf.convert_to_tensor(self.np_seqs_train, tf.float32)
            self.tf_seqs_train = tf_seqs
        elif train_val_type == 'val':
            tf_seqs = tf.convert_to_tensor(self.np_seqs_val, tf.float32)
            self.tf_seqs_val = tf_seqs
        
    def prepare_for_models(self, add_colname, quantiles, mark_type, train_size, 
                           seq_length, seed):
        
        self.add_cols(add_colname)
        self.thin_data(quantiles)
        self.mark_format(mark_type)
        self.train_val_split(train_size)
        self.train_val_seq_for_CTLSTM('train')
        self.train_val_seq_for_CTLSTM('val')
        self.create_seqs(seq_length, 'train')
        self.create_seqs(seq_length, 'val')
        self.shuffle_rows(seed, 'train')
        self.shuffle_rows(seed, 'val')
        self.np_to_tensor('train')
        self.np_to_tensor('val')
        
        self.create_seqs_lists('train')
        self.create_seqs_lists('val')
        self.create_times_list('train')
        self.create_times_list('val')
        self.create_marks_list('train')
        self.create_marks_list('val')
        
        
    def plot_adj_close(self):
        '''Plot of price (time series)'''
        plt.rcParams.update({'figure.autolayout': True})
        fig, ax = plt.subplots(figsize = (12, 5))
        plt.plot(self.df['date_index'], self.df[self.select_colname])
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()
        
    def plot_log_return_all(self):
        plt.rcParams.update({'figure.autolayout': True})
        fig, ax = plt.subplots(figsize = (12, 5))
        markerline, stemlines, baseline = ax.stem(self.df['date_index'], 
                                                  self.df['log_return'],
                                                  markerfmt=' ')
        plt.setp(baseline, color = 'k')
        plt.xlabel('Time')
        plt.ylabel('Log-return')
        plt.show()
    
    def plot_log_return_all_thin(self, train_size):
        train_cut = train_size - (1 - train_size)
        date_index_train_cut = self.df_mark['date_index'].iloc[int(train_cut * self.df_mark.shape[0])]
        date_index_val_cut = self.df_mark['date_index'].iloc[int(train_size * self.df_mark.shape[0])]
        
        return_max = np.max([self.df_mark['log_return_neg'], 
                             self.df_mark['log_return_pos']])
        time_max = np.max(self.df_mark['date_index'])
        
        plt.rcParams.update({'figure.autolayout': True})
        fig, ax = plt.subplots(figsize = (12, 3.5))
        markerline, stemlines, baseline = ax.stem(self.df_mark['date_index'], 
                                                  self.df_mark['log_return_neg'],
                                                  markerfmt=' ')
        ax.axvline(date_index_train_cut, color = 'k', linestyle = ':')
        ax.axvline(date_index_val_cut, color = 'k', linestyle = ':')
        ax.set_ylim([0, return_max + 0.0002])
        ax.set_xlim([-2, time_max + 2])
        ax.text(35, .005, 'Train')
        ax.text(137.5, .005, 'Validate')
        ax.text(175, .005, 'Test')
        plt.setp(baseline, color = 'k')
        ax.set_xlabel('Time')
        ax.set_ylabel('Excess log-return')
        
        plt.rcParams.update({'figure.autolayout': True})
        fig, ax = plt.subplots(figsize = (12, 3.5))
        markerline, stemlines, baseline = ax.stem(self.df_mark['date_index'], 
                                                  self.df_mark['log_return_pos'],
                                                  markerfmt=' ')
        ax.axvline(date_index_train_cut, color = 'k', linestyle = ':')
        ax.axvline(date_index_val_cut, color = 'k', linestyle = ':')
        ax.set_ylim([0, return_max + 0.0002])
        ax.set_xlim([-2, time_max + 2])
        ax.text(35, .005, 'Train')
        ax.text(137.5, .005, 'Validate')
        ax.text(175, .005, 'Test')
        plt.setp(baseline, color = 'k')
        ax.set_xlabel('Time')
        ax.set_ylabel('Excess log-return')
        
        
        '''fig, axs = plt.subplots(2, 1,sharey = 'col')
        markerline, stemlines, baseline = axs[0].stem(self.df_mark['date_index'], 
                                                  self.df_mark['log_return_neg'],
                                                  markerfmt=' ')
        axs[0].axvline(date_index_train_cut, color = 'k', linestyle = ':')
        axs[0].axvline(date_index_val_cut, color = 'k', linestyle = ':')
        axs[0].text(35, .005, 'Train')
        axs[0].text(137.5, .005, 'Validate')
        axs[0].text(175, .005, 'Test')
        plt.setp(baseline, color = 'k')
        axs[0].set_xlabel('Time Index')
        axs[0].set_ylabel('Excess log-return')

        markerline, stemlines, baseline = axs[1].stem(self.df_mark['date_index'], 
                                                  self.df_mark['log_return_pos'],
                                                  markerfmt=' ')
        axs[1].axvline(date_index_train_cut, color = 'k', linestyle = ':')
        axs[1].axvline(date_index_val_cut, color = 'k', linestyle = ':')
        axs[1].text(35, .005, 'Train')
        axs[1].text(137.5, .005, 'Validate')
        axs[1].text(175, .005, 'Test')
        plt.setp(baseline, color = 'k')
        axs[1].set_xlabel('Time Index')
        axs[1].set_ylabel('Excess log-return')
        
        plt.show()'''
    
    def plot_log_return(self, train_val_type):
        '''Plot of log returns'''
        # Stem plot
        if train_val_type == 'train':
            df_plot = self.df_mark_train.copy()
        elif train_val_type == 'val':
            df_plot = self.df_mark_val.copy()
            
        df_plot['excess_log_return'] = df_plot['log_return_pos'] - \
                                        df_plot['log_return_neg']
            
        fig, ax = plt.subplots()
        markerline, stemlines, baseline = ax.stem(df_plot['date_index'], 
                                                  df_plot['excess_log_return'],
                                                  markerfmt=' ')
        plt.setp(baseline, color='k')
        plt.xlabel('Date Index')
        plt.ylabel('Excess log returns')
        plt.show()
        

if __name__ == '__main__':
    '''Nasdaq 100 - Negative and positive excess returns of one the Nasdaq Index'''
    filename = 'nasdaq100'
    select_colname = 'NDX'
    add_colname = ['log_return', 'date_index']
    train_size = 0.9
    quantiles = [2.5,97.5]
    mark_type = 'cont' # 'cont', 'discrete'
    seq_length = 50
    seed = 42
    data = data_loader(filename, select_colname)
    data.prepare_for_models(add_colname, quantiles, mark_type, train_size,
                            seq_length, seed)
    
    #event_times = data.times_list_train
    #event_marks = data.marks_list_train

    '''Nasdaq 100 - Negative excess returns of two Stocks'''
    filename = 'nasdaq100'
    select_colname = ['AAPL', 'MSFT']
    add_colname = ['log_return', 'date_index']
    quantiles = 5
    
    '''S&P 500 data'''
    filename = 'sp500'
    select_colname = 'Adj Close'
    add_colname = ['log_return', 'date_index']
    train_size = 0.9
    quantiles = [5,95]
    mark_type = 'cont' # 'cont', 'discrete'
    seq_length = 50
    seed = 42
    data = data_loader(filename, select_colname)
    data.prepare_for_models(add_colname, quantiles, mark_type, train_size,
                            seq_length, seed)

