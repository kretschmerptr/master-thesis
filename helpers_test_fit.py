import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import kstwobign

def calc_interevent_times(res_event_times):
    '''Function to calculate the interevent times given a series of event times.'''
    interevent_times = []
    for idx in range(len(res_event_times)):
        res_event_times_idx = res_event_times[idx]
        res_event_times_idx_lag_1 = res_event_times_idx[:(len(res_event_times_idx)-1)]
        res_event_times_idx_lag_1 = np.append(np.zeros(1),res_event_times_idx_lag_1)
        
        interevent_times.append(res_event_times_idx - res_event_times_idx_lag_1)
    
    return(interevent_times)

def qq_plot(res_events, dist_type):
    '''Function for the QQ plot of Exp(1) distribution.'''
    #plt.figure()
    dim = len(res_events)
    for idx in range(dim):
        fig, ax = plt.subplots(figsize = (10,10))
        if dist_type == 'exp':
            stats.probplot(res_events[idx],dist = stats.expon,plot = plt)
        elif dist_type == 'uniform':
            stats.probplot(res_events[idx],dist = stats.uniform,plot = plt)
        ax.get_lines()[0].set_color('#1f77b4')
        ax.get_lines()[1].set_color('k')
        plt.title(idx)
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Empirical Quantiles')
    
def res_counting_process(res_event_times, alpha):
    #plt.figure()
    dim = len(res_event_times)
    for idx in range(dim):
        T = np.max(res_event_times[idx])
        y_grid = np.linspace(start = 1 / len(res_event_times[idx]), stop = 1,
                             num = len(res_event_times[idx]))
        
        x_grid = np.linspace(start = 0, stop = 1, num = 50)
        #quantile = norm.ppf(1 - alpha / 2)
        #up_conf_line = x_grid + quantile / (np.sqrt(T))
        #low_conf_line = x_grid - quantile / (np.sqrt(T))
        quantile = kstwobign.ppf(1- alpha)
        up_conf_line = x_grid + quantile / (np.sqrt(len(res_event_times[idx])))
        low_conf_line = x_grid - quantile / (np.sqrt(len(res_event_times[idx])))
        
        plt.figure(figsize = (10,10))
        plt.step(res_event_times[idx] / T, y_grid, where = 'post', color = '#1f77b4')
        plt.plot(x_grid, x_grid, 'k')
        plt.plot(x_grid, up_conf_line, 'k--')
        plt.plot(x_grid, low_conf_line, 'k--')
        plt.ylim(0,1)
        plt.title(idx)

def acf(res_interevent_times, alpha):
    #plt.figure()
    dim = len(res_interevent_times)
    for idx in range(dim):
        fig, ax = plt.subplots(figsize = (10,10))
        plot_acf(res_interevent_times[idx],lags = 20, alpha = alpha, 
                 title = idx ,ax = ax)
        plt.xlabel('Lags')
        plt.ylabel('Autocorrelation Function')
        
        
def uniform_mark(res_event_marks):
    plt.figure()
    dim = len(res_event_marks)
    for idx in range(dim):
        plt.subplot(1,dim,idx+1)
        plt.scatter(res_event_marks[idx],
                    np.random.uniform(low=0,high=1,size=len(res_event_marks[idx])),
                    color = '#1f77b4')
        plt.title('Event marks of residual process on x-axis with random y-axis')
        
def ks_test(events_list, dist_type):
    '''Function to calculate the p-value of the Kolmogorov-Smirnov Test for Exp(1)-distr.'''
    p_values = np.zeros(len(events_list))
    for idx in range(len(events_list)):
        ks_test = stats.kstest(events_list[idx], dist_type)
        p_values[idx] = ks_test[1]
        
    return(p_values)
    
    
if __name__ == '__main__':
    res_interevent_times = calc_interevent_times(res_event_times)    
    exp1_qq_plot(res_interevent_times)
    
    alpha = 0.01
    res_counting_process(res_event_times, alpha)
    
    ks_test(res_event_times, 'expon')
    ks_test(res_event_marks, 'uniform')
    
    
    
    
    
    
    
    
    
