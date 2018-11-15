
# coding: utf-8

# In[177]:


import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import chi2
from scipy import constants

#Graphing variables
pagewidth = (8.5, 5)
colours = ['#ff6868', '#fffc68', '#68ff9d', '#70c3ff', '#db70ff', '#f4428c', '#42f4dc']


# In[142]:


#Function definitions

''' Plots the distributions of data as a histogram against the gaussian distribution with mean and stddev matching
    that of the data. 
'''
def plot_distribution(data, filename, title, figsize=pagewidth, bins='sturges', hrange=(-100, 100), 
                      directory='graphs/distributions/', logmode=False):
    plt.figure(figsize=figsize)
    if (hrange == (-100, 100)): hrange = (min(data), max(data))
    gspace = np.linspace(hrange[0], hrange[1], 1000)
    gnorm = norm.pdf(gspace, loc=np.mean(data), scale=np.std(data))
    if type(bins) is str: bins = len(np.histogram_bin_edges(data, bins=bins)) - 1
    gnorm *= 1000 * len(data) / (sum(gnorm) * bins)
    plt.plot(gspace, gnorm, color=colours[4], label='Normal Distribution')
    plt.hist(data, bins=bins, range=hrange, facecolor=colours[3], alpha=0.8, label='ADC Distribution')
    plt.xlabel('ADC Reading (bits)')
    plt.ylabel('Number of Samples')
    if logmode: plt.yscale('log')
    plt.title('Distribution of Data, with frequency '+title+', # bins='+str(bins))
    plt.legend()
    plt.savefig(directory+filename+'.png', bbox_inches='tight')

''' Sums n adjacent samples together in list data and returns a new list of size len(data)/n. 
    n should be a factor of len(data).
'''
def sum_adjacent(data, n):
    return np.sum(data.reshape(int(len(data)/n), -1), axis=1)

''' Plots chi2 distribution for a power data set. 
'''
def plot_chi2(power, filename, title, bins=200, samples=100000, directory='graphs/chi2/'):
    num = np.array([1, 2, 4, 10, 100])
    plt.figure(figsize=pagewidth)
    plist, clist = [], []
    space = np.linspace(1,10**6,samples)
    mean, var = np.mean(power), np.std(power)
    for i in range(len(num)): 
        plist.append(sum_adjacent(power, num[i]))
        cnorm = len(plist[i]) * (max(plist[i]) - min(plist[i])) / bins
        clist.append(cnorm * chi2.pdf(space, num[i], loc=0, scale=var**0.96))
    for i in range(len(plist)): 
        plt.hist(plist[i], bins=bins, color=colours[i], alpha=0.3)
        plt.plot(space, clist[i], color=colours[i], label='N='+str(num[i]))
    plt.xlabel('Power Summed over N Samples (bits$^2$)')
    plt.ylabel('Number of Samples')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((7*10**1, 4*10**5))
    plt.ylim((10**2, 10**8))
    plt.title('ADC Power Data vs $\chi^2$ Distributions, frequency='+title)
    plt.legend()
    plt.savefig(directory+filename+'.png', bbox_inches='tight')
    
''' Plots the temperatures. 
'''
def plot_temp(temp, filename, title, directory='graphs/temperatures/'):
    plt.figure(figsize=pagewidth)
    plt.scatter(np.linspace(0,20,len(temp)), temp, color=colours[4], s=1.5**2)
    plt.xlabel('Time(s)')
    plt.ylabel('System Temperature (bits$^2$)')
    plt.yscale('log')
    plt.title('System Temperature Estimate vs Time, frequency='+title)
    plt.savefig(directory+filename+'.png', bbox_inches='tight')

''' Prints the basic statistics of the temperature data set. 
'''
def print_stats(temp, title):
    cadence = 1/(5*10**6)
    time = 20
    mean, std = np.mean(temp), np.std(temp)
    print(title,'stats')
    print('mean',mean)
    print('std',std)
    print('time',time)
    print('bandwidth',1/(500*10**12))
    print('ratio',1/np.sqrt(time/(2*cadence)))
    print('ratio observed',std/mean)
    print()

''' Similar to sum_adjacent, but squares, then means. 
'''
def mean_square_adjacent(data, n):
    return np.mean(data.reshape(int(len(data)/n), -1)**2, axis=1)

''' Finds the closest power of 2 greater than n.
'''
def find_power2(n):
    i = 0
    while (2**i < n): i += 1
    return 2**(i-1)

''' Calculates and plots the spectrum of a data set using provided parameters. 
'''
def plot_spectrum(data, filename, title, directory='graphs/spectrums/', plot_error=False, u=None, averages=None):
    if not averages: transform = np.fft.fft(data[int(len(data) - find_power2(len(data))):].reshape(-1,1024), axis=1)
    else:
        #avg_data = sum_adjacent(data, averages)/averages
        avg_data = data.reshape(int(len(data)/averages),-1)
        transform = np.fft.fft(avg_data, axis=1)
    t_power = np.mean((transform.real**2 + transform.imag**2), axis=0)
    print(len(t_power))
    plt.figure(figsize=pagewidth)
    half_len = int(len(t_power)/(2))
    if plot_error: plt.errorbar(np.arange(half_len)+100*10**6, 10*np.log10(t_power[:half_len]), yerr=u, 
                                elinewidth=1.0, capthick=1.0, capsize=2.0, fmt='.', color=colours[0])
    else: plt.scatter(np.arange(half_len)+100*10**6,10*np.log10(t_power[:half_len]), s=1**2, color=colours[1])
    plt.xlabel('Frequency Bin')
    plt.ylabel('Power (dB arb)')
    plt.title('Spectrum from AirSpy, frequency='+title)
    plt.savefig(directory+filename+'.png', bbox_inches='tight')


# In[143]:


get_ipython().run_cell_magic('time', '', "#CPU times: user 380 ms, sys: 442 ms, total: 822 ms Wall time: 11.4 s\n#CPU times: user 1.31 s, sys: 3.61 s, total: 4.91 sWall time: 40 s (just loading two sets)\n\n#Load data, subtract mean\n#Ended up being much faster this way then using an array\n\nroom_uhf = np.fromfile('newdata/room_UHF_100m.dat', dtype='int16')-2.**11 \nroom_uhf -= np.mean(room_uhf)\nroom_fm = np.fromfile('newdata/room_FM_100m.dat', dtype='int16')-2.**11 \nroom_fm -= np.mean(room_fm)\nroom_lte = np.fromfile('newdata/room_LTE_100m.dat', dtype='int16')-2.**11 \nroom_lte -= np.mean(room_lte)\nboiling = np.fromfile('newdata/boiling.dat', dtype='int16')-2.**11 \nboiling -= np.mean(boiling)\nice = np.fromfile('newdata/ice.dat', dtype='int16')-2.**11 \nice -= np.mean(ice)\ndry_ice = np.fromfile('newdata/dry_ice.dat', dtype='int16')-2.**11 \ndry_ice -= np.mean(dry_ice)\nliquid_nitrogen = np.fromfile('newdata/liquid_nitrogen.dat', dtype='int16')-2.**11 \nliquid_nitrogen -= np.mean(liquid_nitrogen)")


# In[144]:


#Title strings for graphing
filenames = ['room_UHF_100m', 'room_FM_100m', 'room_LTE_100m', 
             'boiling', #'boiling2', 'boiling3', 'boiling4',
             'ice', 'dry_ice', 'liquid_nitrogen']
titles = ['1GHz @21.9$^\circ$ C', '100MHz @21.9$^\circ$ C', '720MHz @21.9$^\circ$ C', 
          '1GHz @87.2$^\circ$ C', #'1GHz @77.1$^\circ$ C', '1GHz @61.8$^\circ$ C', '1GHz @56.6$^\circ$ C', 
          '1GHz @0.8$^\circ$ C', '1GHz @-78.5$^\circ$ C', '1GHz @-195.8$^\circ$ C']


# In[145]:


get_ipython().run_cell_magic('time', '', "#CPU times: user 19.6 s, sys: 615 ms, total: 20.2 s Wall time: 19.8 s\n\n#Distribution graphs\nplot_distribution(room_uhf[50000000:], filenames[0], titles[0], bins=100)\nplot_distribution(room_uhf[50000000:], filenames[0]+'log-y', titles[0], logmode=True, bins=100)\n\nplot_distribution(room_fm[50000000:], filenames[1], titles[1], bins=100)\nplot_distribution(room_fm[50000000:], filenames[1]+'log-y', titles[1], logmode=True, bins=100)\n\nplot_distribution(room_lte[50000000:], filenames[2], titles[2], bins=100)\nplot_distribution(room_lte[50000000:], filenames[2]+'log-y', titles[2], logmode=True, bins=100)\n\nplot_distribution(boiling[50000000:], filenames[3], titles[3], bins=100)\nplot_distribution(boiling[50000000:], filenames[3]+'log-y', titles[3], logmode=True, bins=100)\n\nplot_distribution(ice[50000000:], filenames[4], titles[4], bins=100)\nplot_distribution(ice[50000000:], filenames[4]+'log-y', titles[4], logmode=True, bins=100)\n\nplot_distribution(dry_ice[50000000:], filenames[5], titles[5], bins=100)\nplot_distribution(dry_ice[50000000:], filenames[5]+'log-y', titles[5], logmode=True, bins=100)\n\nplot_distribution(liquid_nitrogen[50000000:], filenames[6], titles[6], bins=100)\nplot_distribution(liquid_nitrogen[50000000:], filenames[6]+'log-y', titles[6], logmode=True, bins=100)")


# In[146]:


get_ipython().run_cell_magic('time', '', '#CPU times: user 286 ms, sys: 240 ms, total: 525 ms Wall time: 525 ms\n#CPU times: user 482 ms, sys: 1.83 s, total: 2.32 s Wall time: 14.2 s for both sets\n\n#Calculate power lists\nconstant = constants.epsilon_0 * constants.c / 2\npower_uhf = room_uhf**2\npower_fm = room_fm**2\npower_lte = room_lte**2\npower_b = boiling**2\npower_ice = ice**2\npower_di = dry_ice**2\npower_ln = liquid_nitrogen**2')


# In[153]:


get_ipython().run_cell_magic('time', '', '#CPU times: user 43.2 s, sys: 5.98 s, total: 49.2 s Wall time: 1min 12s\n\n#CPU times: user 39.5 s, sys: 1.53 s, total: 41 s Wall time: 41.7 s\n#CPU times: user 1min 1s, sys: 5.79 s, total: 1min 7s Wall time: 1min 35s with norm\n#CPU times: user 59.1 s, sys: 2.91 s, total: 1min 1s Wall time: 1min 16s with multiplied by norm\n#CPU times: user 54.7 s, sys: 6.83 s, total: 1min 1s Wall time: 2min 25s\n#CPU times: user 1min 45s, sys: 8.3 s, total: 1min 54s Wall time: 2min 51s for two graphs')


# In[ ]:


plot_chi2(power_uhf, filenames[0], titles[0])
plot_chi2(power_fm, filenames[1], titles[1])
plot_chi2(power_lte, filenames[2], titles[2])
plot_chi2(power_b, filenames[3], titles[3])
plot_chi2(power_i, filenames[4], titles[4])
plot_chi2(power_di, filenames[5], titles[5])
plot_chi2(power_ln, filenames[6], titles[6])


# In[156]:


get_ipython().run_cell_magic('time', '', '#CPU times: user 2.34 s, sys: 196 ms, total: 2.54 s Wall time: 2.13 s')


# In[159]:


get_ipython().run_cell_magic('time', '', 'n = 1000\n\ntemp_uhf = mean_square_adjacent(room_uhf, n)')


# In[160]:


temp_fm = mean_square_adjacent(room_fm, n)


# In[161]:


temp_lte = mean_square_adjacent(room_lte, n)


# In[162]:


temp_b = mean_square_adjacent(boiling, n)


# In[164]:


temp_i = mean_square_adjacent(ice, n)


# In[166]:


temp_di = mean_square_adjacent(dry_ice, n)


# In[167]:


temp_ln = mean_square_adjacent(liquid_nitrogen, n)


# In[ ]:


plot_temp(temp_uhf, filenames[0], titles[0])
plot_temp(temp_fm, filenames[1], titles[1])
plot_temp(temp_lte, filenames[2], titles[2])
plot_temp(temp_b, filenames[3], titles[3])
plot_temp(temp_i, filenames[4], titles[4])
plot_temp(temp_di, filenames[5], titles[5])
plot_temp(temp_ln, filenames[6], titles[6])


# In[175]:


get_ipython().run_cell_magic('time', '', '\ntmean, tstd = [], []\ntmean.append(np.mean(temp_uhf))\ntmean.append(np.mean(temp_fm))\ntmean.append(np.mean(temp_lte))\ntmean.append(np.mean(temp_b))\ntmean.append(np.mean(temp_i))\ntmean.append(np.mean(temp_di))\ntmean.append(np.mean(temp_ln))\ntstd.append(np.std(temp_uhf))\ntstd.append(np.std(temp_fm))\ntstd.append(np.std(temp_lte))\ntstd.append(np.std(temp_b))\ntstd.append(np.std(temp_i))\ntstd.append(np.std(temp_di))\ntstd.append(np.std(temp_ln))\n\nprint_stats(temp_uhf, titles[0])\nprint_stats(temp_fm, titles[1])\nprint_stats(temp_lte, titles[2])\nprint_stats(temp_b, titles[3])\nprint_stats(temp_i, titles[4])\nprint_stats(temp_di, titles[5])\nprint_stats(temp_ln, titles[6])')


# In[180]:


get_ipython().run_cell_magic('time', '', "\nn = 100000\n\ntemp_uhf = mean_square_adjacent(room_uhf, n)\ntemp_fm = mean_square_adjacent(room_fm, n)\ntemp_lte = mean_square_adjacent(room_lte, n)\ntemp_b = mean_square_adjacent(boiling, n)\ntemp_i = mean_square_adjacent(ice, n)\ntemp_di = mean_square_adjacent(dry_ice, n)\ntemp_ln = mean_square_adjacent(liquid_nitrogen, n)\n\nplt.figure(figsize=pagewidth)\nplt.scatter(np.linspace(0,20,len(temp_uhf)), temp_uhf, color=colours[0], s=1.5**2, label=titles[0])\n#plt.scatter(np.linspace(0,20,len(temp_fm)), temp_fm, color=colours[1], s=1.5**2, label=titles[1])\n#plt.scatter(np.linspace(0,20,len(temp_lte)), temp_lte, color=colours[2], s=1.5**2, label=titles[2])\nplt.scatter(np.linspace(0,20,len(temp_b)), temp_b, color=colours[3], s=1.5**2, label=titles[3])\nplt.scatter(np.linspace(0,20,len(temp_i)), temp_i, color=colours[4], s=1.5**2, label=titles[4])\nplt.scatter(np.linspace(0,20,len(temp_di)), temp_di, color=colours[5], s=1.5**2, label=titles[5])\nplt.scatter(np.linspace(0,20,len(temp_ln)), temp_ln, color=colours[6], s=1.5**2, label=titles[6])\nplt.xlabel('Time(s)')\nplt.ylabel('System Temperature (bits$^2$)')\nplt.yscale('log')\nplt.title('System Temperature Estimate vs Time, combined and averaged over 100000')\nplt.legend()\nplt.savefig('graphs/temperatures/all.png', bbox_inches='tight')")


# In[183]:


get_ipython().run_cell_magic('time', '', '#CPU times: user 8.79 s, sys: 10 s, total: 18.8 s Wall time: 36.8 s\n\nmean = np.mean(temp_fm)\nratio = 1/np.sqrt(20*1/(2*cadence))\nnum_needed = len(room_fm)/(2/0.01**2)')


# In[186]:


get_ipython().run_cell_magic('time', '', '\nplot_spectrum(boiling, filenames[3], titles[3], plot_error=True, u=0.7)\nplot_spectrum(ice, filenames[4], titles[4], plot_error=True, u=0.7)\nplot_spectrum(dry_ice, filenames[5], titles[5], plot_error=True, u=0.7)\nplot_spectrum(liquid_nitrogen, filenames[6], titles[6], plot_error=True, u=0.7)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nplot_spectrum(room_uhf, filenames[0], titles[0], plot_error=True, u=0.7)\nplot_spectrum(room_fm, filenames[1], titles[1], plot_error=True, u=0.7)\nplot_spectrum(room_lte, filenames[2], titles[2], plot_error=True, u=0.7)\nplot_spectrum(boiling, filenames[3], titles[3], plot_error=True, u=0.7)\nplot_spectrum(ice, filenames[4], titles[4], plot_error=True, u=0.7)\nplot_spectrum(dry_ice, filenames[5], titles[5], plot_error=True, u=0.7)\nplot_spectrum(liquid_nitrogen, filenames[6], titles[6], plot_error=True, u=0.7)')


# In[ ]:


plot_spectrum(room_uhf, filenames[1], titles[1], plot_error=True, u=0.7)
plot_spectrum(room_fm, filenames[1]+'avg', titles[1]+' averaged over 20000', plot_error=True, u=0.8, averages=20000)
#plot_spectrum(room_fm, filenames[1], titles[1], plot_error=True, u=1.1)
#plot_spectrum(dry_ice, filenames[1], titles[1], plot_error=True, u=1.1)


# In[198]:


def linear(x, a, b):
    return a*x + b

def p0(x, y, n):
    return (n * np.sum(x * y) - (np.sum(x) * np.sum(y)))/(n * np.sum(x**2) - np.sum(x)**2)

def p1(x, y, n):
    return 1/n * (np.sum(y) - p0(x, y, n) * np.sum(x))

mean_uhf = np.mean(temp_uhf)
mean_b = np.mean(temp_b)
mean_i = np.mean(temp_i)
mean_di = np.mean(temp_di)
mean_ln = np.mean(temp_ln)


# In[204]:


values = np.array([(295.05, mean_uhf), (360.35, mean_b), (273.95, mean_i), (194.65, mean_di), (77.35, mean_ln)])

plt.figure(figsize=pagewidth)
space = np.linspace(0,360,200)
plt.plot(space, linear(space, p0(values[:,0], values[:,1], 5), p1(values[:,0], values[:,1], 5)), 
         color='orange', label='Best Fit')
plt.scatter(values[:,0], values[:,1], label='Measurements')
plt.xlabel('Load Temperatures (K)')
plt.ylabel('Power (bits$^2$)')
plt.title('Power vs Load Temp')
plt.ylim((0, 1400))
plt.legend()
plt.savefig('graphs/fit.png', bbox_inches='tight')


# In[205]:


print(p0(values[:,0], values[:,1], 5))
print(p1(values[:,0], values[:,1], 5))

