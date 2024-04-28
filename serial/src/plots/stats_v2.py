import numpy as np
import os
import matplotlib.pyplot as plt
from numba import njit
import scipy as sp

############################################

#              LJ parameters

############################################

# Parameters for LJ potential of Kr found in:
'''
Rutkai et Al. (2016). How well does the Lennard-Jones potential 
         represent the thermodynamic properties of noble gases?. 

Molecular Physics. 115. 1-18. 10.1080/00268976.2016.1246760. 
'''

eps = 162.58        # [K]
sigma = 3.6274      # [Angstrom]

mass = 83.798       # [amu]


#-----------------------FUNCTIONS-----------------------
@njit
def rdf(pos, L, N, dr):
    
    n = len(pos)                                # Number of particles
    nbins = int(L/(2*dr))                       # Number of bins
    hist = np.zeros(nbins)                      # Initialize the histogram
    
    for i in range(n):                          # Loop over all pairs of particles
        for j in range(i+1, n):
            rij = pos[i] - pos[j]               # Distance between particles i and j
            rij = rij - L * np.rint(rij/L)      # Apply periodic boundary conditions
            r = np.linalg.norm(rij)
            
            if r < L/2:                         # Increment the histogram
                k = int(r/dr)
                hist[k-1] += 2
    
    rho = N / (L**3)                            # Density
    for i in range(nbins):
        r = (i + 0.5) * dr                              # Compute the radial distance
        hist[i] /= 4 * np.pi * r**2 * dr * rho * N      # Normalize the histogram
    return hist

#@njit
def block_average(data, max_size=16):
    #--------------------BLOCK AVERAGING--------------------|

    #--------------------------------------------------------
    #  -INPUT-
    #   data: array-like, data to be averaged (energy, temperature, etc)
    #   Nblocks: int, number of blocks to divide the data into
    #
    #  -OUTPUT-
    #   means: array-like, mean of each block
    #   stdev2: array-like, standard deviation of each block
    #--------------------------------------------------------
    

    #if max_size == None:
    #    max_size = len(data)

    power = np.arange(int(np.log(max_size)/np.log(2)))               
    m_sizes = 2**power                                                # Array amb les mides dels blocs, de 1 a max_size

    #print('bin sizes',m_sizes)

    block_mean = np.zeros((len(m_sizes), int(data.shape[1])))         # Array per guardar mitjanes de cada bloc
    block_stdev = np.zeros((len(m_sizes), int(data.shape[1])))        # Array per guardar desviacions estandard de cada bloc

    for i in range(len(m_sizes)):
        size = m_sizes[i]
        #print('Size: ',size)
        Nblocks = int(len(data)/size)
        bins = np.array_split(data, Nblocks)                    # Dividim en 10 bins
        
        bin_mean = np.empty((len(bins), bins[0].shape[1]))      # Creem array per guardar mitjanes de CADA bin
        means = np.empty((len(bins), bins[0].shape[1]))         # Creem array per guardar mitjanes de CADA bin
        stdev2 = np.empty((len(bins), bins[0].shape[1]))        # Creem array per guardar desviacions estandard de CADA bin

        print('bins shape',bins[0].shape)

        print('means shape',means.shape)

        print('N blocks: ',Nblocks)

        for j in range(len(bins)):
            bin_mean[j] = np.sum(bins[j], axis=0) / size

        
        means = np.sum(bin_mean, axis=0) / Nblocks                          # Mitjana de totes les mitjanes de cada bloc
        
        squared_diff = [(bin_mean[i] - means) ** 2 for i in range(len(bin_mean))]
        # Calculate the variance of the list by summing the squared differences and dividing by the length of the list
        variance = sum(squared_diff) / (Nblocks-1)
        
        print('variance shape',variance.shape)
        print('variance',variance)

        

        #stdev2 = np.var(bin_mean, axis=0,ddof=1) #/ (Nblocks-1)                    # Desviació estandard de cada bloc
        '''        
        print('bin_mean shape',bin_mean.shape)
        print('bin_mean',bin_mean[0])
        print('means shape',means.shape)
        print('means',means)

        sum = np.zeros((len(bin_mean), bins[0].shape[1]))


        print('sum shape',sum.shape)

        


        for k in range(len(bin_mean)):
            sum[k] = (abs(bin_mean[k] - means))**2

        sum = np.sum(sum, axis=0)
        sum = sum / (Nblocks-1)

        print('sum',sum)

        exit()

        sum = sum()
        
        #stdev2 = np.sum((bin_mean - means)**2, axis=0) / (Nblocks-1)       # Desviació estandard de cada bloc
        '''
        #stdev2 = sum()      # Calculate the sample variance of each block mean

        index = int(np.where(m_sizes == size)[0][0])    # Index de la mida de bin corresponent
        #print(np.where(m_sizes == size))
        block_mean[index] = means                       # Guardem mitjanes de cada bloc amb la mida de bin corresponent
        block_stdev[index] = variance                     # Guardem desviacions estandard de cada bloc amb la mida de bin corresponent
        

    return block_mean, variance, m_sizes


def autocorr(x,a,b,tau):
    '''
    Autocorrelation function
    '''
    return a-b*np.exp(-x/tau)

def fitting(autocorr, m_size, stdev2):
    ''' Fits the autocorrelation function to an exponential decay '''
    parms, cov = sp.optimize.curve_fit(autocorr, m_size, stdev2)
    return parms, cov

def blockav_plot(m_size, blockav_mean, blockav_stdv):
    fig = plt.figure(figsize=(10, 5))

    plt.scatter(m_size, np.sqrt(blockav_mean), color='blue', label='Block averages')
    plt.errorbar(m_size, blockav_mean, yerr=np.sqrt(blockav_stdv), fmt='o', color='blue', ecolor='red', capsize=5)
    plt.xlabel('Block size')
    plt.ylabel('stdev2')
    plt.title('Block averages')
    plt.legend()
    plt.show()

    return


#-------------------------------------------------------------------------|
#                               MAIN PROGRAM                              |
#-------------------------------------------------------------------------|


os.chdir(os.path.dirname(os.path.abspath(__file__)))    # Set path to the directory where the script is located


#---------------------------Read data from files---------------------------|

energy = np.loadtxt('../../energy_verlet.dat', skiprows=4, usecols= (1,2,3,4), dtype=float)
temperature = np.loadtxt('../../Temperatures_verlet.dat', skiprows=1, dtype=float)
pressure = np.loadtxt('../../pressure_verlet.dat', skiprows=1, dtype=float)
momentum = energy[:, 3]
#print('e_shape ',energy.shape)
energy = energy[:,:3]
pos_fin = np.loadtxt('../../traj.xyz', skiprows=2, usecols = (1,2,3), dtype=float) 
N = 125

n_frames = int(len(pos_fin)/N)          # Number of frames
traj = np.array_split(pos_fin, n_frames)      # Split the array into frames


#------------------------------Convert units--------------------------------|

energy = energy*eps/1000                # Convert energy to kJ/mol
temperature = temperature*eps           # Convert Temperature to K
pressure = pressure*eps/(((sigma*1e-10)**3*6.022e23))/1e6   # Convert pressure to Pa


'''# convert time to ps
# WIP!!
#energy[:, 0] = energy[:, 0]*6.022e23*np.sqrt(mass/1000*sigma**2*1e-20/eps)
#temperature[:, 0] = temperature[:, 0]*6.022e23*np.sqrt(mass/1000*sigma**2*1e-20/eps)
#pressure[:, 0] = pressure[:, 0]*6.022e23*np.sqrt(mass/1000*sigma**2*1e-20/eps)
'''

#----------------------------------RDF plot-------------------------------|
L = 8.55
N = 125
dr = 0.04
rdf_hist = np.zeros(int(L/(2*dr)))                  
count=0
for i in range(N,int(len(pos_fin)),N):
    rdf_hist_sum = rdf(pos_fin[i-N:i], L, N, dr)
    rdf_hist += rdf_hist_sum
    count+=1

rdf_hist /= (int(len(pos_fin)/N)-1)

print('count=',count)
print((int(len(pos_fin)/N)-1))

r = np.linspace(0.5*dr, L/2-0.5*dr, len(rdf_hist))
print(len(rdf_hist))

plt.figure()
plt.plot(r, rdf_hist, label='Radial Distribution Function', color='mediumseagreen')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.legend()
plt.title('Radial Distribution Function')
plt.savefig('Radial_Distribution_function.png')
#plt.show()


#---------------------------Block averaging-------------------------------|
print('Energy block averaging...')
emeans, estdevs, m_sizes = block_average(energy, 16)
print('Potential Energy: ', emeans[-1,0], '±', np.sqrt(estdevs[-1,0]), 'kJ/mol')
print('Kinetic Energy: ', emeans[-1,1], '±', np.sqrt(estdevs[-1,1]), 'kJ/mol')
print('Total Energy: ', emeans[-1,2], '±', np.sqrt(estdevs[-1,2]), 'kJ/mol')



print('temp shape',temperature.shape)
print('Temperature block averaging...')
tempmeans, tempstdevs, m_sizes = block_average(temperature, 16)
print(tempstdevs)
print('Temperature: ', tempmeans[-1,1], '±', np.sqrt(tempstdevs[-1,1]), 'K')



print('Pressure block averaging...')
presmeans, presstdevs, m_sizes = block_average(pressure, 16)
print('pres shape', presmeans.shape)
print('Pressure: ', presmeans[-1,1], '±', np.sqrt(presstdevs[-1,1]), 'MPa')


#---------------------------Autocorrelation function-------------------------------|

params, cov = fitting(autocorr, m_sizes, estdevs[:, 1])

print('Fitted parameters: ', params)
print('Covariance matrix: ', cov)


x = np.arange(m_sizes[0], m_sizes[-1])

plt.figure()
plt.plot(x, autocorr(x, *params), label='Fitted autocorrelation', color='mediumseagreen')
plt.scatter(m_sizes, np.sqrt(estdevs[:, 1]), color='blue', label='Block averages')
plt.xlabel('Block size')
plt.ylabel('stdev2')
plt.title('Block averages')
plt.legend()
plt.show()

plt.show()