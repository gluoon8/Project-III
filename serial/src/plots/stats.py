import numpy as np
import os
import matplotlib.pyplot as plt
from numba import jit
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


# Set path to the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Load data from files
energy = np.loadtxt('../../energy_verlet.dat', skiprows=4, dtype=float)
temperature = np.loadtxt('../../Temperatures_verlet.dat', skiprows=1, dtype=float)
pressure = np.loadtxt('../../pressure_verlet.dat', skiprows=1, dtype=float)
momentum = energy[:, 4]

# Convert energy to kJ/mol
energy = energy*eps/1000

# Convert Temperature to K
temperature = temperature*eps

# Convert pressure to Pa
pressure = pressure*eps/(((sigma*1e-10)**3*6.022e23))/1e6

# convert time to ps
# WIP!!
#energy[:, 0] = energy[:, 0]*6.022e23*np.sqrt(mass/1000*sigma**2*1e-20/eps)
#temperature[:, 0] = temperature[:, 0]*6.022e23*np.sqrt(mass/1000*sigma**2*1e-20/eps)
#pressure[:, 0] = pressure[:, 0]*6.022e23*np.sqrt(mass/1000*sigma**2*1e-20/eps)



#-------------------Read the trajectory file-------------------

filename='../../traj.xyz'

with open(filename, mode='r') as file:  # Open the xyz file 
        lines = file.readlines()        # Read the lines    
        n_atoms = int(lines[0])         # Number of atoms

lines = lines[2:]                        # Remove the first two lines

for i in range(len(lines)):
    lines[i] = lines[i].split()

lines = np.array(lines)                     # Transform the list into a numpy array
lines = lines[:,1:].astype(float)           # Remove the first column and convert to float
n_frames = int(len(lines)/n_atoms)          # Number of frames
traj = np.array_split(lines, n_frames)      # Split the array into frames

#-------------------------------------------------------------------------

L = 10.77
N = 125
dr = 0.1
nbins = int(L/(2*dr))

def rdf(pos, L, N, dr):
    # Number of particles
    n = len(pos)
    # Number of bins
    nbins = int(L/(2*dr))

    # Global histogram (initialize before first calculation)
    global hist
    if not hist.any():
        hist = np.zeros(nbins)

    # Loop over all pairs of particles
    for i in range(n):
        for j in range(i+1, n):
            # Compute the distance between particles i and j
            rij = pos[i] - pos[j]
            # Apply periodic boundary conditions
            rij = rij - L * np.rint(rij/L)
            r = np.linalg.norm(rij)
            # Increment the histogram
            if r < L/2:
                k = int(r/dr)
                hist[k-1] += 2

    # Normalize the histogram (after processing all trajectories)
    rho = n / (L**3)
    for i in range(nbins):
        r = (i + 0.5) * dr
        hist[i] /= 4 * np.pi * r**2 * dr * rho * N

    return hist

# Assuming "pos" is an array of particle trajectories

# Initialize global histogram
global hist
hist = np.zeros(nbins)

# Calculate RDF for each trajectory
for frame in traj:
    hist += rdf(frame, L, N, dr)

# Normalize final histogram
hist /= len(traj)

n = int(len(traj))

rho = n / (L**3)
for i in range(nbins):
    r = (i + 0.5) * dr
    hist[i] /= 4 * np.pi * r**2 * dr * rho * N

# The variable "hist" now contains the final RDF for the set of trajectories


# Plot the histogram
plt.figure()
r = np.linspace(0.5*dr, L/2-0.5*dr, len(hist))
plt.plot(r, hist, label='Histogram', color='mediumseagreen')
plt.xlabel('r')
plt.ylabel('Count')
plt.legend()
plt.title('Histogram')
plt.savefig('Histogram.png')
plt.show()

print(hist.shape)


def block_average(data, max_size=16):
    #--------------------BLOCK AVERAGING--------------------|

    #--------------------------------------------------------
    #  -INPUT-
    #   data: array-like, data to be averaged (energy, temperature, etc)
    #   Nblocks: int, number of blocks to divide the data into
    #
    #  -OUTPUT-
    #   means: array-like, mean of each block
    #   stdev: array-like, standard deviation of each block
    #--------------------------------------------------------
    

    if max_size == None:
        max_size = len(data)

    power = np.arange(int(np.log(max_size)/np.log(2)))               
    m_sizes = 2**power                                                # Array amb les mides dels blocs, de 1 a max_size

    print('bin sizes',m_sizes)

    

    block_mean = np.zeros((len(m_sizes), int(data.shape[1])))                               # Array per guardar mitjanes de cada bloc
    block_stdev = np.zeros((len(m_sizes), int(data.shape[1])))                              # Array per guardar desviacions estandard de cada bloc

    for size in m_sizes:
        print('Size: ',size)
        Nblocks = int(len(data)/size)
        bins = np.array_split(data, Nblocks)                      # Dividim en 10 bins
        bin_mean = np.empty((len(bins), bins[0].shape[1]))       # Creem array per guardar mitjanes de CADA bin

        for i in range(len(bins)):
            bin_mean[i] = np.mean(bins[i], axis=0)

        means = np.mean(bin_mean, axis=0)  
        stdev = np.var(bin_mean, axis=0)/(Nblocks-1)

        #print(block_mean.shape)
        #print(means.shape)
        #print(stdev.shape)

        index = int(np.where(m_sizes == size)[0][0])   # Index de la mida de bin corresponent
        print(np.where(m_sizes == size))
        print('Index: ',index)
        block_mean[index] = means                 # Guardem mitjanes de cada bloc amb la mida de bin corresponent
        block_stdev[index] = stdev                  # Guardem desviacions estandard de cada bloc amb la mida de bin corresponent



    # 

    return block_mean, block_stdev, m_sizes

emeans, estdevs, m_sizes = block_average(energy, 16384)



print(emeans)
print(estdevs)

print(emeans.shape)
print(estdevs.shape)

# Autocorrrelation time function 
def autocorr(x,a,b,tau):
    '''
    Autocorrelation function
    '''
    return a-b*np.exp(-x/tau)

def fitting(autocorr, m_size, stdev):
    ''' Fits the autocorrelation function to an exponential decay '''
    parms, cov = sp.optimize.curve_fit(autocorr, m_size, stdev)
    return parms, cov

def blockav_plot(m_size, blockav_mean, blockav_stdv):
    fig = plt.figure(figsize=(10, 5))

    plt.scatter(m_size, np.sqrt(blockav_mean), color='blue', label='Block averages')
    plt.errorbar(m_size, blockav_mean, yerr=np.sqrt(blockav_stdv), fmt='o', color='blue', ecolor='red', capsize=5)
    plt.xlabel('Block size')
    plt.ylabel('Stdev')
    plt.title('Block averages')
    plt.legend()
    plt.show()

    return

blockav_plot(np.arange(len(m_sizes)), emeans[:, 2], estdevs[:, 2])


# Display the block averages and standard deviations
#print('Block averages:')

#ener, stdevs = block_average(energy, 16)
#print('Potential Energy: ', ener[1], '±', stdevs[1], 'kJ/mol')
#print('Kinetic Energy: ', ener[2], '±', stdevs[2], 'kJ/mol')
#print('Total Energy: ', ener[3], '±', stdevs[3], 'kJ/mol')

#temp, stdev_temp = block_average(temperature, 16)
#print('Temperature: ', temp[1], '±', stdev_temp[1], 'K')

#pres, stdev_pres = block_average(pressure, 16)
#print('Pressure: ', pres[1], '±', stdev_pres[1], 'MPa')


params, cov = fitting(autocorr, m_sizes, estdevs[:, 3])

x = np.arange(m_sizes[0], m_sizes[-1])

plt.figure()
plt.plot(x, autocorr(x, *params), label='Fitted autocorrelation', color='mediumseagreen')
plt.scatter(m_sizes, np.sqrt(estdevs[:, 3]), color='blue', label='Block averages')
plt.xlabel('Block size')
plt.ylabel('Stdev')
plt.title('Block averages')
plt.legend()
plt.show()



print(params)
print(cov)
