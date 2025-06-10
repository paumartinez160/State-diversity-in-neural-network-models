import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

le =81 # Number of iterations

parametre_soroll = np.linspace(0, 8, le, endpoint = True) # Noise parameter, from 0 to 8 in 81 steps

def simulacio(W, s, rng):

    N = 7 # Neurons per module
    Nt = N * 4 # Total number of neurons, 4 modules of 7 neurons each

    t_total = 100000 #ms, total time of simulation. It increases parabolically the time of simulation

    # Parameters for regular spiking neurons
    a = .02
    b = .2
    c = -65 # mV
    d = 8

    # Matrices to store the membrane potential and the recovery variable
    v = np.zeros((Nt, t_total))
    u = np.zeros((Nt, t_total))

    # Initial conditions
    v[:,0] = c*np.random.rand(Nt)
    u[:,0] = b*v[:,0]

    # Delay matrix, gaussian with mean 0.6 and standard deviation 0.1
    delay = .1*np.random.randn(Nt,Nt) + .6


    firings = [[] for k in range(Nt)]

    for t in range(1,t_total):
        
        I_ij = np.zeros((Nt,Nt))
        
        f = np.where( v[:,t-1] >= 30 )[0] # Neurons that have fired at the previous time step
    
        # As there is a certain delay in the synaptic current that goes from one neuron to another,
        # I don't consider the firings that occur at the same time. All neurons that fire at the same time
        # will be reset at the same time, and the next iteration will receive the impulse of those that have fired.
        
        # The neurons that have fired are reset and it is specified that they have fired when the voltage reaches 30 mV.
        v[f,t-1] = c
        v[f,t-2] = 30
        u[f,t-1] = u[f,t-1] + d

        # The firings are stored in a list of lists, firings, where each index corresponds to a neuron and contains the times at which it has fired.
        # This is used to calculate the synaptic current that each neuron receives from the others.
        for k in f:
            firings[k].append(t)
        
        # The synaptic current is calculated for each neuron, taking into account the delay and the firings of the other neurons.
        for i in range(Nt):
            tl = np.array(firings[i])# Neurons that have fired before the current time
            # If the number of firings is greater than 80, only the last 80 firings are considered to avoid overflow.
            if np.size(tl) > 80:
                tl = tl[-80:]

            if np.size(tl) == 0:
                pass
            else:
                for j in range(Nt):
                    I_ij[i,j] =  4*W[i,j]*np.sum(np.exp(-((t-delay[i,j])*np.ones(np.size(tl))-tl)/10)*np.heaviside((t-delay[i,j])*np.ones(np.size(tl))-tl, 1)) 
        I = np.sum(I_ij, axis = 0)
        
        # The noise is added to v, which is the membrane potential of the neurons and is a Gaussian noise with a standard deviation of 4 and mean 3.
        noise = s * rng.standard_normal(Nt) + 3
        
        v[:,t] = v[:,t-1] + 1*(.04 * v[:,t-1]**2 + 5.* v[:,t-1] + 140 - u[:,t-1]+ noise + I)
        u[:,t] = u[:,t-1] + (a * (b * v[:,t-1] - u[:,t-1]))


    # The number of firings per neuron is calculated, and the average firing rate is calculated.
    # The average firing rate is the number of firings divided by the total time in seconds.
    # If there are no firings, the average firing rate is set to 0.
    # The first 200 ms are discarded to avoid the transient state of the system.
    fir_rate = np.empty(Nt)
    r = np.zeros((Nt,Nt))
    desv = np.zeros(Nt)
    mitj = np.zeros(Nt)

    fires = (v == 30).astype(int)
    fires = fires[:,200:]

    for i in range(Nt):
        fir = np.where(fires[i,:] == 1)[0]
        if np.size(fir) == 0:
            mitj[i] = 0
        else:
            itera_per_fi = (t_total-200)/len(fir)
            mitj[i] = itera_per_fi
    
    # The average firing rate is calculated as the mean of the firing rates of all neurons.
    # If there are no firings or just one neuron fires, the average firing rate is set to 0.
    w = np.where(mitj != 0)[0]

    if np.size(w) >= 2:
        mitja = np.mean(mitj[w])
        sep = max(1,mitja//3*2) #ms
    else:
        mitja = 0
        sep = 1

    # Rescale the firing array to take into account the correlation between neurons. It is done with windowing.
    # The firing array is divided into sub-arrays of size sep, and the value of each sub-array is set to 1 if at least one neuron has fired in that sub-array, otherwise it is set to 0.

    n_arr = np.round(t_total/sep)
 
    while (np.size(fires,1))%(n_arr) != 0:
        fires = fires[:,1:]

    fires_scaled = np.zeros((0,Nt))

    sub_arrays = np.split(fires[:], n_arr, axis=1)

    for i in range(len(sub_arrays)):
        fires_scaled = np.vstack((fires_scaled, np.any(sub_arrays[i], axis=1)))

    fires_scaled = fires_scaled.astype(int).T

    for i in range(Nt):
        fir_rate[i] = 1/1000*np.size(np.where(fires_scaled[i] == 1))
        desv[i] = np.sqrt(np.sum((fires_scaled[i,:]-fir_rate[i])**2))
    
    # If the standard deviation is 0, the correlation is set to 0 to avoid division by zero.
    for i in range(Nt):
        if desv[i] == 0:
            r[i,:] = 0
        else:
            for j in range(Nt):
                if desv[j] == 0:
                    r[i,j] = 0
                else:
                    r[i,j] = np.sum(( fires_scaled[i,:] - fir_rate[i] ) * ( fires_scaled[j,:] - fir_rate[j] ))/(desv[i] * desv[j])






    B = 20

    hg, _ = np.histogram(r, bins = B, range = (-0.0001,1.0001)) # Cal imposar un límit superior un pèl mes gran per a que pugui detectar els extrems
    hg1,boxsize = np.histogram(r[:7, :7], bins=B, range=(-0.0001, 1.0001))
    hg2,_ = np.histogram(r[7:14, 7:14], bins=B, range=(-0.0001, 1.0001))
    hg3,_ = np.histogram(r[14:21, 14:21], bins=B, range=(-0.0001, 1.0001))
    hg4,_ = np.histogram(r[21:28, 21:28], bins=B, range=(-0.0001, 1.0001))

    hg = hg - hg1 -hg2 - hg3 - hg4 

    bin_edges = np.array([(1/(2*B) + 1/B*i) for i in range(B)])

    # Normalize the histogram. Number of elements is 28*28 - 7*7 - 7*7 - 7*7 - 7*7 = 28*21
    hg = hg/(28*21)
    hg = np.maximum(hg, 0)

    func_complexity = 1- B/(2*(B-1))*np.sum(np.abs(hg-1/B))
    func_complexity = np.maximum(func_complexity, 0)

    return func_complexity

def mi_funcion(iteracio):

    # Set the seed for reproducibility
    rng = np.random.default_rng(seed=os.getpid() + iteracio)
    
    s = parametre_soroll[int(iteracio)]


    N = 7 

    Wij = np.zeros((N,N,4)) # Ab
    Wji = np.zeros((N,N,4))
    Wd = np.ones((N,N))
    np.fill_diagonal(Wd,0)

    Wzero = np.zeros((N,N))

    # Randomly select connections for Wij and Wji
    # The connections are selected in such a way that each module has 4 connections to the other modules.
    # The connections are selected randomly, but the same connections are used for all iterations to ensure reproducibility.
    index_rand = np.array([[[5, 3], [1, 3], [4, 1], [5, 0]], [[2, 4], [0, 2], [5, 4], [3, 1]], [[3, 0], [1, 4], [1, 1], [0, 3]], [[3, 1], [4, 3], [4, 5], [0, 2]]])

    for k in range(4):
            Wij[ index_rand[0, k, 0], index_rand[0, k, 1], k ] = 1
            Wij[ index_rand[1, k, 0], index_rand[1, k, 1], k ] = 1
            Wij[ index_rand[2, k, 0], index_rand[2, k, 1], k ] = 1
            Wji[ index_rand[3, k, 0], index_rand[3, k, 1], k ] = 1

    W1 = np.hstack((    Wd,     Wij[:,:,0],      Wji[:,:,1],    Wzero ))
    W2 = np.hstack((   Wji[:,:,0],      Wd,    Wzero,    Wij[:,:,2]   ))
    W3 = np.hstack((   Wij[:,:,1],   Wzero,       Wd,    Wji[:,:,3]   ))
    W4 = np.hstack(( Wzero,     Wji[:,:,2],      Wij[:,:,3],    Wd    ))

    # Concatenate the matrices to form the complete weight matrix W
    # Each module is a 7x7 matrix, and the complete weight matrix is a 28x28 matrix.
    W = np.vstack((W1,
                    W2,
                    W3,
                    W4))



    # Run the simulation with the weight matrix W and the noise parameter s
    # The function simulacio returns the complexity of the system.

    func_complexity = simulacio(W,s, rng)
    print(s, func_complexity)

    # Return the noise parameter and the complexity
    return (s,func_complexity)



if __name__ == "__main__":
    num_nucleos = 15  # Number of cores to use for multiprocessing
    num_iteraciones = le
    print(f"{num_nucleos} cores are used from {multiprocessing.cpu_count()} available.")
    
    # Create a pool of processes to run the simulation in parallel
    with multiprocessing.Pool(processes=num_nucleos) as pool:
        resultados = pool.map(mi_funcion, range(num_iteraciones))

    if len(resultados) == 0:
        print("No results to save. 'archive.txt' will remain empty.")
    else:
        np.savetxt('archive.txt', resultados, fmt='%.6f', delimiter='\t', header='Noise\tComplexity', comments='')
        print("Results saved to 'archive.txt'.")
    