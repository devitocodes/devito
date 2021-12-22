import os                                                                       
import multiprocessing as mp
import numpy as np
import argparse

def run_python(process):                                                            
    os.system('python3 {}'.format(process)) 

parser = argparse.ArgumentParser(description='Setting the case and the model to be executed.')

parser.add_argument('--model',  type=str, 
                        help='Velocity model')

parser.add_argument('--case',  type=str, 
                        help='Case to be executed')
args  = parser.parse_args()
model = args.model
case  = args.case

# model = 'Marmousi' 
# model = 'Salt' 
# model = 'Circle' 
# model = 'HorizontalLayers' 

# case = 'fwd_reference_u'
# case = 'adjoint_reference'
# case = 'fwd_reference_rec'
# case = 'abc_test'
# case = 'fwi'
# case = 'true_rec'
if case=='fwi' or case=='true_rec':
    freq = [15]
else:
    freq    = [5, 10, 15, 20]

if model=='Marmousi':
    if case=='fwd_reference_u' or case=='fwd_reference_rec' or case=='bwd_reference':
        extension = [7500]
        methods   = ['damping']

    elif case=='bwd_test' or case=='fwd_test':
        extension = [900, 450, 200, 100] 
        methods   = ['damping' , 'habc-a1', 'Higdon', 'pml', 'cpml']
    
    elif case=='fwi' or case=='true_rec':
        extension = [400]
        methods   = ['damping']

elif model=='Salt':
    if case=='fwd_reference_u' or case=='fwd_reference_rec' or case=='bwd_reference':
        extension = [8400]
        methods   = ['damping']

    elif case=='bwd_test' or case=='fwd_test':
        extension = [1440, 720, 360, 180, 90]
        methods   = ['damping' , 'habc-a1', 'Higdon', 'pml', 'cpml']
    
    elif case=='fwi' or case=='true_rec':
        extension = [600]
        methods   = ['damping']

elif model=='Circle' or model=='HorizontalLayers':
    if case=='fwd_reference_u' or case=='fwd_reference_rec' or case=='bwd_reference':
        extension = [2000]
        methods   = ['damping']

    elif case=='bwd_test' or case=='fwd_test':
        extension = [180, 150, 120, 90, 50] 
        methods   = ['damping' , 'habc-a1', 'Higdon', 'pml', 'cpml']


for i in methods:
    for k in extension:
        processes = []
        for j in freq:

            if case=='bwd_reference' or case=='bwd_test':
                processes.append("test_adjoint.py --extension " 
                                + str(k) 
                                + " --freq " 
                                + str(j) 
                                + " --method " 
                                + str(i)
                                + " --model " 
                                + model
                                )     

            elif case=='fwd_reference_u' or case=='fwd_reference_rec' or case=='fwd_test':
                processes.append("test_forward.py --extension " 
                                + str(k) 
                                + " --freq " 
                                + str(j) 
                                + " --method " 
                                + str(i)
                                + " --model " 
                                + model
                                + " --case "
                                + case
                                )                   

            elif case=='true_rec': #either Marmousi ou 2D salt

                processes.append("forward.py --extension " 
                                + str(k) 
                                + " --freq " 
                                + str(j) 
                                + " --method " 
                                + str(i)
                                + " --model " 
                                + model
                                )  
            elif case=='fwi': #either Marmousi ou 2D salt
                processes.append("fwi.py --extension " 
                                + str(k) 
                                + " --freq " 
                                + str(j) 
                                + " --method " 
                                + str(i)
                                + " --model " 
                                + model
                                )  


            
        pool = mp.Pool(processes=len(freq))                                              
        pool.map(run_python, processes)  
        pool.close()
        pool.join()




