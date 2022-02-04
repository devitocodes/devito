#==============================================================================
# Python Imports
#==============================================================================
import numpy             as np
import argparse
#==============================================================================

#==============================================================================
class settings:
#==============================================================================

    #==============================================================================
    parser = argparse.ArgumentParser(description='Setup to run the simulations.')

    parser.add_argument('--extension',  type=int, 
                        help='Extended domain width')

    parser.add_argument('--freq',  type=float, 
                        help='Frequency peak')

    parser.add_argument('--method',  type=str, 
                        help='ABC method')

    parser.add_argument('--model',  type=str, 
                        help='Velocity model')

    args = parser.parse_args()
    #==============================================================================

    #==============================================================================
    model = {
        "vp": args.model   # Marmousi, 2D SEG/EAGE, Circle and Horizontal Layers 
            }
    #==============================================================================

    #==============================================================================
    # FWI Marmousi model
    #==============================================================================
    setting1 = {
        "x0": 0.,                           # x initial in metters
        "z0": 0.,                           # z initial in metters
        "lenx": 17000,                      # x-axis lenght (metters)
        "lenz": 3500,                       # z-axis lenght (metters)
        "lenpmlx": args.extension,          # abc lenght x direction 
        "lenpmlz": args.extension,          # abc lenght z direction 
        "nptx": 1701,                       # number of points in x-axis
        "nptz": 351,                        # number of points in z-axis
        "t0": 0.,                           # initial time
        "tn": 5000.,                        # final time milliseconds
        "CFL": 0.4,                         # cfl parameter
        "f0": args.freq/1000,               # frequency peak KHz
        "Abcs": args.method,                # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,              # shot position from the z0 (metters)
        "recposition_z": 2.25,              # Receiver position from the z0 (metters)
        "shotposition_x":4001.25,           # shot position from the x0 (metters)
        "recposition_x": 14995.0,           # Receiver position from the z0 (metters)
        "rec_n": 551,                       # Receiver number
        "shot_n":20,                        # Shot number
        "habcw": 2,                         # 1=linear , 2=nonlinear weight (used in habc-a1)
        "jump": 5,                          # Jump to save the wave equation solution to be used in adjoint-based gradient
        "shots_dist": 8400,                 # distance between the shots in metters
        "memory": 70.,                      # Restart DASK cluster when more than X% of memory is used
        "dask": True,                       # This variable change if you start the DASK cluster
        "multiscale":True,                  # Frequency multiscale: True or False
        "freq_bands": [5, 8, 10],           # frequence band
        }
    #==============================================================================

    #==============================================================================
    # Setting used to carry out the ABCs tests with Marmousi model
    #==============================================================================
    setting2 = {
       "x0": 7500,                          # x initial in metters
        "z0": 0.,                           # z initial in metters
        "lenx": 5000,                       # x-axis lenght (metters)
        "lenz": 3500,                       # z-axis lenght (metters)
        "lenpmlx": args.extension,          # abc lenght x direction 
        "lenpmlz": args.extension,          # abc lenght z direction 
        "nptx": 501,                        # number of points in x-axis
        "nptz": 351,                        # number of points in z-axis
        "t0": 0.,                           # initial time
        "tn": 100.,                         # final time milliseconds
        "CFL": 0.4,                         # cfl parameter
        "f0": args.freq/1000,               # frequency peak KHz
        "Abcs": args.method,                # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,              # shot position from the z0 (metters)
        "recposition_z": 2.25,              # Receiver position from the z0 (metters)
        "jump": 5,                          # Jump to save the wave equation solution to be used in adjoint-based gradient
        "rec_n": 551,                       # Receiver number
        "shot_n":20,                        # Shot number
        "habcw": 2,                         # 1=linear , 2=nonlinear weight (used in habc-a1)
        "shots_dist": 2400,                 # distance between the shots in metters
        "memory": 70.,                      # Restart DASK cluster when more than X% of memory is used
        "dask": False,                      # This variable change if you start the DASK cluster
        "multiscale":False,                 # Frequency multiscale: True or False
        }
    #==============================================================================

    #==============================================================================
    # Setup used in the ABCs test with Circle/Horizontal Layers vel. model
    #==============================================================================
    setting3 = {
        "x0": 0.,                           # x initial in metters
        "z0": 0.,                           # z initial in metters
        "lenpmlx": args.extension,          # abc lenght x direction
        "lenpmlz": args.extension,          # abc lenght z direction
        "nptx": 101,                        # number of points in x-axis
        "nptz": 101,                        # number of points in z-axis
        "lenx": 1000,                       # x-axis lenght (metters)
        "lenz": 1000,                       # z-axis lenght (metters)
        "t0": 0.,                           # initial time
        "tn": 100.0,                        # final time milliseconds
        "f0" : args.freq/1000,              # frequency peak kHz
        "Abcs": args.method,                # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":30,                # shot position from the z0 (metters)
        "recposition_z": 20,                # Receiver position from the z0 (metters)
        "rec_n": 200,                       # Receiver number
        "CFL": 0.5,                         # cfl parameter
        "jump": 5,                          # Jump to save the wave equation solution to be used in adjoint-based gradient
        "habcw": 2,                         # 1=linear , 2=nonlinear weight (used in habc-a1)
        "shots_dist":325,                   # distance between the shots in metters
        "dask": False,                      # This variable change if you start the DASK cluster
        "multiscale":False,                 # Frequency multiscale: True or False
        }
    #==============================================================================

    #==============================================================================
    # Setup used in the ABCs test with 2D SEG/EAGE Salt Layers vel. model
    #==============================================================================
    setting4 = {
        "x0": 2000.,                        # x initial in metters
        "z0": 0.,                           # z initial in metters
        "lenpmlx": args.extension,          # abc lenght x direction
        "lenpmlz": args.extension,          # abc lenght z direction
        "nptx": 561,                        # number of points in x-axis
        "nptz": 331,                        # number of points in z-axis
        "lenx": 8400,                       # x-axis lenght (metters)
        "lenz": 3300,                       # z-axis lenght (metters)
        "t0": 0.,                           # initial time
        "tn": 500,                          # final time milliseconds
        "f0": args.freq/1000,               # frequency peak KHz
        "Abcs": args.method,                # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "CFL": 0.5,                         # cfl parameter
        "shotposition_z":2.5,               # shot position from the z0 (metters)
        "recposition_z": 5.0,               # Receiver position from the z0 (metters)
        "rec_n": 551,                       # Receiver number
        "jump": 5,                          # Jump to save the wave equation solution to be used in adjoint-based gradient
        "habcw": 2,                         # 1=linear , 2=nonlinear weight (used in habc-a1)
        "shots_dist": 4040,                 # distance between the shots in metters
        "dask": False,                      # This variable change if you start the DASK cluster
        "multiscale":False,                 # Frequency multiscale: True or False
        }
    #==============================================================================

    #==============================================================================
    # 2D SEG/EAGE Marmousi model
    #==============================================================================
    setting5 = {
        "x0": 2000.,                        # x initial in metters
        "z0": 0.,                           # z initial in metters
        "lenpmlx": args.extension,          # abc lenght x direction
        "lenpmlz": args.extension,          # abc lenght z direction
        "nptx": 561,                        # number of points in x-axis
        "nptz": 331,                        # number of points in z-axis
        "lenx": 8400,                       # x-axis lenght (metters)
        "lenz": 3300,                       # z-axis lenght (metters)
        "t0": 0.,                           # initial time
        "tn": 5000,                         # final time milliseconds
        "f0": args.freq/1000,               # frequency peak KHz
        "Abcs": args.method,                # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "CFL": 0.5,                         # cfl parameter
        "shotposition_z":2.5,               # shot position from the z0 (metters)
        "recposition_z": 5.0,               # Receiver position from the z0 (metters)
        "rec_n": 551,                       # Receiver number
        "habcw": 2,                         # 1=linear , 2=nonlinear weight (used in habc-a1)
        "jump": 5,                          # Jump to save the wave equation solution to be used in adjoint-based gradient
        "dask": False,                      # This variable change if you start the DASK cluster   
        "multiscale":False,                 # Frequency multiscale: True or False
        "freq_bands": [5, 8, 10],           # frequence band
        "fwi_iteration": 75,
    }
    #==============================================================================