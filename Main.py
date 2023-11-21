import FunctionModule as fm

import pandas as pd
import numpy as np
import seaborn as sns
import os

from scipy.fft import fft
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

path_root = os.getcwd()
path_to_data = os.path.join(path_root, 'Dados_Real')

# Classification Database
err_type = ['DesalinhHoriz', 'Desbalanc1', 'Desbalanc2', 'Normal', 'RolamDesbal', 'Rolam']
accel = ['Accel Coupled H', 'Accel Uncoupled H', 'Accel Coupled V', 'Accel Uncoupled V']
rot_freq = ['10Hz', '20Hz', '30Hz', '40Hz']

### TRAIN AND TEST ONCE ###
if False:
    to_train = True

    conf_matrix = [[0 for _ in range(24)] for _ in range(24)]
    acc_matrix = [[0 for _ in range(4)] for _ in range(4)]

    [model, x_test, y_test] = fm.train_model(path_to_data=path_to_data, rot_freq=rot_freq[0:4], accel_train=accel[0], accel_test=accel[0], save_model=False, process_type=True)
    [test_acc, cm] = fm.test_model(model, x_test, y_test)

    print(test_acc)
    print(cm)
############################

### PLOT SPECTROGRAMS ###
if True:
    err_type = ['DesalinhHoriz', 'Desbalanc1', 'Desbalanc2', 'Normal', 'RolamDesbal', 'Rolam_']
    err_type = ['Rolam_']

    for machine_state in err_type:
        # Shouldn't call function with multiple rot_freq at the same time
        fm.plot_spectogram(path_to_data = path_to_data,rot_freq = rot_freq[3],accel_to_consider = accel[0:4],fail = machine_state, save = True)
##########################

# COMPOSE CONFUSION MATRIX
if False:
    for i in range(4):
        for j in range(4):
            [model, x_test, y_test] = fm.train_model(path_to_data=path_to_data, rot_freq=rot_freq[0:4], accel_train=accel[i], accel_test=accel[j], save_model=False, process_type=False)
            #[test_acc, cm] = fm.test_model(model_list[i], x_test_list[i], y_test_list[j])
            [test_acc, cm] = fm.test_model(model, x_test, y_test)
            print(cm)

            acc_matrix[i][j] = test_acc

            for k in range(6):
                for l in range(6):
                    conf_matrix[6*i+k][6*j+l] = cm[k][l]
                    #print(conf_matrix)

    print(acc_matrix)
    print(conf_matrix)

