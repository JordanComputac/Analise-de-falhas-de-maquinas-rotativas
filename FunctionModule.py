import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def list_files_in_directory(directory: str):
    file_dict = {}
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file_path)
            file_dict[file_name] = file_path
    return file_dict

def get_classification(string, classification_array):
    
    for classification in classification_array:
        if classification in string:
            return classification
    return None

def filter_files_by_word(file_dict, word):
    filtered_files = {filename: filepath for filename, filepath in file_dict.items() if word in filename}
    return filtered_files

def filter_files_by_words(file_dict, words_to_search):
    filtered_files = {filename: filepath for filename, filepath in file_dict.items() if any(word.lower() in filename.lower() for word in words_to_search)}
    return filtered_files

def split_dict(file_dict, ratio, random_split=False):
    if random_split:
        items = list(file_dict.items())
        random.shuffle(items)
        split_point = int(len(items) * ratio)
        items1 = items[:split_point]
        items2 = items[split_point:]
        dict1 = dict(items1)
        dict2 = dict(items2)
    else:
        total_items = len(file_dict)
        split_point = int(total_items * ratio)
        keys = list(file_dict.keys())
        keys1 = keys[:split_point]
        keys2 = keys[split_point:]
        dict1 = {key: file_dict[key] for key in keys1}
        dict2 = {key: file_dict[key] for key in keys2}

    return dict1, dict2

def process_data(file_dict:dict, accel_to_consider:list, process_type:bool = True):
    err_type = ['DesalinhHoriz', 'Desbalanc1', 'Desbalanc2', 'Normal', 'RolamDesbal', 'Rolam']
    rot_freq = ['10Hz', '20Hz', '30Hz', '40Hz']
    
    x_list = []
    y_list = []

    for filename, filepath in file_dict.items():
        #print('*********************************')
        #print(f"Processing file: {filename}")

        data = np.genfromtxt(filepath, delimiter='\t', skip_header=23)  # Adjust skip_header based on your LVM file format
        df = pd.DataFrame(data)
        df.columns = ['X_Value', 'Time', 'Trigger', 'Accel Coupled H', 'Accel Coupled V', 'Accel Uncoupled H', 'Accel Uncoupled V', 'Vel Coupled H', 'Vel Coupled V', 'Vel Uncoupled H', 'Vel Uncoupled V']

        colunas_acc = ['Accel Coupled H', 'Accel Coupled V', 'Accel Uncoupled H', 'Accel Uncoupled V']

        frequency = get_classification(filename, rot_freq) # Frequency of the signal in Hz
        
        if isinstance(accel_to_consider, str):
            accel_to_consider = [accel_to_consider]

        for col_name in accel_to_consider:

            if process_type:
                # Fourier Calculations
                sampling_rate = 6000  # Sampling rate in Hz
                duration = 6  # Duration of the signal in seconds

                # df_stacked_accel = pd.DataFrame(columns = ['Acceleration'])

                ## Perform the Fourier Transform
                spectrum = fft(df[col_name].to_numpy())

                ## Calculate the corresponding frequencies
                frequencies = np.fft.fftfreq(len(df[col_name]), 1 / sampling_rate)

                positive_freq_indices = np.where(frequencies >= 0)   # > could be, doesn't include 0 frequency

                # Extract the positive frequencies and corresponding amplitudes
                positive_frequencies = frequencies[positive_freq_indices]
                positive_spectrum = spectrum[positive_freq_indices]

                # Assuming complex_data is an array of complex numbers
                magnitude = np.abs(positive_spectrum)
                phase = np.angle(positive_spectrum)

                # Create a real-valued representation by stacking magnitude and phase
                feature_representation = np.stack((magnitude, phase), axis=-1)
                
                # Append the feature representation to the list
                x_list.append(feature_representation)
            
            else:


                # Processing of RAW Data

                feature_representation = np.stack((df[col_name][:18000], df[col_name][18000:]), axis=-1)
                x_list.append(feature_representation)
                ##########    
                # novo_df = pd.DataFrame()

                # # Dividindo os dados em duas colunas
                # novo_df['Coluna2'] = df['Coluna1'][:18000]  # Primeiros 18000 elementos
                # novo_df['Coluna3'] = df['Coluna1'][18000:]
                ##############
                
            ### Defining the x and y
            err_index = err_type.index(get_classification(filename, err_type))
            err_categ = [0]*6
            err_categ[err_index] = 1

            # Append the feature representation to the list
            y_list.append(err_categ)

    # Stack the feature representations to create x_train with shape (number_of_samples, number_of_positive_frequencies, 2)
    x = np.array(x_list)

    y = np.array(y_list)

    return x, y

def train_model(path_to_data, rot_freq, accel_train, accel_test, save_model, process_type):
    # Get dictionary with all {file_name: file_path} within a specified directory
    file_dict = list_files_in_directory(directory = path_to_data)
    # Filter by rotation frequency
    file_dict_by_rate = filter_files_by_words(file_dict = file_dict, words_to_search = rot_freq)
    # Split the data in training data and test data, can be randomized
    [file_dict_train, file_dict_test] = split_dict(file_dict = file_dict_by_rate, ratio = 0.8,random_split = True)

    # Process data by creating the Fourier Spectrum
    print('Process Data for Test')
    [x_test,y_test] = process_data(file_dict = file_dict_test, accel_to_consider = accel_test, process_type=process_type)

    print('Process Data for Train')
    [x_train,y_train] = process_data(file_dict = file_dict_train, accel_to_consider = accel_train, process_type=process_type)

    model = keras.Sequential([
        layers.Input(shape=(18000, 2)),
        layers.Reshape((300, 120, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=4, batch_size=4, validation_split=0.2)
    
    # Save Model
    if save_model:
        # Create a Model Name
        accel = ['Accel Coupled H', 'Accel Uncoupled H', 'Accel Coupled V', 'Accel Uncoupled V']
        accel_name = ['CoupH', 'UncoupH', 'CoupV', 'UncoupV']

        accel_train_name = []

        if isinstance(accel_train, str):
            accel_train = [accel_train]

        for string in accel_train:
            index = accel.index(string)
            value = accel_name[index]
            accel_train_name.append(value)

        model_name = 'Model_' + ''.join(rot_freq)+'_Trained_'+'_'.join(accel_train_name) + '.keras'
        path_to_save = os.getcwd() + '\\Models\\' + ''.join(rot_freq) + '\\' + '\\' + model_name

        model.save(path_to_save)

    return [model, x_test, y_test]

def test_model(model, x_test, y_test):
# Evaluate the model on the test data
    x_test_reshaped = x_test.reshape((-1, 18000, 2))
    test_loss, test_acc = model.evaluate(x_test_reshaped, y_test, verbose=2)
    #print(f'Test accuracy: {test_acc}')
    
    # Calculate the confusion matrix
    y_pred = model.predict(x_test_reshaped)

    # Find the index of the maximum value per row
    max_index = np.argmax(y_pred, axis=1)

    # Create a new array with all zeros, and set 1 in the entire row where the maximum value occurs
    result = np.zeros_like(y_pred)
    result[np.arange(result.shape[0]), max_index] = 1
    result.astype(int)
    
    y_true_classes = y_test.tolist()
    y_pred_classes = result.tolist()
    # print('y_true_classes')
    # print(y_true_classes)
    # print('y_pred_classes')
    # print(y_pred_classes)
    confmat = confusion_matrix(y_true_classes, y_pred_classes)

    # Return the test accuracy and the confusion matrix
    return test_acc, confmat

def confusion_matrix(actual, predicted):
    """Calculates the confusion matrix.

    Args:
        actual: A list of actual class labels.
        predicted: A list of predicted class labels.

    Returns:
        A list of lists representing the confusion matrix.
    """

    # Create a list to store unique classes
    classes = []

    # Iterate over the actual and predicted class labels to find unique classes
    for true_class, predicted_class in zip(actual, predicted):
        if true_class not in classes:
            classes.append(true_class)
        if predicted_class not in classes:
            classes.append(predicted_class)

    classes.sort()

    # Initialize the confusion matrix as a list of lists filled with zeros
    confmat = [[0 for _ in range(len(classes))] for _ in range(len(classes))]

    # Iterate over the labels
    for true_class, predicted_class in zip(actual, predicted):

        true_index = classes.index(true_class)
        predicted_index = classes.index(predicted_class)
        confmat[true_index][predicted_index] += 1

    return confmat

def plot_spectogram(path_to_data, rot_freq, accel_to_consider, fail, save):

    # Get dictionary with all {file_name: file_path} within a specified directory
    file_dict = list_files_in_directory(directory = path_to_data)

    # Filter by rotation frequency
    file_dict_by_rate = filter_files_by_word(file_dict = file_dict, word = rot_freq)

    # Filter by machine state
    file_dict_by_rate_fail = filter_files_by_word(file_dict = file_dict_by_rate, word = fail)


    # Process data by creating the Fourier Spectrum
    print('Process Data for Test')
 
    err_type = ['DesalinhHoriz', 'Desbalanc1', 'Desbalanc2', 'Normal', 'RolamDesbal', 'Rolam']
    rot_freq = ['10Hz', '20Hz', '30Hz', '40Hz']
    
    x_list = []

    if isinstance(accel_to_consider, str):
        accel_to_consider = [accel_to_consider]

    magnitude_spectrogram = [[] for _ in range(len(accel_to_consider))]

    for filename, filepath in file_dict_by_rate_fail.items():
        #print('*********************************')
        #print(f"Processing file: {filename}")

        data = np.genfromtxt(filepath, delimiter='\t', skip_header=23)  # Adjust skip_header based on your LVM file format
        df = pd.DataFrame(data)
        df.columns = ['X_Value', 'Time', 'Trigger', 'Accel Coupled H', 'Accel Coupled V', 'Accel Uncoupled H', 'Accel Uncoupled V', 'Vel Coupled H', 'Vel Coupled V', 'Vel Uncoupled H', 'Vel Uncoupled V']

        colunas_acc = ['Accel Coupled H', 'Accel Coupled V', 'Accel Uncoupled H', 'Accel Uncoupled V']

        for col_name in accel_to_consider:

            # Fourier Calculations
            sampling_rate = 6000  # Sampling rate in Hz
            duration = 6  # Duration of the signal in seconds

            # df_stacked_accel = pd.DataFrame(columns = ['Acceleration'])

            ## Perform the Fourier Transform
            spectrum = fft(df[col_name].to_numpy())

            ## Calculate the corresponding frequencies
            frequencies = np.fft.fftfreq(len(df[col_name]), 1 / sampling_rate)

            #positive_freq_indices = np.where(frequencies >= 0)   # > could be, doesn't include 0 frequency
            positive_freq_indices = np.where(frequencies > 0)

            # Extract the positive frequencies and corresponding amplitudes
            positive_frequencies = frequencies[positive_freq_indices]
            positive_spectrum = spectrum[positive_freq_indices]

            # Assuming complex_data is an array of complex numbers
            magnitude = np.abs(positive_spectrum)
            phase = np.angle(positive_spectrum)

            # Stack in different lists per acceleration considered


            # Assuming magnitude_spectrogram is a list of lists
            index_to_append = accel_to_consider.index(col_name)

            # Make sure that col_name is in accel_to_consider
            if index_to_append != -1:
                # Assuming magnitude is the value you want to append
                magnitude_spectrogram[index_to_append].append(magnitude)
            else:
                print(f"{col_name} is not in accel_to_consider.")
            print(magnitude_spectrogram)

            time = np.linspace(0, len(file_dict_by_rate_fail)*6,len(file_dict_by_rate_fail))

    #PLOT 3D
    for accel_considered in accel_to_consider:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for time and frequency
        T, F = np.meshgrid(time, positive_frequencies)

        accel_index = accel_to_consider.index(accel_considered)

        # Plot the 3D spectrogram
        ax.plot_surface(T, F, np.array(20*np.log10(magnitude_spectrogram[accel_index])).T, cmap='viridis') # plot in db
        #magnitude_spectrogram[accel_index] = np.where(magnitude_spectrogram[accel_index]>5000, 0, magnitude_spectrogram[accel_index])
        #ax.plot_surface(T, F, np.array(magnitude_spectrogram[accel_index]).T, cmap='viridis') # plot in not db
    
        #max_amplitude_db = 10000
        #ax.set_zlim(0, max_amplitude_db)

        fail = fail.replace('_', '')

        # Add labels
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Frequência (Hz)')
        ax.set_zlabel('Amplitude (dB)')
        ax.set_title('Espectrograma 3D - ' + fail + ' - ' + accel_considered)


        # Show the plot
        plt.show()
        if save:
            save_path = os.path.join(os.getcwd(), 'Spectrogram')

            # Ensure the directory exists; create it if not
            os.makedirs(save_path, exist_ok=True)

            # Specify the file name (you can customize the name based on your requirements)
            file_name = f"espectrograma_3D_{fail}_{accel_considered}.png"

            # Combine the path and file name
            file_path = os.path.join(save_path, file_name)

            # Save the figure as an image
            fig.savefig(file_path)

            # Close the figure to free up resources (optional)
            plt.close(fig)

    # PLOT 2D
    for accel_considered in accel_to_consider:
        # Create a 2D plot
        fig, ax = plt.subplots()

        accel_index = accel_to_consider.index(accel_considered)

        # Plot the 2D spectrogram using pcolormesh
        #pcm = ax.pcolormesh(time, positive_frequencies, np.array(magnitude_spectrogram[accel_index]).T, cmap='viridis')
        pcm = ax.pcolormesh(time, positive_frequencies, np.array(20*np.log10(magnitude_spectrogram[accel_index])).T, cmap='viridis')

        # Add labels
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Frequência (Hz)')
        ax.set_title('Espectrograma 2D - ' + fail + ' - ' + accel_considered)

        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('Amplitude (dB)')

        # Show the plot
        plt.show()

        if save:
            save_path = os.path.join(os.getcwd(), 'Spectrogram')
            os.makedirs(save_path, exist_ok=True)

            # Specify the file name
            file_name = f"espectrograma_2D_{fail}_{accel_considered}.png"

            # Combine the path and file name
            file_path = os.path.join(save_path, file_name)

            # Save the figure as an image
            fig.savefig(file_path)

            # Close the figure to free up resources (optional)
            plt.close(fig)