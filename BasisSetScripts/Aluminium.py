import numpy as np
import sklearn
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
import pandas as pd

file_dir = "/Users/Michael/OneDrive - Drexel University/Documents - Chang Lab/General/Group/Data/Ultrasound/Layered electrode study/"
filesAL = ['20240216_WC_GJ_water-ref_dur15_del1p5_50MHz.sqlite3', '20240216_WC_GJ_Al-n=1_dur15_del1p5_50MHz.sqlite3',
         '20240216_WC_GJ_Al-n=2_dur15_del1p5_50MHz.sqlite3', '20240216_WC_GJ_Al-n=3_dur15_del1p5_50MHz.sqlite3',
         '20240216_WC_GJ_Al-n=4_dur15_del1p5_50MHz.sqlite3', '20240216_WC_GJ_Al-n=5_dur15_del1p5_50MHz.sqlite3']

####################################
def fft_acoustics(waves, interval=1 / 5e8, pad_x=2):
    l = len(waves[0]) * pad_x
    n_waves = len(waves)
    z = np.zeros((n_waves, l), dtype='uint8')

    pad_waves = np.concatenate((z, waves, z), axis=1)

    m = np.mean(pad_waves, axis=-1)
    detrend = np.subtract(pad_waves.T, m).T
    amps = np.abs(np.fft.rfft(detrend, axis=-1))
    amps = np.divide(amps.T, np.max(amps, axis=-1)).T

    freqs = np.fft.rfftfreq(len(detrend[0]), d=interval)
    return freqs, amps
################################################



###################
for n, f in enumerate(filesAL):
    connection = sqlite3.connect(file_dir + f)
    cursor = connection.cursor()
    query = """SELECT name FROM sqlite_master WHERE type='table'"""
    cursor.execute(query)
    table = cursor.fetchall()

    # select every nth wave to speed up processing
    query = f'SELECT * FROM "{table[0][0]}" WHERE "time" % 2 == 0'
    df = pd.read_sql(con=connection, sql=query)
    connection.close()

    waves_formatted = df['amps'].str.strip('[]').str.split(',')
    waves = np.zeros(
        (len(waves_formatted), len(waves_formatted[0])),
        dtype=np.float16
    )

    for i, wave in enumerate(waves_formatted):
        waves[i, :] = wave

    #target_wave = waves / np.linalg.norm(waves)
    sparse_coder = SparseCoder(dictionary=waves, transform_algorithm="omp", transform_alpha=20)
    sparse_reps = sparse_coder.transform(waves)
    plt.plot(sparse_reps)
  #  learned_basis_functions = dictionary_learner.components_.T
  #  approximation_dictionary_learning = np.dot(learned_basis_functions, dictionary_learner.transform(
  #      waves).T)
  #  for i, approximation in enumerate(approximation_dictionary_learning):
  #      plt.plot(approximation, label=f'Approximation {i + 1}')
   # approximations = sparse_coder.inverse_transform(sparse_reps)
   # plt.plot(approximations)
    amps = []
    times = (df['time'] - df['time'].iloc[0]) / 3600.
    tofshift = []
    tofs = np.linspace(11, 11 + 3, len(waves[0]))

    freqs, amps = fft_acoustics(waves)
    normalize = Normalize(vmin=0, vmax=len(filesAL))
    print(n, f)
    plt.grid(True)
    plt.show()

    ##########################
