import numpy as np
import pandas as pd
import glob
import csv
import librosa
import data
import os
import subprocess

__author__ = 'namju.kim@kakaobrain.com'

_data_path = "F:/Speech/"


def process_vctk(csv_file):
  # create csv writer
  writer = csv.writer(csv_file, delimiter=',')

  # read label-info
  df = pd.read_table(_data_path + 'VCTK-Corpus/speaker-info.txt', usecols=['ID'],
                     index_col=False, delim_whitespace=True)

  # read file IDs
  file_ids = []
  for d in [_data_path + 'VCTK-Corpus/txt/p%d/' % uid for uid in df.ID.values]:
    file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

  for i, f in enumerate(file_ids):

    # wave file name
    wave_file = _data_path + 'VCTK-Corpus/wav48/%s/' % f[:4] + f + '.wav'
    fn = wave_file.split('/')[-1]
    target_filename = 'data/preprocess/mfcc/' + fn + '.npy'
    if os.path.exists(target_filename):
      continue
    # print info
    print("VCTK corpus preprocessing (%d / %d) - '%s']" % (i, len(file_ids), wave_file))

    # load wave file
    wave, sr = librosa.load(wave_file, mono=True, sr=None)

    # re-sample ( 48K -> 16K )
    wave = wave[::3]

    # get mfcc feature
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # get label index
    label = data.str2index(open(_data_path + 'VCTK-Corpus/txt/%s/' % f[:4] + f + '.txt').read())

    # save result ( exclude small mfcc data to prevent ctc loss )
    if len(label) < mfcc.shape[1]:
      # save meta info
      writer.writerow([fn] + label)
      # save mfcc
      np.save(target_filename, mfcc, allow_pickle=False)


if not os.path.exists('data/preprocess'):
  os.makedirs('data/preprocess')
if not os.path.exists('data/preprocess/meta'):
  os.makedirs('data/preprocess/meta')
if not os.path.exists('data/preprocess/mfcc'):
  os.makedirs('data/preprocess/mfcc')

#
# Run pre-processing for training
#

# VCTK corpus
csv_f = open('data/preprocess/meta/train.csv', 'w')
process_vctk(csv_f)
csv_f.close()