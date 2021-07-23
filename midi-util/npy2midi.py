import  numpy as np

from p2m import  p_2_mmat,mmat_2_mmidi
# from interval2onehot import  d4onehot_to_real,real_to_d4onehot
# from file_batch import beat_resolution,is_monophonic


d4=np.load("test128length.npy")
mmat_2_mmidi(d4, "test128t30.mid",tempo=30)