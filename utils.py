import numpy as np

AA_list = ['A', 'R', 'N', 'D', 'C',
     'Q', 'E', 'G', 'H', 'I',
     'L', 'K', 'M', 'F', 'P',
     'S', 'T', 'W', 'Y', 'V']

AA_dict = dict(zip(AA_list, range(20)))

def aa_to_ord(aa):
     try:
          return AA_dict[aa]
     except:
          return -1

def aa_seq_to_ord(aa_seq):
     return np.vectorize(aa_to_ord)(aa_seq)