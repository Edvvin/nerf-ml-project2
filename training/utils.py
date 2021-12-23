import numpy as np
# List of amino acids
AA_list = ['A', 'R', 'N', 'D', 'C',
     'Q', 'E', 'G', 'H', 'I',
     'L', 'K', 'M', 'F', 'P',
     'S', 'T', 'W', 'Y', 'V']
AAmap = np.array(AA_list)

# Mapping of amino acid (i.e., character) to index
AA_dict = dict(zip(AA_list, range(20)))

def aa_to_ord(aa):
     '''
     Converts character representation of amino acid to integer representation.
     Returns -1 for invalid input.

     Parameters
     ----------
     aa: character
          Amino acid to be converted
     
     Returns
     -------
     int
     '''
     try:
          return AA_dict[aa]
     except:
          return -1

def aa_seq_to_ord(aa_seq):
     '''
     Converts sequence of amino acids to integer representation (i.e, array of integers)

     Parameters
     ----------
     aa_seq: no.array<char>
          Amino acid sequence to be converted
     
     Returns
     -------
     np.array<int> 
          Integer representation
     '''
     return np.vectorize(aa_to_ord)(aa_seq)
