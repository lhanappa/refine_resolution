import os
import numpy as np

def redircwd_back_projroot(project_name='refine_resolution'):
    root = os.getcwd().split('/')
    for i, f in enumerate(root):
        if f == project_name:
            root = root[:i+1]
            break
    root = '/'.join(root)
    os.chdir(root)
    print('current working directory: ', os.getcwd())
    return root

def remove_zeros(matrix):
    idxy = ~np.all(np.isnan(matrix), axis=0)
    M = matrix[idxy, :]
    M = M[:, idxy]
    M = np.asarray(M)
    idxy = np.asarray(idxy)
    return M, idxy