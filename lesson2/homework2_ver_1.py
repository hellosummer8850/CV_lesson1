# much clear version of conv

def padding2(img, kernel, padding_way = 'ZERO'):

    import numpy as np
    img = np.array(img)
    kernel = np.array(kernel)
    l, w = kernel.shape
    if padding_way == 'ZERO':
        img = np.pad(img, ((int(l/2), int(l/2)), (int(w/2), int(w/2))), 'constant', constant_values=0)
    if padding_way == 'REPLICA':
        img = np.pad(img, ((int(l/2), int(l/2)), (int(w/2), int(w/2))), 'edge')

    return img.tolist()

import numpy as np


img = [[1,2,1,3,1],[4,1,3,1,5],[7,1,1,1,9],[2,1,1,1,9],[2,3,5,3,1]]
kernel = [[1,1,1],[1,1,1],[1,1,1]]
print(padding(img, kernel,'ZERO'))