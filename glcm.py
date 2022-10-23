from skimage.feature import graycomatrix
import numpy as np

#This is the function that takes care of the sliding window
def img_transform(img, minimum=0, maximum=255, levels=8):
    bins = np.linspace(minimum, maximum+1, levels+1)
    transformed_img = np.digitize(img, bins) - 1
    return transformed_img

def glcm(img, window_size=5, distance=1, angle=0.0, levels=8):
    # Reduce the number of gray-levels to specified value
    img = img_transform(img, levels=levels)
    i, j = img.shape

    #Output will have a seperate GLCM for every pixel of dimension (levels, levels)
    full_glcm = np.zeros((i, j, levels, levels))
    for row in range(0, i):
        for col in range(0, j):
            i_offset = i - (row + window_size)
            j_offset = j - (col + window_size)
            # The indecies are mirrored when out of bounds
            if row + window_size > i and col + window_size > j:
                seg = img[row :, col :]
                #Add the row i
                seg = np.concatenate((seg, img[i_offset:, col : col + (window_size - 1 - j_offset)][::-1, ::]), axis=0)
                #Add the col j
                seg = np.concatenate((seg, img[row + i_offset:, j_offset:][::,::-1]), axis=1)
            elif row + window_size > i:
                seg = img[row :, col : col + (window_size - 1)]
                seg = np.concatenate((seg, img[i_offset:, col : col + (window_size - 1)][::-1, ::]), axis=0)
            elif col + window_size > j:
                seg = img[row : row + (window_size - 1) :, col :]
                seg = np.concatenate((seg, img[row : row + (window_size - 1), j_offset:][::,::-1]), axis=1)
            else:
                seg = img[row : row + (window_size - 1), col : col + (window_size - 1)]
            #Calculate the GLCM for the given pixel based on the segmented region
            sub_glcm = glcm_sub(seg, distance=distance, angle=angle, levels=levels)
            full_glcm[row, col] = sub_glcm
    return full_glcm

# Calculates the GLCM
def glcm_sub(img, distance=1, angle=0.0, levels=8):
    G = graycomatrix(img, levels=levels, distances=[distance], angles=[angle], symmetric=True, normed=True)[:,:,0,0]    
    return G/np.sum(G)

def tile(mat):
    mat_size = mat.shape[-1]
    nrows = ncols = mat_size//2
    res = np.zeros((mat.shape[0], mat.shape[1], 4, mat_size//2, mat_size//2))
    
    for i in range(0, mat.shape[0]):
            for j in range(0, mat.shape[1]):
                res[i, j] = (mat[i,j].reshape(mat_size//nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))
    return res
                
def compute_Q(glcm): # i, j, 16, 16
    glcm_dim = glcm.shape[2]
    sub_glcms = tile(glcm) # i, j, 4, 8, 8
    quad_div_num = 0
    quad_tiles = tile(sub_glcms[:,:,quad_div_num,:,:]) #divinding the first quadtrant
    
    vals = np.zeros((glcm.shape[0], glcm.shape[1], 4))
    sub_vals = np.zeros((glcm.shape[0], glcm.shape[1], 4))
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            s = np.sum(glcm[i, j])
            for k in range(4):
                vals[i, j, k] = np.sum(sub_glcms[i, j, k])  /  s
                sub_vals[i, j, k] = np.sum(quad_tiles[i, j, k]) / s
    vals = np.delete(vals, quad_div_num, axis=2)

    return np.concatenate((vals, sub_vals), axis=2)
