from scipy.io import loadmat
from itertools import combinations
import numpy as np
import scipy as sp
from skimage.feature import graycomatrix
import matplotlib.pyplot as plt

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
    quad_div_num = 1
    quad_tiles = tile(sub_glcms[:,:,quad_div_num,:,:]) #divinding the second quadtrant

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

def dissimilarity(glcm):
    vals = np.zeros((glcm.shape[0], glcm.shape[1]))
    s = np.sum(glcm)
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[2]):
            vals += glcm[:,:,i,j] * np.abs(i-j)
    return vals

def cluster_shade(glcm):
    vals = np.zeros((glcm.shape[0], glcm.shape[1]))
    s = np.sum(glcm)
    mean_x, mean_y = mean(glcm)
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[2]):
            vals += glcm[:,:,i,j] * np.abs((i+j-mean_x-mean_y)**3)
    return vals

def contrast(glcm):
    vals = np.zeros((glcm.shape[0], glcm.shape[1]))
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[2]):
            vals += glcm[:, :,i,j] * (i-j)**2

    return vals

def homogeneity(glcm):
    vals = np.zeros((glcm.shape[0], glcm.shape[1]))
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[2]):
            vals += glcm[:,:, i, j]/(1.+(i-j)**2)

    return vals

def asm(glcm):
    vals = np.zeros((glcm.shape[0], glcm.shape[1]))
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[2]):
            vals  += glcm[:, :, i,j]**2

    return vals

def mean(glcm):
    mean_x = 0 # mean of columns i
    mean_y = 0 # mean of rows j
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[2]):
            mean_x += glcm[:,:,i,j] * i
            mean_y += glcm[:,:,i,j] * j
    return (mean_x, mean_y)

    #For each subdivision of the glcm, we have a scalar value
    features = np.zeros((8, 4))

    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            features[i, j] = np.sum(glcms[i, j]) / np.sum(temp[i])

class bayesianModel():
    def __init__(self):
        self.mean_vecs = []     # Mean values for each class
        self.cov_mats = []      # Standard deviation for each class
        self.c_priors = 0   # Class prior probability
        self.class_num = 0  # Number of classes
        self.accuracy = 0   # Accuracy from last training

        #Input is in the shape of (W, H, features)
        self.x = None       # Last training input
        self.y = None       # Last training true values

    # Gaussian distribution P(X | Y) * P(Y)
    def gauss(self, X, mean_vec, cov_mat, prior):
        k = X.shape[0]; m = np.matrix(X-mean_vec)
        return (1/( (2*np.pi)**0.5*sp.linalg.det(cov_mat)**0.5)) \
                * np.exp(-0.5*(X-mean_vec)@sp.linalg.inv(cov_mat)@(X-mean_vec).T)

    def softmax(self, x):
        return x/sum(x)

    # Predict for each pixel
    def predict(self, x):
        preds = np.zeros((x.shape[0], x.shape[1])).astype(int)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                norm_vals = ([self.gauss(x[i, j], self.mean_vecs[c], self.cov_mats[c], self.c_priors[c]) for c in range(self.class_num)])
                preds[i, j] = np.argmax(norm_vals)
                #print(self.softmax(norm_vals))
        self.accuracy = self.get_accuracy(preds, self.y)
        return preds

    def get_accuracy(self, y_pred, y):
        return np.mean(y_pred == y)

    # Train the classifier
    def fit(self, x, y):
        self.x = x; self.y = y
        self.class_num = len(np.unique(y))
        #Mean-value vector for each feature f calcualted for each class c
        self.mean_vecs = [np.mean(x[y==i], axis=(0, 1)) for i in range(self.class_num)]
        #Covariance matrix from all features f calculated for each class c
        self.cov_mats = [np.cov(x[y==i], rowvar=False) for i in range(self.class_num)]
        self.c_priors = np.array([y[y==i].shape[0] for i in range(self.class_num)]) / (y.shape[0] * y.shape[1])

if __name__ == '__main__':
# Load mosaic1
    img = np.loadtxt('mosaic1_train.txt', delimiter=',')

    plt.figure()
    plt.imshow(img, cmap='gray')

# Training mask
    mask = np.loadtxt('training_mask.txt', delimiter=',')

    g2 = glcm(img, distance=1, angle=0.0, levels=16, window_size=31) #dx=3 best so far
    g4 = glcm(img, distance=3, angle=45.0, levels=16, window_size=31)

    features1 = compute_Q(g2)
    features2 = compute_Q(g4)

    f_full = np.concatenate((features1, features2), axis=2)
    f_full = np.concatenate((features1, features2), axis=2)
    f_full = np.delete(f_full, [12,11,10,1], axis=2)

    current_best_model = None
    current_best_acc = 0.0
    current_best_indecies = None
    count = 0
    A = [i for i in range(f_full.shape[2])]
    for i in range(2, f_full.shape[2]):
        for j in combinations(A, i):
            model = bayesianModel()
            model.fit(f_full[:, :, j], mask)
            cont = True
            for cov in model.cov_mats:
                if (np.linalg.eigh(cov)[0] <= 0).all() == True:
                    cont = False
            if cont:
                preds = model.predict(f_full[:,:,j])
                train_acc = model.accuracy
                if train_acc > current_best_acc:
                    current_best_model = model
                    current_best_acc = train_acc
                    current_best_indecies = j
                    print('New best accuracy: %f'%(current_best_acc))
                    print('New best accuracy: %s'%(current_best_indecies, ))
            count += 1
            print('Estimated time remaining: %d'%((63777-count)*0.005704))
