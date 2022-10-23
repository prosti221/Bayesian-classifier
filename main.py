from gaussian_classifier import *
from one_vs_rest import *
from glcm import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
   # Load mosaic1
    img = np.loadtxt('mosaic1_train.txt', delimiter=',')

    # Load mosaic2_test
    img_test1 = np.loadtxt('mosaic2_test.txt', delimiter=',')

    # Load mosaic3_test
    img_test2 = np.loadtxt('mosaic3_test.txt', delimiter=',')

    # test mask 1
    mask_test1 = np.loadtxt('mask2_mosaic2_test.txt', delimiter=',')

    # test mask 2
    mask_test2 = np.loadtxt('mask3_mosaic3_test.txt', delimiter=',')

    # Training mask
    mask = np.loadtxt('training_mask.txt', delimiter=',')

    g2 = glcm(img, distance=1, angle=0.0, levels=16, window_size=29) 
    g4 = glcm(img, distance=3, angle=45.0, levels=16, window_size=29)

    features1 = compute_Q(g2)
    features2 = compute_Q(g4)

    f_full = np.concatenate((features1, features2), axis=2)
    f_full = np.delete(f_full, [1, 5, 1, 12, 8], axis=2)

    # Creating a single gaussian classifier from the best found feature combination
    model = bayesianModel()
    zero_p = 0.1e-3
    model.c_priors = [zero_p, 0.25-(zero_p/4), 0.25-(zero_p/4), 0.25-(zero_p/4), 0.25-(zero_p/4)]
    model.fit(f_full[:, :, (1, 4)], mask)
    preds = model.predict(f_full[:,:,(1, 4)])

    print('Single gaussian classifier accuracy on training data: %f' %(model.accuracy))
    '''
    plt.figure()
    plt.imshow(preds)
    plt.show()
    model.plot_confusion_mat()
    '''

    # Training a set of gaussian classifiers on the good features 
    # These gaussian classifiers can then be used in conjunction to find the most probable class
    feat = [f_full[:,:,(1, 3, 9)],
            f_full[:,:,(4, 5, 6)]
           ]
    gaus_models = []

    for i in range(len(feat)):
        model = bayesianModel()
        model.fit(feat[i], mask)
        model.predict(feat[i])
        gaus_models.append(model)

    classifier = oneVsRest(np.array(gaus_models))
    preds = classifier.predict()

    print('Multi-gaussian classifier accuracy on training data: %f' %(classifier.accuracy))
    '''
    plt.figure()
    plt.imshow(preds)
    plt.show()

    classifier.plot_confusion_mat()
    '''

    # Testing the classifier on the first test image 
    g2_test1 = glcm(img_test1, distance=1, angle=0.0, levels=16, window_size=29) #dx=3 best so far
    g4_test1 = glcm(img_test1, distance=3, angle=45.0, levels=16, window_size=29)

    features1_test1 = compute_Q(g2_test1)
    features2_test1 = compute_Q(g4_test1)

    f_full_test1 = np.concatenate((features1_test1, features2_test1), axis=2)
    f_full_test1 = np.delete(f_full_test1, [1, 5, 1, 12, 8], axis=2)

    feat_test1 = [f_full_test1[:,:,(1, 3, 9)],
                  f_full_test1[:,:,(4, 5, 6)]
                 ]

    for i in range(len(feat_test1)):
        gaus_models[i].predict(feat_test1[i])

    classifier_test1 = oneVsRest(np.array(gaus_models))
    preds_test1 = classifier_test1.predict()

    # Setting accuracy and confusion matrix based on test mask
    classifier_test1.get_accuracy(preds_test1, mask_test1)
    classifier_test1.get_confusion_mat(preds_test1, mask_test1)

    print('Multi-gaussian classifier accuracy on training data: %f' %(classifier_test1.accuracy))

    '''
    plt.figure()
    plt.imshow(preds_test1)
    plt.show()

    classifier_test1.plot_confusion_mat()
    '''
