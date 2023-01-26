# Perceptron-Project

## Music Genre Predictor 🎵🎶

This project uses a Jupyter notebook to train a single layer perceptron on a Kaggle dataset of music genres. The model is trained on 75% of the dataset and tested on the remaining 25% in order to predict the genre of a song based on tempo, chroma_stft, and spectral_centroid.

## Requirements

- Python 3.x
- Jupyter Notebook
- Numpy
- Matplotlib

## The algorithm

```python
def perceptron(X, Y, max_num_iterations):
    
    curr_w = np.zeros(X.shape[1]) # init the w vector
    best_w = np.zeros(X.shape[1]) # keep track of the best solution
    num_samples = X.shape[0]
    best_error = 2 #init
    curr_error = 1 #init
    
    index_misclassified = -2  
    num_misclassified = 0 
    
    #main loop continue until all samples correctly classified or max # iterations reached
    num_iter = 1
    
    while ((index_misclassified != -1) and (num_iter < max_num_iterations)):
        
        index_misclassified = -1 # if at the end of the loop it's still -1 .. no misclass. found!
        num_misclassified = 0    # counter for the error
        
        # avoid working always on the same sample with random permutation
        permutation = np.random.permutation(num_samples) 
        X = X[permutation]
        Y = Y[permutation]
        
        for i in range(num_samples):
            check_sum = np.sum(X[i] * curr_w)      # Sum of the elements averaged by the vector W
            if Y[i] * check_sum <= 0:                     
                num_misclassified += 1
                index_misclassified = i            
        
        if index_misclassified == -1 : 
            print("No errors found, the algorithm classified perfectly the training sample")
            break
            
        #update error count, keep track of best solution
        curr_error = num_misclassified / num_samples
        if curr_error < best_error:
            best_error = curr_error
            best_w = curr_w
            
        num_iter += 1
        if num_iter == max_num_iterations : print("Max iteration exceded")
        
        #call update function using a misclassifed sample
        if index_misclassified != -1: #a misclassification has been found.. otherwise no update
            curr_w = perceptron_update(curr_w, X[index_misclassified], Y[index_misclassified]) 
    
    return best_w, best_error
```

## Conclusion
The model was able to achieve an accuracy of around 90% on the testing dataset. However, the model can be improved by using other machine learning algorithms or by using more features. 🚀

