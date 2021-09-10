def loadMATfile(matfilename):
# Loading a 3D .mat file after preprocessing EEG data in MATLAB using LETSWAVE
    import scipy.io as scio
    import numpy as np
    file_type = ".mat"
    if ".mat" in matfilename:
        matfilename=matfilename[:-4]
    full_file = matfilename + file_type
    data_dict = scio.loadmat(full_file)
    array_name = list(data_dict)[-1]
    data = data_dict[array_name]
    del array_name 
    del data_dict
    return data

def runSVMshort(convertedmatfile, plot=None, emo1=None, emo2=None):
    #Arguments
    #convertedmatfile = output of loadMATfile function
    #emo1 = 1 of the 24 emotions 
    #emo2 = 1 of the 24 emotions 
    #plot = visualizing the support vectors; 1 = yes plot; 0 = no plot, default is no plot
    
    import random
    import numpy as np
    import numpy.matlib
    import sklearn 
    import matplotlib.pyplot as plt

    if emo1 is None:
        emo1 = random.randint(1, 24)
    if emo2 is None:
        emo2 = random.randint(1, 24)
    if emo1 == emo2:
        emo2 = random.randint(1, 24)
  
    print("First emotion is", emo1)
    print("Second emotion is",emo2)
    if plot == 1:
        print("A plot will be made.")
    if plot == 0 or plot is None:
        print("No plot will be made.")
        
    emo1 = np.squeeze(convertedmatfile[:, emo1, :])
    emo2 = np.squeeze(convertedmatfile[:, emo2, :])
    all_data_for_svm = np.concatenate((emo1, emo2), axis=0)
    #target labels
    targets = np.concatenate((np.matlib.repmat(1, emo1.shape[0], 1), np.matlib.repmat(0, emo2.shape[0], 1)), axis = 0)
    
    #building classifier
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(all_data_for_svm, targets, test_size=0.25,random_state=109) 
    #75% training and 25% test
    #Import svm model
    from sklearn import svm
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel
    #Train the model using the training sets
    clf.fit(X_train, y_train.ravel())
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred))
    
    if plot == 1:
        # Get support vectors themselves
        support_vectors = clf.support_vectors_
        # Visualize support vectors
        plt.scatter(X_train[:,0], X_train[:,1])
        plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
        plt.title('Linearly separable data with support vectors')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
        
    return all_data_for_svm
    return clf

def runSVMlong(convertedmatfile):
    #Arguments
    #convertedmatfile = output of loadMATfile function
    
    import random
    import numpy as np
    import numpy.matlib
    import sklearn 
    import matplotlib.pyplot as plt
    import itertools
    import multiprocessing as mp
    print("Number of processors: ", mp.cpu_count())

    all_combos = np.array(list(itertools.combinations(range(1, 25), 2)))
    recall = []
    accuracy = []
    precision = []
    
    for x in range(0, all_combos.shape[0]-1):
        emo1 = np.squeeze(convertedmatfile[:, all_combos[x, 0], :])
        emo2 = np.squeeze(convertedmatfile[:, all_combos[x, 1], :])
        print("Computing combo", x+1, "of", all_combos.shape[0])
        #print("First emotion is", all_combos[x, 0])
        #print("Second emotion is",all_combos[x, 1])
        all_data_for_svm = np.concatenate((emo1, emo2), axis=0)
        #target labels
        targets = np.concatenate((np.matlib.repmat(1, emo1.shape[0], 1), np.matlib.repmat(0, emo2.shape[0], 1)), axis = 0)
        #building classifier
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(all_data_for_svm, targets, test_size=0.25,random_state=109) 
        #75% training and 25% test
        #Import svm model
        from sklearn import svm
        #Create a svm Classifier
        clf = svm.SVC(kernel='linear') # Linear Kernel
        #Train the model using the training sets
        clf.fit(X_train, y_train.ravel())
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)
        #Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics
        # Model Accuracy: how often is the classifier correct?
        #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        # Model Precision: what percentage of positive tuples are labeled as such?
        #print("Precision:",metrics.precision_score(y_test, y_pred))
        # Model Recall: what percentage of positive tuples are labelled as such?
        #print("Recall:",metrics.recall_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        
    print("Average accuracy is ", np.mean(recall, axis = 0))
    #del all_data_for_svm
    return recall
    return accuracy
    return precision
