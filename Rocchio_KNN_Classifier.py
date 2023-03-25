#takes as input the training data matrix and the training labels, returning the prototype vectors for each class.

def Rocchio_Train(train, labels):
    p = np.zeros((len(np.unique(labels)),train.shape[1]))
    for x in train[labels==0]:
        p[0] += x
    for z in train[labels==1]:
        p[1] += z
    return prototype

#take as input the prototypes and the testmatrix
#measure Cosine similarity of the test instance to each prototype vector
#return predicted class for the test instance and the similarity values of the instance to each of the category prototypes

def Rocchio_classifier(prototype, testmatrix):  
    sims = np.zeros([prototype.shape[0],testmatrix.shape[0]])
    m = -2
    for p in range(prototype.shape[0]):
        for i in range(testmatrix.shape[0]):
            num = np.dot(prototype[p],testmatrix[i])
            den = np.linalg.norm(prototype[p]) * np.linalg.norm(testmatrix[i])
            sim = num/den
            if sim > -2:
                m = sim
                sims[p,i] += m
            else:
                sims[p,i] += m
    sims = sims.T
    indexes = np.zeros([sims.shape[0],1])
    
    for r in range(sims.shape[0]):
        a =  np.argmax(sims[r])
        indexes[r,0] += a
    pred_labels = np.int32(indexes)

    return sims, pred_labels

#evaluates accuracy of classifier based on ratio of correct predictions to the number of test instances

def rocchio_evaluate(testmatrix, test_lab, prototype):
    sims, pred_labels = Rocchio_classifier(prototype, testmatrix)
    c = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == test_lab[i]:
            c += 1
    accuracy = c/len(test_lab)
    return accuracy