
from embedder import NLP_embedder
import torch
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    
    from data import load_data
    
    X_train, X_test, Y_train, Y_test = load_data() # data is glue/sst2 from tensorflow datasets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size= 8
    num_classes = 2 
    
    ## Beispiel wo als classifier auch ein neuronales netz benutzt wird.

    hyperparameters = {}
    hyperparameters["model_name"] = 'google/electra-small-discriminator'
    hyperparameters["hidden_fc"] = 50
    hyperparameters["lr"] = 1e-5
    classifier = NLP_embedder(hyperparameters, num_classes, batch_size)
    classifier = classifier.to(device)
    classifier.fit(X_train, Y_train, epochs = 1)
    Y_pred = classifier.predict(X_test).to("cpu")
    Y_pred= np.argmax(Y_pred, axis = 1)
    acc = accuracy_score(Y_test, Y_pred)
    print("accuracy_score", acc)
    
    ## Beispiel wo als classifier svm von scikit learn benutzt wird
    from sklearn import svm
    
    hyperparameters = {}
    hyperparameters["model_name"] = 'google/electra-small-discriminator'
    hyperparameters["hidden_fc"] = 50
    hyperparameters["lr"] = 1e-5
    classifier = NLP_embedder(hyperparameters, num_classes, batch_size)
    classifier = classifier.to(device)
    embedded_X = classifier.embed(X_train[:1024]).to("cpu")
    clf = svm.SVC()
    clf.fit(embedded_X, Y_train[:1024])
    
    X_test = X_test[:96]
    Y_test = Y_test[:96]
    embedded_X_test = classifier.embed(X_test).to("cpu")
    Y_pred = clf.predict(embedded_X_test)
    
    acc = accuracy_score(Y_test, Y_pred)
    print("accuracy_score", acc)
    
if __name__ == '__main__':
    main()