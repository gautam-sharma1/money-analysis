################################################################
# @author Gautam Sharma                                        #
# Project 1 B                                                  #
# Program to test put different ML algorithms                  #
# Best parameters are printed out at the end                   #
################################################################

from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import Perceptron            # the algorithm
from sklearn.metrics import accuracy_score             # grade the results
from sklearn.linear_model import LogisticRegression    # the algorithm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

bank_note = pd.read_csv('data_banknote_authentication.txt')
X = bank_note.get(["variance","skewness","curtosis","entropy"])     # separate the features we want
y = bank_note.get(["class"])                                        # extract the classifications

############################################################
# Class that contains all ML algorithms                    #
# inputs to constructor:                                   #
#           X = data to help predict y                     #
#           y = labels                                     #
# outputs:                                                 #
#   function call to different ML algorithms               #
############################################################


class Model:

    def __init__(self, X, y):
        # split the problem into train and test this will yield 70% training and 30% test
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3, random_state=0)

        # scale X by removing the mean and setting the variance to 1 on all features.
        # the formula is z=(x-u)/s where u is the mean and s is the standard deviation.
        # (mean and standard deviation may be overridden with options...)
        self.sc = StandardScaler()
        self.sc.fit(self.X_train)  # compute the required transformation
        self.X_train_std = self.sc.transform(self.X_train)  # apply to the training data
        self.X_test_std = self.sc.transform(self.X_test)  # and SAME transformation of test data

    ###################################################################################
    # Function that performs grid search to find the best parameters                  #
    # inputs :                                                                        #
    #           model = ML model to use                                               #
    #           param_grid = parameters to optimize on                                #
    #           verbose = output to the console?                                      #
    # outputs:                                                                        #
    #           best_model = model having the best parameters                         #
    #           prospective_models = all models considered with different parameters  #
    ###################################################################################

    def best_model(self, model, param_grid, verbose=0):
        prospective_models = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, verbose=verbose)
        prospective_models.fit(self.X_train_std, self.y_train.values.ravel())
        best_model = prospective_models.best_estimator_
        return best_model, prospective_models

    ###################################################################################
    # Function that calculates test and train accuracy                                #
    # inputs :                                                                        #
    #           best model = best model selected from GridSearch                      #
    #           verbose = output to the console?                                      #
    # outputs:                                                                        #
    #           test_accuracy = accuracy of the test data set                         #
    ###################################################################################
    def accuracy(self,best_model, verbose=0):
        y_pred_train = best_model.predict(self.X_train_std)
        y_pred_test = best_model.predict(self.X_test_std)  # now try with the test data
        train_accuracy = accuracy_score(self.y_train.values.ravel(), y_pred_train.ravel())
        test_accuracy = accuracy_score(self.y_test.values.ravel(), y_pred_test.ravel())
        if verbose:
            print('Number in test ', len(self.y_test))

            print('Misclassified samples during training: %d' % (
                    self.y_train.values.ravel() != y_pred_train.ravel()).sum())  # how'd we do?
            print('Average Accuracy during training  : %.3f' % train_accuracy)

            print('Misclassified samples during testing: %d' % (
                    self.y_test.values.ravel() != y_pred_test.ravel()).sum())  # how'd we do?
            print('Average Accuracy during testing  : %.3f' % test_accuracy)

        return test_accuracy

    ##########################################################
    # Function that runs the perceptron algorithm            #
    # inputs :                                               #
    #           verbose = output to the console?             #
    # outputs:                                               #
    #           calls accuracy() to return test set accuracy #
    ##########################################################

    def perceptron(self, verbose=0):
        param_grid = [{'max_iter': [5,10,20,40,80,100,500]}]                         # optimizing over max_iter
        ppn = Perceptron(tol=1e-3, eta0=0.001,
                         fit_intercept=True, random_state=0, verbose=verbose)
        best_model, prospective_models = self.best_model(ppn, param_grid, verbose)   # get the best model

        # print the best parameter
        print("Parameters that gives the highest accuracy achieved for perceptron ", prospective_models.best_params_)

        return self.accuracy(best_model, verbose)                                    # return the best test set accuracy

    ##########################################################
    # Function that runs the logistic regression algorithm   #
    # inputs :                                               #
    #           verbose = output to the console?             #
    # outputs:                                               #
    #           calls accuracy() to return test set accuracy #
    ##########################################################

    def logistic_regression(self,verbose = 0):
        param_grid = [{'C': [1,5,10,20,30,50,100]}]                            # optimizing over C
        lr = LogisticRegression(solver='liblinear', \
                                multi_class='ovr', random_state=0)
        best_model, prospective_models = self.best_model(lr, param_grid, verbose)     # get the best model

        # print the best parameter
        print("Parameters that gives the highest accuracy achieved for logistic regression ",\
              prospective_models.best_params_)

        return self.accuracy(best_model,  verbose)                 # return the best test set accuracy

    ###########################################################
    # Function that runs the support vector machine algorithm #
    # inputs :                                                #
    #           verbose = output to the console?              #
    # outputs:                                                #
    #           calls accuracy() to return test set accuracy  #
    ###########################################################

    def support_vector_machine(self,verbose=0):
        param_grid = [{'C': [0.1,1.0,5.0,10.0,20.0]}]                       # optimizing over C
        svm = SVC(kernel='linear',random_state=0)
        best_model, prospective_models = self.best_model(svm, param_grid, verbose)  # get the best model

        # print the best parameter
        print("Parameters that gives the highest accuracy achieved for SVM ",
        prospective_models.best_params_)

        return self.accuracy(best_model, verbose)   # return the best test set accuracy

    ###########################################################
    # Function that runs the decision tree algorithm          #
    # inputs :                                                #
    #           verbose = output to the console?              #
    # outputs:                                                #
    #           calls accuracy() to return test set accuracy  #
    ###########################################################

    def decision_tree(self, verbose = 0):
        # optimizing over type of criterion and max depth of tree search
        param_grid = [{'criterion': ['gini','entropy'], 'max_depth': [1,2,3,4,5,6,7,8,9,10]}]
        tree = DecisionTreeClassifier(random_state=0)
        best_model, prospective_models = self.best_model(tree, param_grid, verbose)   # get the best model

        # print the best parameters
        print("Parameters that gives the highest accuracy achieved for decision trees ",
        prospective_models.best_params_)

        return self.accuracy(best_model, verbose)  # return the best test set accuracy

    ###########################################################
    # Function that runs the random forest algorithm          #
    # inputs :                                                #
    #           verbose = output to the console?              #
    # outputs:                                                #
    #           calls accuracy() to return test set accuracy  #
    ###########################################################

    def random_forest(self,verbose=0):
        # optimizing over type of criterion and number of decision trees
        param_grid = [{'criterion': ['gini'], 'n_estimators': [1, 4, 8, 10, 20, 40, 60, 100, 200]}]
        forest = RandomForestClassifier(random_state=0)
        best_model, prospective_models = self.best_model(forest, param_grid, verbose)   # get the best model

        # print the best parameters
        print("Parameters that gives the highest accuracy achieved for random forest ",
        prospective_models.best_params_)

        return self.accuracy(best_model,  verbose)   # return the best test set accuracy

    ###########################################################
    # Function that runs the KNN algorithm                    #
    # inputs :                                                #
    #           verbose = output to the console?              #
    # outputs:                                                #
    #           calls accuracy() to return test set accuracy  #
    ###########################################################

    def k_nearest_neighbour(self,verbose=0):
        # optimizing over number of neighbors and type of distance measurement technique
        param_grid = [{'n_neighbors': [5,10,15,20,25,30,35,40,45,50], 'p': [1,2]}]
        knn = KNeighborsClassifier(metric='minkowski')  # initialising the classifier of K nearest neighbour
        best_model, prospective_models = self.best_model(knn, param_grid, verbose)  # get the best model

        # print the best parameters
        print("Parameters that gives the highest accuracy achieved for KNN ",
        prospective_models.best_params_)

        return self.accuracy(best_model,  verbose)   # return the best test set accuracy


#######################################################################################################################
# Start of the main program...                                                                                        #
#######################################################################################################################


if __name__ == "__main__":

    model = Model(X, y)                                                  # instantiate the class with dataset

    p_acc = model.perceptron()                                           # best accuracy for perceptron
    lr_acc = model.logistic_regression()                                 # best accuracy for logistic regression
    svm_acc = model.support_vector_machine()                             # best accuracy for SVM
    tree_acc = model.decision_tree()                                     # best accuracy for decision tree
    rf_acc = model.random_forest()                                       # best accuracy for random forest
    knn_acc = model.k_nearest_neighbour()                                # best accuracy for KNN

    accuracies = [p_acc, lr_acc, svm_acc, tree_acc, rf_acc, knn_acc]     # list of all the accuracies

    # dictionary of algorithms with their corresponding test accuracies
    accuracy_dict = {"perceptron": p_acc, "logistic regression": lr_acc, "support vector machine": svm_acc,\
                     "decision tree": tree_acc, "random forest": rf_acc, "k nearest neighbour": knn_acc}

    for key, value in accuracy_dict.items():                             # prints max accuracy to the console
        print("max accuracy for", key, value)