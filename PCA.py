from array import *
import numpy as np



from numpy import linalg as LA
from sklearn.decomposition import PCA as sklearnPCA

def CovarianceFunction(x):
    global Centered
    global Covariance
    Mean = np.mean(x, axis=0)
    #print 'Mean: ',Mean,'\n'
    Centered=x-Mean
    #print 'Centered:\n',Centered,'\n'
    #return Centered    
    Shape=x.shape[0]
    #print'Length:\n',Shape,'\n'
    DotProduct=np.dot(Centered.T,Centered)
    Covariance=DotProduct/Shape
    #print 'Function Covariance:\n',Covariance,'\n'
    return Covariance

def main():
#1
#Load in the dataset using the columns we want (exclude the class label)
    Magic = np.loadtxt('magic04.txt', delimiter=',', usecols = (0,1,2,3,4,5,6,7,8,9))#Select the columns we want from the data file
    print 30*'-'
    print'PROBLEM 1'
    print 30*'-'


#1.1 Write a function to compute covariance matrix
    CovarianceFunction(Magic)
    #Compute the Covariance of the Centered Data using built in function
    NumpyConversion=np.cov(Centered, rowvar=0,bias=True)
#1.2 Show that the result from your function matches the one using the numpy.cov function
    if np.all(NumpyConversion) == np.all(Covariance):
        print'\nNumpy Covariance matches Function Covariance\n'
    else:
        print'\nNumpy Covariance does not match Function Covariance'
    


        
#2
    print 30*'-'
    print'PROBLEM 2'
    print 30*'-'
#2.1 Displaying the first two dominant eigenvectors:
    Eigens = np.dot(Centered.T, Centered)#Square up the Centered data matrix in order to find the Eigenvalues and Eigenvectors
    eig_vals, eig_vec = LA.eig(Eigens)#Assign the eigenvalues and vectors to variables
    Top2 = eig_vec[:2]
    print'\nFirst two dominant eigenvectors:\n', Top2
#2.2 Compute the projection of data points on the subspace spanned by these two E-Vects
    SubspaceProjection = Centered.dot(Top2.T)
    print '\nSubspace Projection:\n', SubspaceProjection,'\n'
#2.3 Compute the variance of the datapoints in the projected subspace using the Subroutine that you wrote for question 1
    CovarianceFunction(SubspaceProjection)
#2.4(Do not print the projected datapoints on stdout, only print the value of the variance
    print '\nVariance of Datapoints:\n', np.var(Centered)



    
    print 30*'-'   
#3. Use linalg.eig to find all the E-Vectors, and print the covariance matrix in its E-Decomposition from (U A U^transpose)
    print'PROBLEM 3'
    print 30*'-'
#3.1 Find All Eigenvectors
    #See line 43
#3.2Compute the Covariance matrix in eigendecompositon form...
    #Take Eigenvectors times Lambda diagonal matrix of Eigenvalues
    Diag = np.diag(eig_vals)
    #print'Diagonal Matrix:\n',Diag,'\n'
    #Now, mutliply Lambda Diagonal Matrix times the EigenVectors
    First_Decomposition_Multiplication=np.dot(eig_vec,Diag)
    #Take product from above times the Transposed EigenVectors
    Second_Decomposition_Multiplication=np.dot(First_Decomposition_Multiplication, eig_vec.T)
    #print the covariance in its EigenDecomposition form
    print'\nCovariance Matrix Sigma in EigenDecomposition Form:\n', Second_Decomposition_Multiplication
    PrincipalComponentAnalysis(Magic)









#4. Write a subroutine to implement PCA Algorithm:
def PrincipalComponentAnalysis(x):
    print 30*'-'
    print'PROBLEM 4 & 5'
    print 30*'-'    
    #4.1 Compute Mean
    Mean = np.mean(x, axis=0)
    #4.2 Center the Data
    Centered=x-Mean
    #4.3 Compute Covariance Matrix   
    Shape=x.shape[0]
    DotProduct=np.dot(Centered.T,Centered)
    Covariance=DotProduct/Shape
    #4.4/4.5 Compute Eigenvalues/Eigenvectors
    Eigens = np.dot(Centered.T, Centered)#Square up the Centered data matrix in order to find the Eigenvalues and Eigenvectors
    eig_vals, eig_vec = LA.eig(Eigens)
    #Print out eigenvalues in descending order:
    eig_pairs = [((eig_vals[i]), eig_vec[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    '''print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])'''
    #4.6 Fraction of Total Variance / Find 90%     
    PreservedData = 90
    PreservedData = PreservedData*.01
    #print PreservedData
    EigenValuesKept=(eig_vals[0:np.flatnonzero((eig_vals.cumsum()/sum(eig_vals))>PreservedData)[0]+1])
    #print'\n', VarianceKept
    #you need the eigenvectors that correspond to those eigenvalues
    #Then (those vectors^transposed)dot(Centered)
    
    print'Eigenvalues to maintain 90%\n:',EigenValuesKept[:]
    EigenVectsRetained=[(eig_vec[:,i]) for i in range(len(EigenValuesKept))]
    print'\nEigenvectors Retained:\n',EigenVectsRetained
    #print'\nPairs:',eig_pairs
    #Compute Subspace
    ReducedDimensionalityData = np.dot(EigenVectsRetained, Centered.T)
    print'\nReduced Dimensionality Data: \n',ReducedDimensionalityData.T
    print'\nFirst 10 Data Points: \n', ReducedDimensionalityData.T[:10]

    '''
    #Compare it to the built-in PCA funcion for sci-kit learn:
    sklearn_pca = sklearnPCA(n_components=4)
    Y_sklearn = sklearn_pca.fit_transform(Centered)
    print'\nY_Sklearn:\n',Y_sklearn[:]
    if np.all(ReducedDimensionalityData.T) == np.all(Y_sklearn):
        print'The reduced dimensionality is the same!'
    else:
        print'The reducent dimensionality is not the same'
    
    
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    print cum_var_exp
    ''' 
main()

