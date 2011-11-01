# Implementation of Pegasos Algorithm
# Described in Paper: http://www.cs.huji.ac.il/~shais/papers/ShalevSiSrCo10.pdf

import os
import math
import cPickle

# There are two vector types available: special_vector
# and normal_vector. Additional ones can be built as long
# as they implement the necessary functions.


# special_vector is especially well suited to
# sparse datasets or those where features may be 
# unordered. It uses dictionaries, and does not
# store zero valued components which is a Critical Optimization
# optimization that can save a tremendous amount of memory
class special_vector(dict):
    def add(self, component, value):
        if(value): self[component] = value

    @staticmethod
    def zeros(_Number):
        return special_vector()

    @staticmethod
    def dotproduct(vector1, vector2):
        if(len(vector1) < len(vector2)): smallvector, largevector = vector1, vector2
        else: smallvector, largevector = vector2, vector1

        accumulator = 0
        for key in smallvector.keys():
            if key in largevector:				
                accumulator += smallvector[key]*largevector[key]
        return accumulator

    @staticmethod
    def subtract(vector1, vector2):
        if(len(vector1) < len(vector2)): smallvector, largevector = vector1, vector2
        else: smallvector, largevector = vector2, vector1

        newvector = special_vector()
        for key in smallvector.keys():
            if key in largevector:
                newvector.add(key, vector1[key] - vector2[key])
        return newvector

    # L^2 Norm
    def norm(self):
        accumulator = 0
        for key in self.keys():
            accumulator += self[key]*self[key]
        return math.sqrt(accumulator) 

    def setcomponent(self, component, value):
        self.add(component, value)

    def getcomponent(self, component):
        try: return self[component]
        except KeyError: return 0

    def getnonzeroindices(self):
        return self.keys()

    def scalarproduct(self, value):
        self = map(lambda key: self[key]*value, self.keys())


# Traditional Vector is the more intuitive approach
# to representing vectors, the operations are
# straightforward.
class traditional_vector(list):
    def add(self, component, value):
        self.append(value)

    @staticmethod
    def zeros(number):
        return traditionalvector([0]*100)

    @staticmethod
    def dotproduct(vector1, vector2):
        return reduce(lambda accumulator, newitem: accumulator + newitem[0]*newitem[1], zip(vector1, vector2), 0)

    @staticmethod
    def subtract(vector1, vector2):
        return map(lambda x, y: x-y, zip(vector1, vector2))

    def norm(self):
        return math.sqrt(reduce(lambda accumulator, newitem: accumulator+x*x, self, 0))

    def setcomponent(self, component, value):
        try: self[component] = value
        except IndexError: raise

    def getcomponent(self, component):
        try: return self[component]
        except IndexError: return 0

    def getnonzeroindices(self):
        return filter(lambda x: self[x]!=0, range(len(self)))

    def scalarproduct(self, value):
        self = map(lambda i: value*i, self)


# Primary Computation of Gram Matrix can be done in Parallel, as operations
# are independent one another
class gram_matrix(object):
    def __init__(self): self.data = dict()

    def save(self, filename):
        if os.path.isfile(filename):
            overwrite = raw_input("A file with the name %s already exists. Would you like to Overwrite it? (y/n) " % filename)
            if(overwrite.lower() == "y"):
                print("Overwriting File %s" % filename)
            else:
                print "Gram Matrix not Saved, as File with Same Name exists"
                return		
        with open(filename, "wb") as fh:
            cPickle.dump(self.data, fh)

    def load(self, filename, force = False):
        if os.path.isfile(filename):
            with open(filename, "rb") as fh:
                try:
                    loadeddata = cPickle.load(fh)
                    if(force):
                        self.data = loadeddata
                        return True
                    elif self.verify(loadeddata): 
                        self.data = loadeddata
                        return True
                    else: print "The Format of the Data was Illegal, Please Rebuild A Gram Matrix and Resave It."
                except EOFError: return False
        return False

    def verify(self, data):
        if not isinstance(data, dict): return False
        for key in data.keys():
            if not isinstance(key, tuple): return False
            if len(key) != 2: return False
            if not isinstance(data[key], float): return False
        return True

    def compute(self, TrainingSamples, Kernel):
        for i in range(len(TrainingSamples)):
            for j in range(i, len(TrainingSamples)):		
                self.data[(i,j)] = Kernel(TrainingSamples[i], TrainingSamples[j])

    def clear(self): self.data = dict()

    def query(self, (i,j)):
        if (i>j): return self.data[(j,i)]
        else: return self.data[(i,j)]

    def get(self, numberofsamples, jvalue):
        vector = special_vector()
        for i in range(numberofsamples):
            vector.add(i, self.query((i, jvalue)))
        return vector
		



# Reading a File and Loading the Vectors
# Filename		- Name of File Input	-String
# Does Heavy Checks on the Validity of the Data
# Each line should be of the following form:
# Class Label (Either 1 or -1) Component1:Value1 ... Componentn:Valuen

# For instance the training file may be "training":
# -1 182:0.890196 183:0.992157 184:0.988235 185:0.937255 186:0.913725
# -1 183:0.376471 184:0.741176 185:0.984314 186:0.984314 187:0.992157
# 1 183:0.301961 184:0.713725 185:1 186:0.996078
# 1 153:0.72549 154:0.72549

# The Testing file has the same format as the training file

def read(filename):
    assert isinstance(filename, str), "Filename is not a String Type"
    assert os.path.isfile(filename), "File with name %s was not found in Current Directory" % filename
    vectors = list()
    classlabels = list()
    with open(filename) as fh:		
        for linenum, line in enumerate(fh.xreadlines()):
            data = line.split()
            if(len(data)):
                linedictionary = special_vector()
                classlabel = data[0]
                if(classlabel != "1" and classlabel != "-1"):
                    print "Skipping Line %d in File %s due to Illegal Class Label, Must be either 1, or -1" % (linenum, filename)
                    continue
                classlabel = int(classlabel)
                for datapoint in data[1:]:
                    try: 
                        component, value = datapoint.split(":")
                        component = int(component)
                        value = float(value)		
                    except ValueError as error:
                        print "Skipping Line Line %d in File %s due to Incorrectly formatted Datapoint '%s'. Must be of the Form Component:Value, such as 27:.0201, where component is an integer and value any floating point number." % (linenum, filename, datapoint)
                        print "Recieved Error", error
                        continue
                    linedictionary.add(component,value)
                vectors.append(linedictionary)
                classlabels.append(classlabel)
    return vectors, classlabels


# These are Essentially Wrappers with Paramaters
# Which will Return the Corresponding Kernel Functions
def radial_basis(gamma):
    assert gamma>0 , "Parameter in Radial Basis function is not a float > 0"
    def function(vector1, vector2):
        new_vector = special_vector().subtract(vector1, vector2)
        return math.exp(-1 * gamma * new_vector.norm() ** 2)
    return function

def homogeneous_polynomial(degree):
    assert isinstance(degree, int) and degree>0, "Parameter in Homogeneous Polynomial function is not an integer > 0" 
    def function(vector1, vector2):
        value = special_vector().dotproduct(vector1, vector2)		
        return value ** degree
    return function

def inhomogeneous_polynomial(degree):
    assert isinstance(degree, int) and degree>0, "Parameter in Homogeneous Polynomial function is not an integer > 0" 
    def function(vector1, vector2):
        value = special_vector().dotproduct(vector1, vector2) + 1
        return value ** degree
    return function

def linear(): return homogeneous_polynomial(1)

def hyperbolic_tangent(kappa, c):
    assert kappa>0 and c<0, "Parameter in Hyperbolic Tangent Function is not Valid, kappa must be > 0 and c must be < 0"
    def function(vector1, vector2):
        value = special_vector().dotproduct(vector1, vector2)
        return math.tanh(kappa*value+c)
    return function


# Entry Into the SVM Solver
# TrainingFilename  -   Name of File Input                  - String
# TestingFilename   -   Name of File Input                  - String
# Kernel            -   Kernel Funtion                      - Function Returing Integer Type
# Iterations        -   Iterations for Gradient Descent     - Integer
# Eta               -   Learning Rate                       - Float
# GramFile          -   Filename to Save the Gram Matrix    - String
#                   -   This is optional, but will speed
#                   -   up future classification since
#                   -   the computation of the matrix is costly
# SupportVecFile    -   File to Save Support Vectors. Saving,
#                   -   is useful, since it can be a costly operation
#                   -   to constantly find the support vectors
def main(TrainingFilename, TestingFilename, Kernel, Iterations, eta = .001, GramFile = "", SupportVecFile = ""):
    TrainingSamples, TrainingLabels=read(TrainingFilename)
    print("Loaded Training Samples")

    # Testing Code
    #for sample, label in zip(TrainingSamples, TrainingLabels):
    #	print sample, len(sample), label
    #input("Continue?")

    TestingSamples, TestingLabels=read(TestingFilename)
    print("Loaded Testing Samples")

    # Once we compute the Gram Matrix Once, We may
    # Pickle it so that we don't have to recompute
    # the Data Structure because Computing the Gram
    # Matrix is a generally costly operation
	
    print("Computing Gram Matrix")
    GramMatrix = gram_matrix()
    if(GramFile):
        if GramMatrix.load(GramFile, force = True): print "Loaded Gram Matrix Successfully"
        else:
            GramMatrix.compute(TrainingSamples, Kernel)
            print("Saving New Gram Matrix")
            GramMatrix.save(GramFile)
    else: GramMatrix.compute(TrainingSamples, Kernel)
    print("Computed Gram Matrix")

    # Testing Code
    #for vecnumi in range(len(TrainingSamples)):
    #	for vecnumj in range(len(TrainingSamples)):
    #		print str(vecnumi)+":"+str(vecnumj)+"\t"+str(GramMatrix.query((vecnumi,vecnumj)))
    #
    #input("Continue?")


    # Apply Gradient Descent SVM Solver and Generate
    # Necessary Support Vectors
    print("Computing %d Iterations of Pegasos" % Iterations)
    Coeffecients, SupportVectors=Pegasos(TrainingSamples, TrainingLabels, eta, Iterations, GramMatrix)
    print('Completed Pegasos')

    if(SupportVecFile and isinstance(SupportVecFile, str)):
        with open(SupportVecFile, "wb") as fh:
            cPickle.dump((Coeffecients, SupportVectors), fh)

    # Run Tests using the Support Vectors For Classification
    error=RunTests(Coeffecients, SupportVectors, Kernel, TestingSamples, TestingLabels)
    print("Error Value was %s" % str(error))


def Pegasos(TrainingSamples, TrainingLabels, eta, Iterations, GramMatrix):
    samples = len(TrainingSamples)
    a = special_vector().zeros(samples)
    time = 1
    for i in range(Iterations):
        print "Iteration %d Started" % i
        for tau in range(len(TrainingSamples)):
            wx = special_vector.dotproduct(a, GramMatrix.get(len(TrainingSamples), tau))
            a.scalarproduct((1-1/time))
            if(TrainingLabels[tau]*wx < 1):
                a.setcomponent(tau, a.getcomponent(tau) + float(TrainingLabels[tau])/float(eta*time))
            time += 1
    non_zero_indices = a.getnonzeroindices()
    a = [a.getcomponent(index) for index in non_zero_indices]
    SV = [TrainingSamples[index] for index in non_zero_indices] 
    return a, SV


# Assigns a Class Label to a Particular Sample
def Predictor(a, SV, Kernel, Sample):
    accumulator = 0	
    for index in range(len(a)):
        accumulator += a[index]*Kernel(SV[index], Sample)
    if accumulator < 0: return -1
    return 1

def RunTests(a, SV, Kernel, TestingSamples, TestingLabels):
    errors = 0
    for Sample, correctlabel in zip(TestingSamples, TestingLabels):
        predictedlabel=Predictor(a, SV, Kernel, Sample)
        if(predictedlabel*correctlabel<0): errors += 1
    if len(TestingSamples): return float(errors)/float(len(TestingSamples))
    return 0

if __name__ == "__main__":
    trainingfile = "training"
    testingfile = "testing"
    main(trainingfile, testingfile, linear(), 5, GramFile = "GramMatrix", SupportVecFile = "Supports")
