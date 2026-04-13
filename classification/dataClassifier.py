# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

'''
We used AI tools to assist with understanding concepts and debugging code.
'''

# This file contains feature extraction methods and harness
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import perceptron_pacman
import mira
import samples
import sys
import util
from pacman import GameState

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    Enhanced features for digit classification:

    1. Hole Count Features (zero_holes, one_hole, two_holes):
       - Counts enclosed white regions (topological holes) within the digit
       - Applies BFS/flood fill from borders to identify interior holes
       - Distinguishes digits like 0,6,8,9 (multiple holes) from 1,2,3,4,5,7 (no holes)
       - Encoded as 3 binary features (mutually exclusive categories)

    2. High Density Feature:
       - Measures the ratio of filled pixels to total pixels
       - Binary feature: 1 if density > 25%, 0 otherwise
       - Separates thick digits (0,8,6,9) from thin digits (1,7)

    These features complement the pixel-level features by capturing structural
    properties that distinguish digit shapes, particularly differences in topology
    (holes) and overall ink density.

    ##
    """
    features =  basicFeatureExtractorDigit(datum)

    "*** YOUR CODE HERE ***"

    # Helper to get pixel value (0 or 1)
    def getP(x, y):
        return datum.getPixel(x, y) > 0

    # 1. Count Connected Components (Holes)
    
    visited = set()
    num_regions = 0
    
    # Iterate through the datum including a 1-pixel border to ensure 
    # the outer background is counted as a single connected component.
    for x in range(-1, DIGIT_DATUM_WIDTH + 1):
        for y in range(-1, DIGIT_DATUM_HEIGHT + 1):
            if (x, y) not in visited:
                # If it's a white pixel (or outside the bounds)
                is_pixel_off = False
                if 0 <= x < DIGIT_DATUM_WIDTH and 0 <= y < DIGIT_DATUM_HEIGHT:
                    if datum.getPixel(x, y) == 0:
                        is_pixel_off = True
                else:
                    is_pixel_off = True # The area outside the digit is "white"
                
                if is_pixel_off:
                    num_regions += 1
                    # BFS to mark the entire region
                    queue = [(x, y)]
                    visited.add((x, y))
                    while queue:
                        curr_x, curr_y = queue.pop(0)
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            next_x, next_y = curr_x + dx, curr_y + dy
                            if -1 <= next_x <= DIGIT_DATUM_WIDTH and \
                               -1 <= next_y <= DIGIT_DATUM_HEIGHT:
                                if (next_x, next_y) not in visited:
                                    # Check if the neighbor is also white space
                                    is_neighbor_off = False
                                    if 0 <= next_x < DIGIT_DATUM_WIDTH and \
                                       0 <= next_y < DIGIT_DATUM_HEIGHT:
                                        if datum.getPixel(next_x, next_y) == 0:
                                            is_neighbor_off = True
                                    else:
                                        is_neighbor_off = True
                                    
                                    if is_neighbor_off:
                                        visited.add((next_x, next_y))
                                        queue.append((next_x, next_y))

    # Binary features for number of holes
    features['zero_holes'] = 1 if num_regions == 1 else 0
    features['one_hole'] = 1 if num_regions == 2 else 0
    features['two_holes'] = 1 if num_regions >= 3 else 0

    # 2. Ratio of filled pixels (Symmetry/Density)
    total_pixels = DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT
    on_pixels = sum(features[(x, y)] for x in range(DIGIT_DATUM_WIDTH) for y in range(DIGIT_DATUM_HEIGHT))
    features['high_density'] = 1 if (float(on_pixels) / total_pixels) > 0.25 else 0

    return features


def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter
    return features, state.getLegalActions()

def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **enhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()

def enhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    successor = state.generateSuccessor(0, action)
    currentPos = state.getPacmanPosition()
    successorPos = successor.getPacmanPosition()

    food = state.getFood()
    successorFood = successor.getFood()
    capsules = state.getCapsules()
    successorCapsules = successor.getCapsules()
    ghostStates = state.getGhostStates()

    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # 1. Stop action
    features['is_stop'] = 1 if action == 'Stop' else 0

    # 2. Food features
    foodList = food.asList()
    if foodList:
        currentFoodDistances = [manhattan(currentPos, f) for f in foodList]
        successorFoodDistances = [manhattan(successorPos, f) for f in foodList]
        currentClosestFood = min(currentFoodDistances)
        successorClosestFood = min(successorFoodDistances)
        features['closestFood'] = float(successorClosestFood)
        features['foodDistanceDecreased'] = 1 if successorClosestFood < currentClosestFood else 0
        features['foodDistanceIncreased'] = 1 if successorClosestFood > currentClosestFood else 0
        features['foodDistanceSame'] = 1 if successorClosestFood == currentClosestFood else 0
        features['foodNearby'] = 1 if successorClosestFood <= 2 else 0
        features['eats_food'] = 1 if successorFood.count() < food.count() else 0
        features['foodZero'] = 1 if successorClosestFood == 0 else 0
        features['foodOne'] = 1 if successorClosestFood == 1 else 0
        features['foodTwo'] = 1 if successorClosestFood == 2 else 0
        features['foodThreePlus'] = 1 if successorClosestFood >= 3 else 0
    else:
        features['closestFood'] = 0
        features['foodDistanceDecreased'] = 0
        features['foodDistanceIncreased'] = 0
        features['foodDistanceSame'] = 0
        features['foodNearby'] = 0
        features['eats_food'] = 0
        features['foodZero'] = 0
        features['foodOne'] = 0
        features['foodTwo'] = 0
        features['foodThreePlus'] = 0

    # 3. Capsule features
    if capsules:
        currentCapsuleDistances = [manhattan(currentPos, c) for c in capsules]
        successorCapsuleDistances = [manhattan(successorPos, c) for c in capsules]
        currentClosestCapsule = min(currentCapsuleDistances)
        successorClosestCapsule = min(successorCapsuleDistances)
        features['closestCapsule'] = float(successorClosestCapsule)
        features['capsuleDistanceDecreased'] = 1 if successorClosestCapsule < currentClosestCapsule else 0
        features['capsuleDistanceIncreased'] = 1 if successorClosestCapsule > currentClosestCapsule else 0
        features['eats_capsule'] = 1 if len(successorCapsules) < len(capsules) else 0
        features['capsuleNearby'] = 1 if successorClosestCapsule <= 2 else 0
        features['capsuleZero'] = 1 if successorClosestCapsule == 0 else 0
        features['capsuleOne'] = 1 if successorClosestCapsule == 1 else 0
        features['capsuleTwoPlus'] = 1 if successorClosestCapsule >= 2 else 0
    else:
        features['closestCapsule'] = 0
        features['capsuleDistanceDecreased'] = 0
        features['capsuleDistanceIncreased'] = 0
        features['eats_capsule'] = 0
        features['capsuleNearby'] = 0
        features['capsuleZero'] = 0
        features['capsuleOne'] = 0
        features['capsuleTwoPlus'] = 0

    # 4. Ghost features
    activeGhostDistances = []
    scaredGhostDistances = []
    activeCurrentDistances = []
    scaredCurrentDistances = []

    for ghost in ghostStates:
        currentDist = manhattan(currentPos, ghost.getPosition())
        successorDist = manhattan(successorPos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            scaredGhostDistances.append(successorDist)
            scaredCurrentDistances.append(currentDist)
        else:
            activeGhostDistances.append(successorDist)
            activeCurrentDistances.append(currentDist)

    if activeGhostDistances:
        closestActive = min(activeGhostDistances)
        closestActiveCurrent = min(activeCurrentDistances)
        features['closestGhost'] = float(closestActive)
        features['ghostThreat'] = 1 if closestActive <= 2 else 0
        features['ghostVeryClose'] = 1 if closestActive <= 1 else 0
        features['ghostDistanceDecreased'] = 1 if closestActive < closestActiveCurrent else 0
        features['ghostDistanceIncreased'] = 1 if closestActive > closestActiveCurrent else 0
        features['ghostNearbyCount'] = sum(1 for d in activeGhostDistances if d <= 2)
        features['ghostCloseCount'] = sum(1 for d in activeGhostDistances if d <= 4)
    else:
        features['closestGhost'] = 10
        features['ghostThreat'] = 0
        features['ghostVeryClose'] = 0
        features['ghostDistanceDecreased'] = 0
        features['ghostDistanceIncreased'] = 0
        features['ghostNearbyCount'] = 0
        features['ghostCloseCount'] = 0

    if scaredGhostDistances:
        closestScared = min(scaredGhostDistances)
        closestScaredCurrent = min(scaredCurrentDistances)
        features['closestScaredGhost'] = float(closestScared)
        features['scaredGhostNearby'] = 1 if closestScared <= 4 else 0
        features['scaredGhostVeryClose'] = 1 if closestScared <= 2 else 0
        features['scaredGhostDistanceDecreased'] = 1 if closestScared < closestScaredCurrent else 0
        features['scaredGhostDistanceIncreased'] = 1 if closestScared > closestScaredCurrent else 0
        features['chaseScaredGhost'] = 1 if closestScared <= 4 else 0
        features['scaredGhostCountNear'] = sum(1 for d in scaredGhostDistances if d <= 4)
    else:
        features['closestScaredGhost'] = 10
        features['scaredGhostNearby'] = 0
        features['scaredGhostVeryClose'] = 0
        features['scaredGhostDistanceDecreased'] = 0
        features['scaredGhostDistanceIncreased'] = 0
        features['chaseScaredGhost'] = 0
        features['scaredGhostCountNear'] = 0

    # 5. successor state features
    features['scoreChange'] = successor.getScore() - state.getScore()
    features['isWin'] = 1 if successor.isWin() else 0
    features['isLose'] = 1 if successor.isLose() else 0
    features['nextFoodCount'] = successorFood.count()
    features['nextCapsuleCount'] = len(successorCapsules)

    return features


def contestFeatureExtractorDigit(datum):
    """
    Specify features to use for the minicontest
    """
    features =  basicFeatureExtractorDigit(datum)
    return features

def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features =  basicFeatureExtractorFace(datum)
    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(guesses)):
    #     prediction = guesses[i]
    #     truth = testLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print rawTestData[i]
    #         break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print("new features:", pix)
                continue
        print(image)

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Doing classification")
    print("--------------------")
    print("data:\t\t" + options.data)
    print("classifier:\t\t" + options.classifier)
    if not options.classifier == 'minicontest':
        print("using enhanced features?:\t" + str(options.features))
    else:
        print("using minicontest feature extractor")
    print("training set size:\t" + str(options.training))
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    elif(options.data=="faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    elif(options.data=="pacman"):
        printImage = None
        if (options.features):
            featureFunction = enhancedFeatureExtractorPacman
        else:
            featureFunction = basicFeatureExtractorPacman
    else:
        print("Unknown dataset", options.data)
        print(USAGE_STRING)
        sys.exit(2)

    if(options.data=="digits"):
        legalLabels = list(range(10))
    else:
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.smoothing <= 0:
        print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
        print(USAGE_STRING)
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
            print(USAGE_STRING)
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print("using automatic tuning for naivebayes")
            classifier.automaticTuning = True
        else:
            print("using smoothing parameter k=%f for naivebayes" %  options.smoothing)
    elif(options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
        else:
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,options.iterations)
    elif(options.classifier == "mira"):
        if options.data != 'pacman':
            classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print("using automatic tuning for MIRA")
            classifier.automaticTuning = True
        else:
            print("using default C=0.001 for MIRA")
    elif(options.classifier == 'minicontest'):
        import minicontest
        classifier = minicontest.contestClassifier(legalLabels)
    else:
        print("Unknown classifier:", options.classifier)
        print(USAGE_STRING)

        sys.exit(2)

    args['agentToClone'] = options.agentToClone

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code



def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
    
    # Load data
    numTraining = options.training
    numTest = options.test

    if(options.data=="pacman"):
        agentToClone = args.get('agentToClone', None)
        trainingData, validationData, testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agentToClone, (None, None, None))
        trainingData = trainingData or args.get('trainingData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validationData = validationData or args.get('validationData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        testData = testData or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
        rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
        rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
        rawTestData, testLabels = samples.loadPacmanData(testData, numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print("Extracting features...")
    trainingData = list(map(featureFunction, rawTrainingData))
    validationData = list(map(featureFunction, rawValidationData))
    testData = list(map(featureFunction, rawTestData))

    # Conduct training and testing
    print("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print("Validating...")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
    print("Testing...")
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print(string3)
        printImage(features_odds)

    if((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print(("=== Features with high weight for label %d ==="%l))
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)