import pandas as pd
import numpy as np
from math import log

FEATURES = {}

"""
Creates and tests decision tree based on the ID3 algorithm modified to discretize continuous values.
@author Damian Henry
@params csv (str) The filename path of the csv containing example data
        The header must the names of the features. The last column must be the classification
        The columns in the examples must align with the names in the header
@params test_prop (float) The proportion of the input data to test the tree on. Defaults to 20%
@params continuous_threshold (int) The number of unique values a given feature must have to be
        considered continuous
@params discretization_quantiles (int) The number of groupings to split continuous data
        The groups should roughly have the same frequency of elements
@params print_tree (bool) Whether to print the created tree or not
"""
def main(csv, test_prop = 0.2, continuous_threshold = 10,  discretization_quantiles = 4, print_tree = False):
    global FEATURES
    
    df = preProcessData(csv)
    train_df, test_df = splitData(df, test_prop)
    features = df.columns[:-1]
    FEATURES = dict(zip(features.tolist(), df.index.values))
    examples = train_df.to_numpy()
    
    featureSplits = getFeatureSplits(examples, continuous_threshold, discretization_quantiles)
    defaultLabel = getMajorityLabel(examples)
    tree = createDecisionTree(examples, features, featureSplits, defaultLabel)

    if print_tree:
        print("Tree:\n~~~~~~~~~~")
        print(preorder(0, tree).strip())
        print("~~~~~~~~~~\n")
    print("Accuracy: {}%".format(round(calculate_accuracy(test_df, tree), 2) * 100 ))
    print("Size of tree: {}".format(getSize(tree)))


"""
Node
@attribute feature (str) The feature it represents
@attribute isContinuous (bool) Whether the feature contains continuous values
@attribute chidlren dict<str, Node> Mapping of values of the Node's feature to Node children
"""
class Node:
    def __init__(self, feature):
        global FEATURES
        self.feature = feature
        self.isContinuous = FEATURES[feature][1] == "continuous"
        self.children = {}
        
"""
LeafNode
@attribute label (str) The label classification
"""
class LeafNode(Node):
    def __init__(self, label):
        self.label = label

"""
@author Georgia Institute of Technology
Prints out the decision tree
@params depth (int) The current depth of the tree
@params node (Node) The current node
"""
def preorder(depth,node):
    if node is None:
      return '|---'*depth+str(None)+'\n'
    if isinstance(node,LeafNode):
      return '|---'*depth+str(node.label)+'\n'
    string = ''
    for val in node.children.keys():
      childStr = '|---'*depth
      childStr += '%s = %s'%(str(node.feature),str(val))
      string+=str(childStr)+"\n"+preorder(depth+1, node.children[val])
    return string

"""
Calculates the size of the tree
@params node (Node) The current node
@returns size (int) The size of the tree
"""
def getSize(node):
    if isinstance(node, LeafNode):
      return 1
    size = 1
    for child in node.children.values():
      if child is not None:
        size += getSize(child)
    return size

"""
Reades in the data from a csv to a Pandas DataFrame.
Renames the last column to 'label' and removes and null containing examples
@params csv (str) The filename path of the csv containing example data
        The header must the names of the features. The last column must be the classification
        The columns in the examples must align with the names in the header
@returns df (Pandas DataFram) The dataframe containing the entire data
"""
def preProcessData(csv):
    df = pd.read_csv(csv)
    df.rename(columns= {df.columns[-1]: "label"}, inplace= True)
    df.dropna(how='any', inplace= True)
    return df

"""
Splits the data into training and testing sets according to a testing proportion
@params df (Pandas DataFrame) The dataframe containing the entire data
@params test_prop (float) The proportion of the input data to test the tree on.
@returns (tuple<Pandas DataFrame, Pandas Dataframe>) The training and testing dataframes
"""
def splitData(df, test_prop = 0.2):
    test_size = int(test_prop * len(df))
    df_indices = df.index.tolist()
    test_indices = np.random.choice(df_indices, test_size, replace = False)
    train_df = df.drop(test_indices)
    test_df = df.iloc[test_indices]
    return (train_df, test_df)

"""
Returns the majority label classification of the given examples
@params examples (Numpy 2d array) The example data to find the majority label of
@returns majorityLabel (str) The majority label classification
"""
def getMajorityLabel(examples):
    labelCol = examples[:, -1]
    distinctClasses, counts = np.unique(labelCol, return_counts=True)
    index = counts.argmax()
    return distinctClasses[index]

"""
Returns whether the given examples have the same label classification
@params examples (Numpy 2d array) The example data to check purity
@returns (bool) Whether the given examples have the same label classification
"""
def isSameClassification(examples):
    labelCol = examples[:, -1]
    return len(np.unique(labelCol)) == 1

"""
Returns a dictionary mapping features to their potential splits.
Discretizes continious values based on the continuous_threshold and discretization_quantiles
@params examples (Numpy 2d array) The example data the features split
@params continuous_threshold (int) The number of unique values a given feature must have to be
        considered continuous.
@params discretization_quantiles (int) The number of groupings to split continuous data.
        The groups should roughly have the same frequency of elements
@returns featureSplits (dict<str,Numpy 1d array>)  Mapping of feature names to feature categories
"""
def getFeatureSplits(examples, continuous_threshold, discretization_quantiles):
    featureSplits = {}
    global FEATURES
    for feature, index in FEATURES.items():
        col = examples[:, index]
        uniqueValues = np.unique(col)
        if len(uniqueValues) < continuous_threshold:
            featureSplits[feature] =  uniqueValues
            FEATURES[feature] = (index, "categorical")
        else:
            splits =  pd.qcut(col, discretization_quantiles, duplicates = "drop").categories.tolist()
            splits.append(pd.Interval(left = float("-inf"), right = min(splits).left, closed = "right"))
            splits.append(pd.Interval(left = max(splits).right, right = float("inf"), closed = "neither"))
            splits = np.array(splits)
            featureSplits[feature] = splits
            FEATURES[feature] = (index, "continuous")
                         
    return featureSplits

"""
Calculates the counts of each label in the given examples
@params examples (Numpy 2d array) The example data to count the labels
@returns dictionary (dict<str, int>) The mapping of label names to their counts
"""
def getLabelCounts(examples):
    labelCol = examples[:, -1]
    distinctClasses, counts = np.unique(labelCol, return_counts=True)
    return dict(zip(distinctClasses, counts))

"""
Gets all examples given a feature split
@params examples (Numpy 2d array) The example data to filter
@params feature (str) the feature to filter on
@params split (str, Interval, etc.) The value the feature should have in all examples
@returns examples (Numpy 2d array) filtered examples
"""
def getRelevantExamples(examples, feature, split):
    global FEATURES
    featureIndex, featureType = FEATURES[feature]
    col = examples[: , featureIndex]
    if featureType == "continuous":
        examples = examples[np.logical_and(col > split.left, col < split.right)]
    else:
        examples = examples[col == split]
    return examples

"""
Calculates the entropy given the counts of labels
@params labelCounts (iterable) iterable of label counts
@returns entropy (float)
"""
def getEntropy(labelCounts):
    entropy = 0
    totalCount = sum(labelCounts)
    for count in labelCounts:
        prob = float(count) / totalCount
        entropy -= (prob * log(prob, 2))
    return entropy

"""
Calculates the entropy of the data given the given the specified feature's values
@params examples (Numpy 2d array) example data to split
@params feature (str) the feature to split the data on
@params featureSplits (dict<str,Numpy 1d array>)  Mapping of feature names to feature categories
@returns remainder (float)
"""
def getRemainder(examples, feature, featureSplits):
    remainder = 0
    labelCounts = getLabelCounts(examples)
    splits = featureSplits
    for split in splits:
        relevantExamples = getRelevantExamples(examples, feature, split)
        newLabelCounts = getLabelCounts(relevantExamples)
        entropy = getEntropy(newLabelCounts.values())
        remainder += ( sum(newLabelCounts.values()) / sum(labelCounts.values()) ) * entropy
    return remainder

"""
Calculates the information gain of a certain feature
@params examples (Numpy 2d array) example data to split
@params feature (str) the feature to split the data on
@params featureSplits (dict<str,Numpy 1d array>)  Mapping of feature names to feature categories
@returns information gain of given feature and examples
"""
def infoGain(examples, feature, featureSplits):
    entropy = getEntropy(getLabelCounts(examples).values())
    remainder = getRemainder(examples, feature, featureSplits)
    return entropy - remainder
    
"""
Selects the feature best to split the data on.
Uses information gain.
@params examples (Numpy 2d array) example data to split 
@params features(Panda Index) the potential features to split on
@params featureSplits (dict<str,Numpy 1d array>)  Mapping of feature names to feature categories
@returns bestFeature (str) The best feature to split on
"""
def getBestFeature(examples, features, featureSplits):
    bestFeature= features[0]
    for feature in features:
        currentGain = infoGain(examples, feature, featureSplits[feature])
        bestGain = infoGain(examples, bestFeature, featureSplits[bestFeature])
        if currentGain > bestGain:
            bestFeature = feature
    return bestFeature

"""
Modified ID3 algorithm to discretize continuous data into equal frequency categories
@params examples (Numpy 2d array) example data the tree is based on
@params remainingFeatures (Panda Index) the remaining features to split on
@params featureSplits (dict<str,Numpy 1d array>)  Mapping of feature names to feature categories
@params defaultLabel (str) The default label if needed
"""
def createDecisionTree(examples, remainingFeatures, featureSplits, defaultLabel):
    if len(examples) == 0:
        return LeafNode(defaultLabel)
    elif isSameClassification(examples):
        return LeafNode(examples[0][-1])
    elif len(remainingFeatures) == 0:
        majorityLabel = getMajorityLabel(examples)
        return LeafNode(majorityLabel)
    else:
        bestFeature = getBestFeature(examples, remainingFeatures, featureSplits)
        remainingFeatures = remainingFeatures.drop(bestFeature)
        splits = featureSplits[bestFeature]
        root = Node(bestFeature)
        for split in splits:
            relevantExamples = getRelevantExamples(examples, bestFeature, split)
            defaultLabel = getMajorityLabel(examples)
            child = createDecisionTree(relevantExamples, remainingFeatures, featureSplits, defaultLabel)
            root.children[split]= child
        return root

"""
Classifies a problem instance
@params instance (Pandas Series) The problem instance you want to classify
@params root (Node) The current node of the decision tree
@returns classification (str) of the problem instance
"""
def classify(instance, root):
    global FEATURES
    if isinstance(root, LeafNode):
          return root.label
    else:
        feature = root.feature
        index = FEATURES[feature][0]
        if root.isContinuous:
            for interval in root.children:
                if instance[index] in interval:
                    new = root.children[interval]
                    break
        else:
            new = root.children[instance[index]]
        return classify(instance,new)
    
"""
Calculates the accuracy of the tree
@params df (Pandas DataFrame) The Pandas data frame containing the problem instances to classify
@params tree (Node) The decision tree to be tested
@returns classification rate (float) of the tree
"""
def calculate_accuracy(df, tree):
    pd.options.mode.chained_assignment = None
    df["classification"] = df.apply(classify, args=(tree,), axis=1)
    df["classification_correct"] = df["classification"] == df["label"]
    accuracy = df["classification_correct"].mean()
    return accuracy

"""
Runs main with specified csv
"""
if __name__ == "__main__":
    print("Creating and testing decision tree...\n")
    main("./datasets/test.csv", print_tree = True)



    
