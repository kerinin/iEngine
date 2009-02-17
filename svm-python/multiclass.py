"""A module for SVM^python for multiclass learning."""

# The svmlight package lets us use some useful portions of the C code.
import svmlight

# These parameters are set to their default values so this declaration
# is technically unnecessary.
svmpython_parameters = {'index_from_one':True}

def read_struct_examples(filename, sparm):
    # This reads example files of the type read by SVM^multiclass.
    examples = []
    sparm.num_features = sparm.num_classes = 0
    # Open the file and read each example.
    for line in file(filename):
        # Get rid of comments.
        if line.find('#'): line = line[:line.find('#')]
        tokens = line.split()
        # If the line is empty, who cares?
        if not tokens: continue
        # Get the target.
        target = int(tokens[0])
        sparm.num_classes = max(target, sparm.num_classes)
        # Get the features.
        tokens = [tuple(t.split(':')) for t in tokens[1:]]
        features = [(int(k),float(v)) for k,v in tokens]
        if features:
            sparm.num_features = max(features[-1][0], sparm.num_features)
        # Add the example to the list
        examples.append((features, target))
    # Print out some very useful statistics.
    print len(examples),'examples read with',sparm.num_features,
    print 'features and',sparm.num_classes,'classes'
    return examples

def loss(y, ybar, sparm):
    # We use zero-one loss.
    if y==ybar: return 0
    return 1

def init_struct_model(sample, sm, sparm):
    # In the corresponding C code, the counting of features and
    # classes was done in the model initialization, not here.
    sm.size_psi = sparm.num_features * sparm.num_classes
    print 'size_psi set to',sm.size_psi

def classify_struct_example(x, sm, sparm):
    # I am a very bad man.  There is no class 0, of course.
    return find_most_violated_constraint(x, 0, sm, sparm)

def find_most_violated_constraint(x, y, sm, sparm):
    # Get all the wrong classes.
    classes = [c+1 for c in range(sparm.num_classes) if c+1 is not y]
    # Get the psi vectors for each example in each class.
    vectors = [(psi(x,c,sm,sparm),c) for c in classes]
    # Get the predictions for each psi vector.
    predictions = [(svmlight.classify_example(sm, p),c) for p,c in vectors]
    # Return the class associated with the maximum prediction!
    return max(predictions)[1]

def psi(x, y, sm, sparm):
    # Just increment the feature index to the appropriate stack position.
    return svmlight.create_svector([(f+(y-1)*sparm.num_features,v)
                                    for f,v in x])

# The default action of printing out all the losses or labels is
# irritating for the 300 training examples and 2200 testing examples
# in the sample task.
def print_struct_learning_stats(sample, sm, cset, alpha, sparm):
    predictions = [classify_struct_example(x,sm,sparm) for x,y in sample]
    losses = [loss(y,ybar,sparm) for (x,y),ybar in zip(sample,predictions)]
    print 'Average loss:',float(sum(losses))/len(losses)

def print_struct_testing_stats(sample, sm, sparm, teststats): pass
