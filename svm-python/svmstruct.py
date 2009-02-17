"""A module that SVM^python interacts with to do its evil bidding."""

svmpython_parameters = {'index_from_one':False}

def parse_struct_parameters(sparm):
    """Sets attributes of sparm based on command line arguments.
    
    This gives the user code a chance to change sparm based on the
    custom command line arguments.  The command line arguments are
    stored in sparm.argv as a list of strings.  The command line
    arguments have also been preliminarily processed as sparm.argd as
    a dictionary.  For example, if the custom command line arguments
    were '--key1 value1 --key2 value2' then sparm.argd would equal
    {'key1':'value1', 'key2':'value2'}.  This function returns
    nothing.  It is called only during learning, not classification.

    If this function is not implemented, any custom command line
    arguments (aside from --m, of course) are ignored and sparm remains
    unchanged."""
    sparm.arbitrary_parameter = 'I am an arbitrary parameter!'

def read_struct_examples(filename, sparm):
    """Reads and returns x,y example pairs from a file.
    
    This reads the examples contained at the file at path filename and
    returns them as a sequence.  Each element of the sequence should
    be an object "e" where e[0] and e[1] is the pattern (x) and label
    (y) respectively.  Specifically, the intention is that the element
    be a two-element tuple containing an x-y pair."""
    # We're not actually reading a file in this sample binary
    # classification task, but rather just returning a contrived
    # problem for learning.  The correct hypothesis would obviously
    # tend to have a positive weight for the first feature, and a
    # negative weight for the 4th feature.
    return [([1,1,0,0], 1), ([1,0,1,0], 1), ([0,1,0,1],-1),
            ([0,0,1,1],-1), ([1,0,0,0], 1), ([0,0,0,1],-1)]

def init_struct_model(sample, sm, sparm):
    """Initializes the learning model.
    
    Initialize the structure model sm.  The major intention is that we
    set sm.size_psi to the number of features.  The ancillary purpose
    is to add any information to sm that is necessary from the user
    code perspective.  This function returns nothing."""
    # In our binary classification task, we've encoded a pattern as a
    # list of four features.  We just want a linear rule, so we have a
    # weight corresponding to each feature.  We also add one to allow
    # for a last "bias" feature.
    sm.size_psi = len(sample[0][0])+1

def init_struct_constraints(sample, sm, sparm):
    """Initializes special constraints.

    Returns a sequence of initial constraints.  Each constraint in the
    returned sequence is itself a sequence with two items (the
    intention is to be a tuple).  The first item of the tuple is a
    document object, with at least its fvec attribute set to a support
    vector object, or list of support vector objects.  The second item
    is a number, indicating that the inner product of the feature
    vector of the document object with the linear weights must be
    greater than or equal to the number (or, in the nonlinear case,
    the evaluation of the kernel on the feature vector with the
    current model must be greater).  This initializes the optimization
    problem by allowing the introduction of special constraints.
    Typically no special constraints are necessary.

    Note that the docnum attribute of each document returned by the
    user is ignored.  Also, regarding the slackid of each document,
    the slack IDs 0 through len(sample)-1 are reserved for each
    training example in the sample.  Note that if you leave the
    slackid of a document as None, which is the default for
    svmlight.create_doc, that the document encoded as a constraint
    will get slackid=len(sample)+i, where i is the position of the
    constraint within the returned list.

    If this function is not implemented, it is equivalent to returning
    an empty list, i.e., no constraints."""
    # Return some really goofy constraints!  Normally, if the SVM is
    # allowed to converge normally, the second and fourth features are
    # 0 and -1 respectively for sufficiently high C.  Let's make them
    # be greater than 1 and 0.2 respectively!!  Both forms of a
    # feature vector (sparse and then full) are shown.
    import svmlight
    c = svmlight.create_svector
    d = svmlight.create_doc
    return [(d(c([(1,1)])),1), (d(c([0,0,0,1]),5000000),.2)]

def classify_struct_example(x, sm, sparm):
    """Given a pattern x, return the predicted label."""
    # Believe it or not, this is a dot product.  The last element of
    # sm.w is assumed to be the weight associated with the bias
    # feature as explained earlier.
    return sum([i*j for i,j in zip(x,sm.w[:-1])]) + sm.w[-1]

def find_most_violated_constraint(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.
    
    Returns the label ybar for pattern x corresponding to the most
    violated constraint according to SVM^struct cost function.  To
    find which cost function you should use, check sparm.loss_type for
    whether this is slack or margin rescaling (1 or 2 respectively),
    and check sparm.slack_norm for whether the slack vector is in an
    L1-norm or L2-norm in the QP (1 or 2 respectively).  If there's no
    incorrect label, then return None.

    If this function is not implemented, this function is equivalent
    to 'classify(x, sm, sparm)'.  The guarantees of optimality of
    Tsochantaridis et al. no longer hold since this doesn't take the
    loss into account at all, but it isn't always a terrible
    approximation, and indeed impiracally speaking on many clustering
    problems I have looked at it doesn't yield a statistically
    significant difference in performance on a test set."""
    # In the case of binary classification, there's only one wrong
    # possible classification of course!
    return -y

def psi(x, y, sm, sparm):
    """Return a feature vector describing pattern x and label y.
    
    This returns a sequence representing the feature vector describing
    the relationship between a pattern x and label y.  What psi is
    depends on the problem.  Its particulars are described in the
    Tsochantaridis paper.  The return value should be either a support
    vector object of the type returned by svmlight.create_svector, or
    a list of such objects."""
    # In the case of binary classification, psi is just the class (+1
    # or -1) times the feature vector for x, including that special
    # constant bias feature we pretend that we have.
    import svmlight
    thePsi = [0.5*y*i for i in x]
    thePsi.append(0.5*y) # Pretend as though x had an 1 at the end.
    return svmlight.create_svector(thePsi)

def loss(y, ybar, sparm):
    """Return the loss of ybar relative to the true labeling y.
    
    Returns the loss for the correct label y and the predicted label
    ybar.  In the event that y and ybar are identical loss must be 0.
    Presumably as y and ybar grow more and more dissimilar the
    returned value will increase from that point.  sparm.loss_function
    holds the loss function option specified on the command line via
    the -l option.

    If this function is not implemented, the default behavior is to
    perform 0/1 loss based on the truth of y==ybar."""
    # If they're the same sign, then the loss should be 0.
    if y*ybar > 0: return 0
    return 1

def print_struct_learning_stats(sample, sm, cset, alpha, sparm):
    """Print statistics once learning has finished.
    
    This is called after training primarily to compute and print any
    statistics regarding the learning (e.g., training error) of the
    model on the training sample.  You may also use it to make final
    changes to sm before it is written out to a file.  For example, if
    you defined any non-pickle-able attributes in sm, this is a good
    time to turn them into a pickle-able object before it is written
    out.  Also passed in is the set of constraints cset as a sequence
    of (left-hand-side, right-hand-side) two-element tuples, and an
    alpha of the same length holding the Lagrange multipliers for each
    constraint.

    If this function is not implemented, the default behavior is
    equivalent to:
    'print [loss(e[1], classify(e.[0], sm, sparm)) for e in sample]'."""
    print [loss(e[1], classify_struct_example(e[0], sm, sparm), sparm)
           for e in sample]

def print_struct_testing_stats(sample, sm, sparm, teststats):
    """Print statistics once classification has finished.
    
    This is called after all test predictions are made to allow the
    display of any summary statistics that have been accumulated in
    the teststats object through use of the eval_prediction function.

    If this function is not implemented, the default behavior is
    equivalent to 'print teststats'."""
    print teststats

def eval_prediction(exnum, x, y, ypred, sm, sparm, teststats):
    """Accumulate statistics about a single training example.
    
    Allows accumulated statistics regarding how well the predicted
    label ypred for pattern x matches the true label y.  The first
    time this function is called teststats is None.  This function's
    return value will be passed along to the next call to
    eval_prediction.  After all test predictions are made, the last
    value returned will be passed along to print_testing_stats.

    If this function is not implemented, the default behavior is
    equivalent to initialize teststats as an empty list on the first
    example, and thence for each prediction appending the loss between
    y and ypred to teststats, and returning teststats."""
    if exnum==0: teststats = []
    print 'on example',exnum,'predicted',ypred,'where correct is',y
    teststats.append(loss(y, ypred, sparm))
    return teststats

def write_struct_model(filename, sm, sparm):
    """Dump the structmodel sm to a file.
    
    Write the structmodel sm to a file at path filename.

    If this function is not implemented, the default behavior is
    equivalent to 'pickle.dump(sm, file(filename,'w'))'."""
    import pickle
    print sm.w
    f = file(filename, 'w')
    pickle.dump(sm, f)
    f.close()

def read_struct_model(filename, sparm):
    """Load the structure model from a file.
    
    Return the structmodel stored in the file at path filename, or
    None if the file could not be read for some reason.

    If this function is not implemented, the default behavior is
    equivalent to 'return pickle.load(file(filename))'."""
    import pickle
    return pickle.load(file(filename))

def write_label(fileptr, y):
    """Write a predicted label to an open file.
    
    Called during classification, the idea is to write a string
    representation of y to the file fileptr.  Note that unlike other
    functions, fileptr an actual open file, not a filename.  It is not
    to be closed by this function.  Any attempt to close it is
    ignored.

    If this function is not implemented, the default behavior is
    equivalent to 'fileptr.write(repr(y)+'\\n')'."""
    fileptr.write(repr(y)+'\n')

def print_struct_help():
    """Help printed for badly formed CL-arguments when learning.

    If this function is not implemented, the program prints the
    default SVM^struct help string as well as a note about the use of
    the --m option to load a Python module."""
    print """Help!  I need somebody.  Help!  Not just anybody.
    Help!  You know, I need someone.  Help!"""
