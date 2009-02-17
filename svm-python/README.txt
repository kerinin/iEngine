This file briefly describes SVM^python.

--------
WHAT IT IS

This is SVM^struct, except that all of the C API functions (except those dealing with C specific problems) instead call a function of the same name in a Python module, so you can write an SVM^struct instance in pure Python.  If you don't have Python you can get it here:

	http://www.python.org/

Python tends to be easier and faster to code than C, less resistant to change and code reorganization, tends to be MANY times more compact, there's no explicit memory management, and Python's object oriented-ness means that some tedious tasks in SVM^struct can be easily replaced with default built in behavior.

My favorite example of this last point is that, since Python objects can be assigned any attribute, and since many Python objects are easily serializable with the "pickle" module, adding a field to the struct-model in Python code consists of a simple assignment like "sm.foo = 5", and that's it.  In C one would add it to the struct, add an assignment, add code to write it to a model file, add code to parse it from a model file, and then test it to make sure all these little changes work well with each other.

--------
BUILDING

A simple "make" should do it, UNLESS the Python library you want to use is not the library corresponding to the Python interpreter you get when you just type "python".

You might want to modify the Makefile to modify the PYTHON variable to the path of the desired interpreter.  When you install Python, you install a library and an interpreter.  This interpreter is able to output where its corresponding library is stored.  The Makefile calls the Python interpreter to get this information, as well as other important information relevant to building a C application with embedded Python.  You can specify the path of your desired interpreter by setting PYTHON to something other than "python".

I have tried building SVM^python with both Python 2.3 and 2.4 on OS X and Linux.  Obviously if the Python module you write uses features specific to Python 2.4 (like generator expressions or the long overdue "sorted") you wouldn't be able to use the module with an SVM^python built against the Python 2.3 library.

--------
USING

One thing that is very annoying is that your PYTHONPATH environment variable has to contain "." so the executable knows where to look for the module to load.

The file svmstruct.py is a Python module, and also contains documentation on all the functions which the C code may attempt to call.  This is a good place to start reading if you are already familiar with SVM^struct and want to get familiar with how to build a SVM^python Python script.  This describes what each function should do and, for non-required functions, describes the default behavior that happens if you DON'T implement them. The multiclass.py file is an example implementation of multiclass classification in Python.

Once you've written a Python module in the file "foo.py" based on "svmstruct.py" and you want to use SVM^python with this module, you would use the following command line options.

./svm_python_learn    --m foo [options] <train> <model>
./svm_python_classify --m foo [options] <test>  <model> <output>

Note that SVM^python accepts the same arguments as SVM^struct except for this extra --m option.  If the --m option is omitted it is equivalent to including the command line arguments "--m svmstruct".

--------
EXAMPLE

I've included an implementation of SVM^multiclass in Python with this package.  This includes the appropriate example training files in "multi-example" from SVM^multiclass.  The commands below use the module in "multiclass.py" to build a multi-class model for data in "multi-example/train.dat", write the model to "themodel", and test the model on data from "multi-example/test.dat".

./svm_python_learn --m multiclass -c 1 multi-example/train.dat themodel
./svm_python_classify --m multiclass multi-example/test.dat themodel output

