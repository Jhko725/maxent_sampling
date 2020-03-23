# Package graphsampler

This is the package for the final project of M2177.003000 Advanced Data Mining (Fall 2019) at the Department of Computer Science and Engineering at SNU.

The package includes the following modules:
* GraphIO.py: Implements class SimpleGraph to deal with graph data. Instances of SimpleGraph can be saved to table format to be loaded by other packages such as snap-stanford. 
* Sampling.py: Implements traversal-based graph sampling algorithms to be used in the project. Implemented algorithms include:
    - Simple Random Walk
    - Uniform Node Random Walk (via Metropolis-Hastings algorithm)
    - Degree Proportional Random Walk (corresponds to first order approximation of Maximal Entropy Random Walk)
    - Maximal Entropy Random Walk
* Utility.py: A module containing miscellaneous functions used in this package.

For more information on classes and methods regarding the package, use the python help() function to access the documentation. 