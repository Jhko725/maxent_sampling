from numpy import array, sqrt
from time import time
from .GraphIO import SimpleGraph
import sys

def RecursiveLinReg():
    """
    Computes recursive linear regression on numpy array x, y according to the linear model y = alpha + beta*x. The algorithm is from the paper: Klotz, Jerome H. "UPDATING SIMPLE LINEAR REGRESSION." Statistica Sinica 5, no. 1 (1995): 399-403.
    A basic implementation of the algorithm is also presented at: https://stackoverflow.com/questions/52070293/efficient-online-linear-regression-algorithm-in-python. Here, higher order function syntax of python was used to clean up the code.

    Returns
    -------
    UpdateRegression(x_new, y_new) : function
        A function that returns the updated regression parameters given new set of observations x_new and y_new
    """
    # Initialize internal function variables
    n = 0 # Number of previous samples. Initially, set to zero.
    x_avg = 0 # Average of previous x samples. Initially, set to zero.
    y_avg = 0 # Average of previous y samples. Initially, set to zero.
    Sxy = 0 # Covariance of previous x and y samples. Initially, set to zero.
    Sx = 0  # Variance of previous x samples. Initially, set to zero.

    def UpdateRegression(x_new, y_new):
        """
        Updates the regression parameters alpha, beta for the linear model y = alpha + beta*x. The algorithm is from the paper: Klotz, Jerome H. "UPDATING SIMPLE LINEAR REGRESSION." Statistica Sinica 5, no. 1 (1995): 399-403.

        Parameters
        ----------
        x_new : 1D list or any other ordered iterable compatible with numpy.array()
            An array of new samples for x.
        y_new : 1D list or any other ordered iterable compatible with numpy.array()
            An array of new samples for y.  
        """
        nonlocal n, x_avg, y_avg, Sxy, Sx

        # Type casting to 1D numpy array
        x_new = array(x_new)
        y_new = array(y_new)
        n_new = n + len(x_new)

        x_avg_new = (x_avg*n + x_new.sum())/n_new
        y_avg_new = (y_avg*n + y_new.sum())/n_new

        if n > 0:
            x_star = (x_avg*sqrt(n) + x_avg_new*sqrt(n_new))/(sqrt(n)+sqrt(n_new))
            y_star = (y_avg*sqrt(n) + y_avg_new*sqrt(n_new))/(sqrt(n)+sqrt(n_new))
        elif n == 0:
            x_star = x_avg_new
            y_star = y_avg_new
        else:
            raise ValueError

        Sx_new = Sx + ((x_new-x_star)**2).sum()
        Sxy_new = Sxy + ((x_new-x_star).reshape(-1) * (y_new-y_star).reshape(-1)).sum()

        beta = Sxy_new/Sx_new
        alpha = y_avg_new - beta * x_avg_new

        x_avg, y_avg, Sxy, Sx, n = x_avg_new, y_avg_new, Sxy_new, Sx_new, n_new

        return alpha, beta
    return UpdateRegression
    
def StopWatch(delta_t):
    time_start = time()

    def TimePassed():
        nonlocal time_start, delta_t
        if time() - time_start < delta_t:
            return False
        else:
            time_start = time()
            return True
    return TimePassed

def PrintProgress(do_print, percent):
    if do_print:
        sys.stdout.write('\r')
        sys.stdout.write("{:8.5f} percent completed".format(percent*100))
        sys.stdout.flush()

    pass