import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        Write the code in this method to solve the incoming ArcProblem.
        Your agent will receive 1 problem at a time.

        You can add up to THREE (3) the predictions to the
        predictions list provided below that you need to
        return at the end of this method.

        In the Autograder, the test data output in the arc problem will be set to None
        so your agent cannot peek at the answer (even on the public problems).

        Also, if you return more than 3 predictions in the list it
        is considered an ERROR and the test will be automatically
        marked as INCORRECT.
        """

        #predictions: list[np.ndarray] = list()

        '''
        The next 2 lines are only an example of how to populate the predictions list.
        This will just be an empty answer the size of the input data;
        delete it before you start adding your own predictions.
        '''
        #output = np.zeros_like(arc_problem.test_set().get_input_data().data())
        #predictions.append(output)

        predictions = []

        grid = arc_problem.test_set().get_input_data().data()

        #find any rows/cols with non zero value. This is the nonblack "shape"
        #component we are extracting
        rows = np.any(grid != 0 , axis = 1)
        cols = np.any(grid != 0, axis = 0)

        #strip out only the ones that were True. so we have list of rows/cols where a nonzero appears.
        nonzero_rows = np.where(rows)[0]
        nonzero_cols = np.where(cols)[0]
        
        #if there are at least one non zero pixel in both row/col, we have a shape.
        #else we can just return empty set
        if len(nonzero_rows) > 0 and len(nonzero_cols) > 0:
            #first/last row with nonzero value
            first_row = nonzero_rows[0]
            last_row = nonzero_rows[-1]

            #first/last col with nonzero value
            first_col = nonzero_cols[0]
            last_col = nonzero_cols[-1]

            #slice grid from top left corner to bottom right to get the shape out from canvas
            cropped = grid[first_row:last_row+1, first_col:last_col+1]

            predictions.append(cropped)


        return predictions
    
