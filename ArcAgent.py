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

        #store a list of possible transforms. Our library
        self.transforms = [
            lambda g: g,                    # identity (no change)
            lambda g: np.rot90(g, 1),      # 90 degrees
            lambda g: np.rot90(g, 2),      # 180 degrees
            lambda g: np.rot90(g, 3),      # 270 degrees
            lambda g: np.fliplr(g),        # horizontal flip
            lambda g: np.flipud(g),        # vertical flip
            lambda g: np.transpose(g),  #transpose
        ]
    

    #Given a non zero pixel, expand a 3x3 box around it. So basically
    #turn any individual nonzero pixel into a 3x3 pixel with same color/attributes
    # as the original, bounded by grid size obviously
    def try_block_expand(self, training, test_input):
        
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()
            
            #output size has to equzl input for this transform
            if in_grid.shape != out_grid.shape:
                return None
            
            rows = in_grid.shape[0]
            cols = in_grid.shape[1]
            
            # clone input grid with blank slate
            expected = np.zeros_like(in_grid)

            #find non zero pixels
            nonzero = np.argwhere(in_grid != 0)
            
            for nonzero_pixel in nonzero:
                pixel_row = nonzero_pixel[0]
                pixel_col = nonzero_pixel[1]
                
                # fill 3x3 block centered at pixel. check
                # for out of bounds in grid. + 2 bc of exclusive
                # python slciing
                row_start = max(0, pixel_row - 1)
                row_end = min(rows, pixel_row + 2)
                col_start = max(0, pixel_col - 1)
                col_end = min(cols, pixel_col + 2)
                
                #fill box of 1s 
                expected[row_start:row_end, col_start:col_end] = 1
            
            if not np.array_equal(expected, out_grid):
                return None
        
        # apply to test input
        rows = test_input.shape[0]
        cols = test_input.shape[1]
        result = np.zeros_like(test_input)
        nonzero = np.argwhere(test_input != 0)
        
        for nonzero_pixel in nonzero:
            pixel_row= nonzero_pixel[0]
            pixel_col = nonzero_pixel[1]
            
            #define box + clamp for grid boundaries like before
            row_start = max(0, pixel_row - 1)
            row_end = min(rows, pixel_row + 2)
            col_start = max(0, pixel_col - 1)
            col_end = min(cols, pixel_col + 2)
            
            #apply 3x3 box transform
            result[row_start:row_end, col_start:col_end] = 1
        
        return result

    #Find a non black or nonzero pixel, draw diagonals from it that extend outward
    #in all four directions (diagonals). 
    def try_x_pattern(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()
            
            # find all non zero pixels in input
            nonzero = np.argwhere(in_grid != 0)
            
            # if more than one this transform fails
            if len(nonzero) != 1:
                return None
            
            # get pos and color
            pr = nonzero[0][0]
            pc = nonzero[0][1]
            color = in_grid[pr][pc]
            
            # check if transform applies throughout
            rows = out_grid.shape[0]
            cols = out_grid.shape[1]
            
            for r in range(rows):
                for c in range(cols):
                    # two cells are on the same diagonal if their
                    # row/col difference is the same
                    row_diff = r - pr
                    col_diff = c - pc
                    
                    # down right + up left diagonal: as row increases col increases.
                    # move tgt
                    down_right = row_diff == col_diff
                    
                    # down left + up right diagonal: as row decreases col increases.
                    #move opposite
                    down_left = row_diff == -col_diff
                    
                    on_x = down_right or down_left
                    actual = out_grid[r][c]
                    
                    if on_x:
                        if actual != color:
                            return None
                    else:
                        if actual != 0:
                            return None
        
        # apply to test input
        nonzero = np.argwhere(test_input != 0)
        if len(nonzero) != 1:
            return None
        
        pr = nonzero[0][0]
        pc = nonzero[0][1]
        color = test_input[pr][pc]
        
        rows = test_input.shape[0]
        cols = test_input.shape[1]
        result = np.zeros_like(test_input)

        for r in range(rows):
            for c in range(cols):
                row_diff = r - pr
                col_diff = c - pc
                
                # on diagonal if distances are equal in magnitude
                # am I the same number of steps away in rows as I am in cols?
                same_distance = abs(row_diff) == abs(col_diff)
                
                if same_distance:
                    result[r][c] = color

        return result
    
    #For each input element in grid, check corresponding output
    # element. if theres a consistent mapping to some value across
    # entire input, we have found a color transform, return mapping
    def try_color_substitution(self, training):
        mapping = {}
        
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()
            
            # if not same shape return
            if in_grid.shape != out_grid.shape:
                return None
            
            rows = in_grid.shape[0]
            cols = in_grid.shape[1]
            
            for r in range(rows):
                for c in range(cols):
                    in_val = in_grid[r][c]
                    out_val = out_grid[r][c]
                    
                    #map input index to output for 1:1
                    if in_val not in mapping:
                        mapping[in_val] = out_val
                    else:
                        # check if same map applies. if not
                        # then its an inconsistent transform so we return
                        existing = mapping[in_val]
                        if existing != out_val:
                            #print('conflict:', in_val, '->', existing, 'vs', out_val)
                            return None
        
        # if it maps to itself then its no transform so return
        is_identity = True
        for key in mapping:
            if mapping[key] != key:
                is_identity = False
        
        if is_identity:
            return None
        
        return mapping
    
    #once mapping generated, apply it to input
    #and create our transform output to compare
    def apply_color_mapping(self, grid, mapping):
        rows = grid.shape[0]
        cols = grid.shape[1]
        result = grid.copy()
        
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                if val in mapping:
                    result[r][c] = mapping[val]
        
        return result


    #Helper method: given a transform, check if it matches all training pairs
    def check_transform(self, transform, training_pair):
        for pair in training_pair:
            input_grid = pair.get_input_data().data()
            output_grid = pair.get_output_data().data()
            result = transform(input_grid)

            if result.shape != output_grid.shape:we
                return  False
            if not np.array_equal(result, output_grid):
                return False
        return True

    # pattern: input has a shape or cluster of nonzero pixels floating in a sea of zeros.
    # output is just that shape cropped out with the zero padding stripped away.
    # find the first and last row/col that contain any nonzero value, then slice
    # the grid between those boundaries to extract just the shape itself.
    def try_bounding_box(self, test_input):
        # find any rows or cols with non zero value
        rows_with_nonzero = np.any(test_input != 0, axis=1)
        cols_with_nonzero = np.any(test_input != 0, axis=0)
        
        # get the index where True (nonzero elems)
        nonzero_rows = np.where(rows_with_nonzero)[0]
        nonzero_cols = np.where(cols_with_nonzero)[0]
        
        # if no nonzero pixels at all, return None
        if len(nonzero_rows) == 0 or len(nonzero_cols) == 0:
            return None
        
        # first and last row/col with nonzero value
        first_row = nonzero_rows[0]
        last_row = nonzero_rows[-1]
        first_col = nonzero_cols[0]
        last_col = nonzero_cols[-1]
        
        # slice out the bounding box
        cropped = test_input[first_row:last_row+1, first_col:last_col+1]
        
        return cropped
    

    #pattern: input is just one row with some nonzero cells. Output grid rows = input cols // 2, output grid cols = input cols.
    #first row of output grid is same number of nonzero cells in input grid (so the input row essentially). fill out the rest
    # of the output grid row by row, appending one more nonzero cell of the same color to the n+1th index, creating a staircase type
    #formation. So if we have 3 nonzero and 5 zero in first row, we then have 4 nonzero and 4 zero in second, 5 nonzero and 3 zero in third,
    # etc til we fill out output grid
    def try_staircase(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()
            
            # input has to be exactly one row
            if in_grid.shape[0] != 1:
                return None
            
            num_cols = in_grid.shape[1]
            
            # count nonzero cells and then find the color (assuming same color throughout input)
            nonzero_count = 0
            color = 0
            for c in range(num_cols):
                if in_grid[0][c] != 0:
                    nonzero_count = nonzero_count + 1
                    color = in_grid[0][c]
        
            
            if nonzero_count == 0:
                return None
            
            # define output rows as input rows // 2
            num_output_rows = num_cols // 2

                           
           # print('num_cols:', num_cols, 'nonzero_count:', nonzero_count, 'actual:', out_grid.shape[0], 'last row filled:', np.count_nonzero(out_grid[-1]))
            
            # build expected output row by row
            expected = np.zeros((num_output_rows, num_cols), dtype=in_grid.dtype)
            for r in range(num_output_rows):
                #staircase so increment 1 per row to form staircase til we run out of space
                fill_up_to = nonzero_count + r
                for c in range(fill_up_to):
                    expected[r][c] = color
            
            if not np.array_equal(expected, out_grid):
                return None

        # apply to test input
        if test_input.shape[0] != 1:
            return None
            
        num_cols = test_input.shape[1]
            
        nonzero_count = 0
        color = 0
        for c in range(num_cols):
            if test_input[0][c] != 0:
                nonzero_count = nonzero_count + 1
                color = test_input[0][c]
            
        if nonzero_count == 0:
            return None
            
        num_output_rows = num_cols // 2
        result = np.zeros((num_output_rows, num_cols), dtype=test_input.dtype)
            
        for r in range(num_output_rows):
            fill_up_to = nonzero_count + r
            for c in range(fill_up_to):
                result[r][c] = color

        return result
        

    #consolidate color substitution + mapping into one function for simplicity
    def try_color_substitution_and_apply(self, training, test_input):
        mapping = self.try_color_substitution(training)
        if mapping is None:
            return None
        return self.apply_color_mapping(test_input, mapping)
    
    

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
                
        predictions = []
        training = arc_problem.training_set()
        test_input = arc_problem.test_set().get_input_data().data()

        # geometric transforms
        for transform in self.transforms:
            if self.check_transform(transform, training):
                predictions.append(transform(test_input))
                return predictions

        # parameterized transforms — each returns result or None
        transform_attempts = [
            self.try_color_substitution_and_apply(training, test_input),
            self.try_x_pattern(training, test_input),
            self.try_block_expand(training, test_input),
            self.try_staircase(training, test_input),
            self.try_bounding_box(test_input),
        ]

        for result in transform_attempts:
            if result is not None:
                predictions.append(result)
                return predictions

        return predictions
