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

            if result.shape != output_grid.shape:
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
    
    # pattern: input is NxM, output is 2Nx2M, so 4 quadrants of the original. the input is split into 4 quadrants.
    # top left = original as is
    # top right = fliplr (flip left right, mirror across vertical axis, cols reverse) so flip across y basically
    # bottom left = flipud (flip up down, mirror across horizontal axis, rows reverse) so flip across x basically
    # bottom right = rot90 twice = 180 degrees = both fliplr and flipud combined, so essentially flip across x and then y.
    # np.hstack joins arrays side by side (left to right) to form each row of quadrants
    # np.vstack joins arrays top to bottom to combine top row and bottom row
    def try_mirror_tile(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()
            
            # output must be exactly double input in both dimensions
            if out_grid.shape[0] != in_grid.shape[0] * 2:
                return None
            if out_grid.shape[1] != in_grid.shape[1] * 2:
                return None
            
            # top left quadrant = original
            top_left = in_grid

            # top right quadrant: mirror left right across y axis (cols reverse, rows same)
            top_right = np.fliplr(in_grid)

            # bottom left quadrant: mirror up down across x axis (rows reverse, cols same)
            bottom_left = np.flipud(in_grid)

            # bottom right quadrant: both flips combined = rotate 180, flipping across x + y.
            bottom_right = np.rot90(in_grid, 2)

            # first join left and right quadrants side by side for each row
            top_row = np.hstack([top_left, top_right])
            bottom_row = np.hstack([bottom_left, bottom_right])

            # join both on top of each other to combine all 4 quadrants
            expected = np.vstack([top_row, bottom_row])
            
            if not np.array_equal(expected, out_grid):
                return None
        
        # apply same logic above to form quadrants combine + return
        top_left = test_input
        top_right = np.fliplr(test_input)
        bottom_left = np.flipud(test_input)
        bottom_right = np.rot90(test_input, 2)

        top_row = np.hstack([top_left, top_right])
        bottom_row = np.hstack([bottom_left, bottom_right])

        result = np.vstack([top_row, bottom_row])
        
        return result

    # pattern: each row has at most two nonzero pixels, one on the left and one on the right.
    # if both those pixels are the same color, fill the entire row with that color.
    # if they are different colors, leave the row the same. so iterate by elem per row, find
    # two nonzero ones, if they match replace that row with row filled with those nonzero pixels.
    def try_fill_matching_border(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()
            
            if in_grid.shape != out_grid.shape:
                return None
            
            rows = in_grid.shape[0]
            cols = in_grid.shape[1]

            # this transform only applies if all nonzero values are on the left or right border
            # if any nonzero value is in the middle columns, this is not our transform
            for r in range(rows):
                for c in range(1, cols - 1):
                    if in_grid[r][c] != 0:
                        return None
            
            # same size output as inpuit
            expected = np.zeros_like(in_grid)
            
            for r in range(rows):
                left_color = in_grid[r][0]
                right_color = in_grid[r][cols - 1]
                
                # if both pixels match and are nonzero then fill entire row
                if left_color != 0 and left_color == right_color:
                    for c in range(cols):
                        expected[r][c] = left_color
                else:
                    # otherwise just copy the row as is 
                    for c in range(cols):
                        expected[r][c] = in_grid[r][c]
            
            if not np.array_equal(expected, out_grid):
                    #print('failed on row mismatch')
                    #print('expected:', expected)
                   # print('out_grid:', out_grid)
                    return None
            # apply to test input
        rows = test_input.shape[0]
        cols = test_input.shape[1]
        result = np.zeros_like(test_input)
        
        for r in range(rows):
            left_color = test_input[r][0]
            right_color = test_input[r][cols - 1]
            
            if left_color != 0 and left_color == right_color:
                for c in range(cols):
                    result[r][c] = left_color
            else:
                for c in range(cols):
                    result[r][c] = test_input[r][c]
        
        return result
    

    # pattern: grid is split exactly in half by a separator row or column of repeated values.
    #  find the midpoint of the grid, check if that row/col is valid separator
    # (all same nonzero value in a row or col basically that splits grid into two equal halves), 
    #  then split into two equal panels and try OR/AND/XOR/NOR. 
    # logic between them. output color is fi from training output.
    # this generalizes across problems with different separator colors and boolean operations.
    def try_separator_overlay(self, training, test_input):
        
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()
            
            rows = in_grid.shape[0]
            cols = in_grid.shape[1]
            
            # separator has to be exactly in the middle so grid size has to be odd
            separator_row = -1
            separator_col = -1
            
            #check for row split first, if not then chekck col
            if rows % 2 == 1:
                mid = rows // 2
                first_val = in_grid[mid][0]
                if first_val != 0:
                    is_separator = True
                    for c in range(cols):
                        if in_grid[mid][c] != first_val:
                            is_separator = False
                            break
                    if is_separator:
                        separator_row = mid
            
            if separator_row == -1 and cols % 2 == 1:
                mid = cols // 2
                first_val = in_grid[0][mid]
                if first_val != 0:
                    is_separator = True
                    for r in range(rows):
                        if in_grid[r][mid] != first_val:
                            is_separator = False
                            break
                    if is_separator:
                        separator_col = mid
            
            if separator_row == -1 and separator_col == -1:
                return None
            
            # split into two equal panels
            if separator_row != -1:
                panel_a = in_grid[0:separator_row, :]
                panel_b = in_grid[separator_row + 1:, :]
                panel_rows = separator_row
                panel_cols = cols
            else:
                panel_a = in_grid[:, 0:separator_col]
                panel_b = in_grid[:, separator_col + 1:]
                panel_rows = rows
                panel_cols = separator_col
            
            # output has to match panel size
            if separator_row != -1 and out_grid.shape[0] != panel_rows:
                return None
            if separator_col != -1 and out_grid.shape[1] != panel_cols:
                return None
            
            # find output color from first nonzero in out_grid, assuming
            # output is only one color throughout
            output_color = 0
            for r in range(out_grid.shape[0]):
                for c in range(out_grid.shape[1]):
                    if out_grid[r][c] != 0:
                        output_color = out_grid[r][c]
                        break
                if output_color != 0:
                    break
            
            if output_color == 0:
                return None
            
            # try each boolean operation against training output, and find one that matches
            # the training output. use that as we assume consistent op throughout the problem if it matches
            matched_op = None
            for op in ['OR', 'AND', 'XOR', 'NOR']:
                expected = np.zeros_like(out_grid)
                for r in range(panel_rows):
                    for c in range(panel_cols):
                        a_val = panel_a[r][c] != 0
                        b_val = panel_b[r][c] != 0
                        
                        if op == 'OR':
                            condition = a_val or b_val
                        elif op == 'AND':
                            condition = a_val and b_val
                        elif op == 'XOR':
                            condition = a_val != b_val
                        elif op == 'NOR':
                            condition = not a_val and not b_val
                        
                        if condition:
                            expected[r][c] = output_color
                
                if np.array_equal(expected, out_grid):
                    matched_op = op
                    break
            
            if matched_op is None:
                return None
        
        # apply to test input using matched operation
        rows = test_input.shape[0]
        cols = test_input.shape[1]
        
        separator_row = -1
        separator_col = -1
        
        if rows % 2 == 1:
            mid = rows // 2
            first_val = test_input[mid][0]
            if first_val != 0:
                is_separator = True
                for c in range(cols):
                    if test_input[mid][c] != first_val:
                        is_separator = False
                        break
                if is_separator:
                    separator_row = mid
        
        if separator_row == -1 and cols % 2 == 1:
            mid = cols // 2
            first_val = test_input[0][mid]
            if first_val != 0:
                is_separator = True
                for r in range(rows):
                    if test_input[r][mid] != first_val:
                        is_separator = False
                        break
                if is_separator:
                    separator_col = mid
        
        if separator_row == -1 and separator_col == -1:
            return None
        
        if separator_row != -1:
            panel_a = test_input[0:separator_row, :]
            panel_b = test_input[separator_row + 1:, :]
            panel_rows = separator_row
            panel_cols = cols
        else:
            panel_a = test_input[:, 0:separator_col]
            panel_b = test_input[:, separator_col + 1:]
            panel_rows = rows
            panel_cols = separator_col
        
        result = np.zeros((panel_rows, panel_cols), dtype=test_input.dtype)
        
        for r in range(panel_rows):
            for c in range(panel_cols):
                a_val = panel_a[r][c] != 0
                b_val = panel_b[r][c] != 0
                
                if matched_op == 'OR':
                    condition = a_val or b_val
                elif matched_op == 'AND':
                    condition = a_val and b_val
                elif matched_op == 'XOR':
                    condition = a_val != b_val
                elif matched_op == 'NOR':
                    condition = not a_val and not b_val
                
                if condition:
                    result[r][c] = output_color
        
        return result


    #consolidate color
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
            self.try_mirror_tile(training, test_input),
            self.try_fill_matching_border(training, test_input),
            self.try_separator_overlay(training, test_input),
            self.try_bounding_box(test_input),
        ]

        for result in transform_attempts:
            if result is not None:
                predictions.append(result)
                return predictions

        return predictions
