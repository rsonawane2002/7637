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
    
    # helper function: find a separator at exact midpoint of grid that splts
    # it into two equal halves (in terms of dimension). could be col wise or row wise
    # returns (separator_row, separator_col), either can be -1 if not found
    def find_separator(self, grid):
        rows = grid.shape[0]
        cols = grid.shape[1]
        separator_row = -1
        separator_col = -1
        
        #has to be odd rows to have midpoint. search for row midpoint
        #first, that entire midpoint row has to be the same value + nonzero
        #to be separator. if not then we move to col and check the same thing
        if rows % 2 == 1:
            mid = rows // 2
            first_val = grid[mid][0]
            if first_val != 0:
                is_sep = True
                for c in range(cols):
                    if grid[mid][c] != first_val:
                        is_sep = False
                        break
                if is_sep:
                    separator_row = mid
        
        if separator_row == -1 and cols % 2 == 1:
            mid = cols // 2
            first_val = grid[0][mid]
            if first_val != 0:
                is_sep = True
                for r in range(rows):
                    if grid[r][mid] != first_val:
                        is_sep = False
                        break
                if is_sep:
                    separator_col = mid
        
        return separator_row, separator_col


    # helper: given two boolean values and an operation name, return the result.
    # allows for robust checking of all logic gates and simplifies stuff.
    def apply_op(self, a_nonzero, b_nonzero, op):
        if op == 'OR':
            return a_nonzero or b_nonzero
        elif op == 'AND':
            return a_nonzero and b_nonzero
        elif op == 'XOR':
            return a_nonzero != b_nonzero
        elif op == 'NOR':
            return not a_nonzero and not b_nonzero
        return False


    # helper: build result grid by applying op between two panels that were split in half equally.
    # preserve_colors mode keeps original pixel values instead of recoloring to output_color. 
    # this is in the case that output color has multiple colors like e918
    def build_panel_result(self, panel_a, panel_b, op, preserve_colors, output_color):
        panel_rows = panel_a.shape[0]
        panel_cols = panel_a.shape[1]
        result = np.zeros((panel_rows, panel_cols), dtype=panel_a.dtype)
        
        for r in range(panel_rows):
            for c in range(panel_cols):
                a_val = panel_a[r][c]
                b_val = panel_b[r][c]
                a_nonzero = a_val != 0
                b_nonzero = b_val != 0
                
                if self.apply_op(a_nonzero, b_nonzero, op):
                    if preserve_colors:
                        if a_nonzero:
                            result[r][c] = a_val
                        else:
                            result[r][c] = b_val
                    else:
                        result[r][c] = output_color
        
        return result
    
    # helper: split a grid into two panels using a real separator line
    # (marker row/col found by find_separator). returns (None, None) if
    # there's no separator. shared by any transform that needs a
    # "border/piece" or "left half/right half" split with a marker removed.
    def split_by_separator(self, grid):
        sep_row, sep_col = self.find_separator(grid)
 
        if sep_row != -1:
            return grid[0:sep_row, :], grid[sep_row+1:, :]
        if sep_col != -1:
            return grid[:, 0:sep_col], grid[:, sep_col+1:]
 
        return None, None

    # helper: split a grid into two panels for panel_overlay style transforms.
    # tries a real separator first (via split_by_separator). if there isn't
    # one, falls back to an even row/col split with nothing removed. when
    # expected_shape is given (the training output shape), it prefers whichever
    # split actually produces panels of that shape, since a grid can
    # sometimes split evenly on both axes.
    def split_into_panels(self, grid, expected_shape=None):
        panel_a, panel_b = self.split_by_separator(grid)
        if panel_a is not None:
            return panel_a, panel_b
 
        rows = len(grid)
        cols = len(grid[0])
 
        # figure out the even row split, if the grid has an even
        # number of rows. has_row_split checks whether this split is even
        # possible, since row_split_a/row_split_b stay None otherwise.
        has_row_split = False
        row_split_a = None
        row_split_b = None
        if rows % 2 == 0:
            half = rows // 2
            row_split_a = grid[0:half, :]
            row_split_b = grid[half:, :]
            has_row_split = True
 
        # same idea for an even column-wise split
        has_col_split = False
        col_split_a = None
        col_split_b = None
        if cols % 2 == 0:
            half = cols // 2
            col_split_a = grid[:, 0:half]
            col_split_b = grid[:, half:]
            has_col_split = True
 
        # if we know the expected output shape (training case), then prefer
        # whichever split actually matches it. a grid can also be split evenly
        # on both axes at once.
        if expected_shape is not None:
            if has_row_split and row_split_a.shape == expected_shape:
                return row_split_a, row_split_b
            if has_col_split and col_split_a.shape == expected_shape:
                return col_split_a, col_split_b
            return None, None
 
        # no expected shape to check against (test case), so just take
        # whichever split is available, preferring rows
        if has_row_split:
            return row_split_a, row_split_b
        if has_col_split:
            return col_split_a, col_split_b
 
        return None, None
    

    # pattern: grid splits into two equal panels, either through a real separator
    # line or (if there's no separator) an even row/col split. try OR/AND/XOR/NOR
    # logic between the panels. two modes: single output color (recolor to one
    # color) or preserve original colors from whichever panel had the nonzero
    # value. output color is guessed from training output. covers both
    # marker separated and plain even split problems since they share the
    # exact same cell by cell combining logic. The only thing thats diff is how panels are
    # found 
    def try_panel_overlay(self, training, test_input):

        matched_op = None
        preserve_colors = False
        output_color = 0

        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()
            panel_a, panel_b = self.split_into_panels(in_grid, expected_shape=out_grid.shape)

            if panel_a is None:
                return None
            if panel_a.shape != out_grid.shape:
                return None
            
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
            
            matched_op = None
            preserve_colors = False
            for op in ['OR', 'AND', 'XOR', 'NOR']:
                expected = self.build_panel_result(panel_a, panel_b, op, False, output_color)
                if np.array_equal(expected, out_grid):
                    matched_op = op
                    preserve_colors = False
                    break
                expected2 = self.build_panel_result(panel_a, panel_b, op, True, output_color)
                if np.array_equal(expected2, out_grid):
                    matched_op = op
                    preserve_colors = True
                    break

            if matched_op is None:
                return None
            
        panel_a, panel_b = self.split_into_panels(test_input)

        if panel_a is None:
            return None
        
        return self.build_panel_result(panel_a, panel_b, matched_op, preserve_colors, output_color)
    
    # pattern: separator splits into a border shape with a hole (panel_a) and a
    # colored piece (panel_b). if the piece exactly matches the hole shape it
    # fits in and gets stamped with its color, otherwise it's the wrong piece
    # and we just leave the hole blank. only uses a real separator split (not
    # the even split fallback) since the hole/piece idea only makes sense
    # with a marker line actually dividing the two.
    def try_shape_fit_overlay(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()
 
            panel_a, panel_b = self.split_by_separator(in_grid)
            if panel_a is None:
                return None
 
            if panel_a.shape != panel_b.shape:
                return None
            if panel_a.shape != out_grid.shape:
                return None
 
            rows = len(panel_a)
            cols = len(panel_a[0])

            # check if the piece's shape (panel_b's nonzero cells) exactly
            # matches the hole's shape (panel_a's blank cells), one cell at
            # a time. shapes_match starts True and flips to False for good
            # the moment any cell disagrees.
            shapes_match = True
            for r in range(rows):
                for c in range(cols):
                    is_hole = panel_a[r][c] == 0
                    is_piece = panel_b[r][c] != 0
                    if is_hole != is_piece:
                        shapes_match = False

            expected = panel_a.copy()

            # only paint the piece's colors into the hole if it's an exact
            # match. otherwise the piece is wrong and panel_a stays
            # as is with all blanks included.
            if shapes_match:
                for r in range(rows):
                    for c in range(cols):
                        if panel_a[r][c] == 0:
                            expected[r][c] = panel_b[r][c]
            if not np.array_equal(expected, out_grid):
                return None
 
        # same rule, applied to the test input
        panel_a, panel_b = self.split_by_separator(test_input)
        if panel_a is None:
            return None

        if panel_a.shape != panel_b.shape:
            return None

        rows = len(panel_a)
        cols = len(panel_a[0])

        shapes_match = True
        for r in range(rows):
            for c in range(cols):
                is_hole = panel_a[r][c] == 0
                is_piece = panel_b[r][c] != 0
                if is_hole != is_piece:
                    shapes_match = False

        result = panel_a.copy()
        if shapes_match:
            for r in range(rows):
                for c in range(cols):
                    if panel_a[r][c] == 0:
                        result[r][c] = panel_b[r][c]

        return result


    #consolidate color
    def try_color_substitution_and_apply(self, training, test_input):
        mapping = self.try_color_substitution(training)
        if mapping is None:
            return None
        return self.apply_color_mapping(test_input, mapping)

    # helper: group all nonzero cells into connected components using
    # 4 way connections(up/down/left/right, no diagonals). cells only join a
    # component if they're the same color and directly adjacent in those directions. 
    # returns a list of components, each with its color, cell list, and bounding box.
    # this is the object detection goal from the milestone C plan, treating
    # clusters of same colored pixels as their own objects instead of reasoning
    # about the whole grid pixsel by pixel as one flat pattern.
    def get_connected_components(self, grid, background=0):
        rows = len(grid)
        cols = len(grid[0])
        visited = []
        for r in range(rows):
            visited_row = []
            for c in range(cols):
                visited_row.append(False)
            visited.append(visited_row)
        components = []

        for start_r in range(rows):
            for start_c in range(cols):

                #if already visited, skip cell
                if visited[start_r][start_c]:
                    continue

                color = grid[start_r][start_c]
                #if cell is same as background color 
                #cant be unique object. mark as visited and continue
                if color == background:
                    visited[start_r][start_c] = True
                    continue

                # if not background color, flood fill with a stack:
                # essentially check up, down, left, righ for similar
                # color, add to stack, continue till entire shape is 
                # mapped out
                stack = [(start_r, start_c)]
                visited[start_r][start_c] = True
                cells = []

                while len(stack) > 0:
                    r, c = stack.pop()
                    cells.append((r, c))

                    #check all neighbors of a cell for same color: up, down, left, right
                    neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                    for nr, nc in neighbors:
                        #out of bounds check, already visited cell, different color
                        # neighbor check. continue if this is hte case
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            continue
                        if visited[nr][nc]:
                            continue
                        if grid[nr][nc] != color:
                            continue

                        #else append to stack and contonue flood fill
                        visited[nr][nc] = True
                        stack.append((nr, nc))

                #find min row, max row, min col, max col
                #to extract the shape. just set as first elem
                #values for now and iterate through entire shape
                min_row = cells[0][0]
                max_row = cells[0][0]
                min_col = cells[0][1]
                max_col = cells[0][1]
                for cell_r, cell_c in cells:
                    if cell_r < min_row:
                        min_row = cell_r
                    if cell_r > max_row:
                        max_row = cell_r
                    if cell_c < min_col:
                        min_col = cell_c
                    if cell_c > max_col:
                        max_col = cell_c

                #add component. grid could have multiple
                #disjoint components
                component = {
                    'color': color,
                    'cells': cells,
                    'min_row': min_row,
                    'max_row': max_row,
                    'min_col': min_col,
                    'max_col': max_col,
                }
                components.append(component)

        return components


    # helper: given a sub grid, fill in the missing (background) cells by
    # mirroring the interior against itself. mirror_modes controls which
    # reflections to try: 'horizontal' mirrors left right, 'vertical' mirrors
    # top bottom. only writes into cells that are still background, so it never
    # overwrites a real pixel, whether that's pattern color or a stray border
    # pixel poking into the interior from a container shape.
    def repair_interior_symmetry(self, interior, mirror_modes):
        result = interior.copy()
        rows = len(interior)
        cols = len(interior[0])

        for mode in mirror_modes:
            if mode == 'horizontal':
                mirrored = np.fliplr(interior)
            else:
                mirrored = np.flipud(interior)

            for r in range(rows):
                for c in range(cols):
                    if result[r][c] == 0 and mirrored[r][c] != 0:
                        result[r][c] = mirrored[r][c]

        return result


    # helper: run the container detection + interior repair pass on one grid.
    # any connected component with a component that is at least 3x3 gets treated as
    # a container: big enough to have a real interior. smaller blobs (stray
    # marker pixels) get skipped since they can't enclose anything.
    def apply_symmetry_repair(self, grid, mirror_modes):
        result = grid.copy()
        components = self.get_connected_components(grid)

        for component in components:
            min_row = component['min_row']
            max_row = component['max_row']
            min_col = component['min_col']
            max_col = component['max_col']

            height = max_row - min_row + 1
            width = max_col - min_col + 1

            #if not 3x3, at least, cant enclose anything so skip
            if height < 3 or width < 3:
                continue

            interior = grid[min_row+1:max_row, min_col+1:max_col]
            repaired = self.repair_interior_symmetry(interior, mirror_modes)
            result[min_row+1:max_row, min_col+1:max_col] = repaired

        return result


    # pattern: grid has one or more container objects (found through connected
    # components helper method above), each having a partial symmetric shape inside it
    # "repair" every container by mirroring its interior against itself
    # and filling in whatever's missing. tries horizontal only, vertical only,
    # and both repair against training and then uses whichever mode reproduces
    # every training pair exactly, then applies that mode to the test input.
    def try_symmetry_repair_regions(self, training, test_input):

        mirror_mode_options = [
            ['horizontal'],
            ['vertical'],
            ['horizontal', 'vertical'],
        ]

        matched_modes = None

        for mirror_modes in mirror_mode_options:
            all_pairs_match = True

            for pair in training:
                in_grid = pair.get_input_data().data()
                out_grid = pair.get_output_data().data()
                
                #has to match size
                if in_grid.shape != out_grid.shape:
                    all_pairs_match = False
                    break

                expected = self.apply_symmetry_repair(in_grid, mirror_modes)

                if not np.array_equal(expected, out_grid):
                    all_pairs_match = False
                    break

            if all_pairs_match:
                matched_modes = mirror_modes
                break

        if matched_modes is None:
            return None

        return self.apply_symmetry_repair(test_input, matched_modes)
    
   # helper: find every row and column that's a full separator line, so a
    # row/col where every cell is the same nonzero value. unlike
    # find_separator (which is a single midpoint line), this finds ALL
    # lines, so a grid can be split up into an NxM grid of panels by
    # however many separators it has, not just one.
    def find_grid_lines(self, grid):
        rows = len(grid)
        cols = len(grid[0])
        row_lines = []
        row_line_colors = []
        col_lines = []
        col_line_colors = []

        # check row wise, also track what color each detected line is
        for r in range(rows):
            first_val = grid[r][0]
            if first_val == 0:
                continue
            is_line = True
            for c in range(cols):
                if grid[r][c] != first_val:
                    is_line = False
                    break
            if is_line:
                row_lines.append(r)
                row_line_colors.append(first_val)

        # check col wise, same deal
        for c in range(cols):
            first_val = grid[0][c]
            if first_val == 0:
                continue
            is_line = True
            for r in range(rows):
                if grid[r][c] != first_val:
                    is_line = False
                    break
            if is_line:
                col_lines.append(c)
                col_line_colors.append(first_val)

        # sometimes a row/col inside a panel just happens to be filled
        # with one repeated color by pure coincidence, not an actual
        # separator. this shows up as a false positive above. real
        # separator lines always share the same color throughout the
        # whole grid, so filter down to whichever color shows up the most
        # among the detected lines and toss out anything that doesn't
        # match, since thats probably just coincidental panel content
        row_lines = self.filter_lines_by_majority_color(row_lines, row_line_colors)
        col_lines = self.filter_lines_by_majority_color(col_lines, col_line_colors)

        return row_lines, col_lines


    # helper: given a list of detected line positions and the color found
    # at each one, keep only the lines whose color matches whichever color
    # shows up the most across all of them. this is what weeds out
    # coincidental uniform rows/cols (like a panel happening to be filled
    # with one color) from actual separator lines, since real separators
    # are always the same color throughout the grid
    def filter_lines_by_majority_color(self, lines, line_colors):
        if len(lines) == 0:
            return lines

        color_counts = {}
        for color in line_colors:
            if color not in color_counts:
                color_counts[color] = 0
            color_counts[color] = color_counts[color] + 1

        majority_color = 0
        highest_count_seen = 0
        for color in color_counts:
            count_for_color = color_counts[color]
            if count_for_color > highest_count_seen:
                highest_count_seen = count_for_color
                majority_color = color

        filtered_lines = []
        for i in range(len(lines)):
            if line_colors[i] == majority_color:
                filtered_lines.append(lines[i])

        return filtered_lines
 
    # helper: given a dimension size, either total number of rows or cols, and a list of
    # separator line indices along it, return the list of (start, end)
    # segments that are between the separators. So it just returns the segmented portion
    #of the grid essentially, that the separator splits the grid into.
    def get_panel_segments(self, size, lines):
        segments = []
        start = 0

        #go through all separator indices. if the line index is 
        #after our current segment start, we have a new segment.
        #so append curr segment, and reset start to next non separator
        #row
        for line in sorted(lines):
            if line > start:
                segments.append((start, line))
            start = line + 1

        #if we reach end of the grid, that also counts as a separator.
        #add last part as a segment
        if start < size:
            segments.append((start, size))
        return segments
 
    # pattern: grid is split by full separator lines into a grid of panels.
    # each panel is either "blank" (its inside is just the separator's own
    # color) or filled with one real color. we count how many panels each real
    # color occupies, that count becomes that color's frequency. then collect the
    # frequencies that appear and sort them smallest to largest.
    # that sorted list becomes the output grids rows, one row per distinct
    # frequency seen (not one row per possible count so if no color has a
    # count of 3, there's no row for 3). the row for a  frequency is
    # filled with that many copies of whichever color has that frequency,
    # left aligned. output width is the largest
    # frequency seen, since that's the longest row that we need.
    def compute_panel_color_histogram(self, grid):

        #find the separators
        row_lines, col_lines = self.find_grid_lines(grid)

        #if 0 return
        if len(row_lines) == 0 or len(col_lines) == 0:
            return None
        
        #find color of the separators
        separator_color = grid[row_lines[0]][0]

        #get the non separator parts of the grid, col and row wise
        row_segments = self.get_panel_segments(len(grid), row_lines)
        col_segments = self.get_panel_segments(len(grid[0]), col_lines)

        #go through each "panel", find panel color of each panel
        #compute a count for each and map each color to a count,
        # assuming its not 0 or separator color
        color_counts = {}
        for row_start, row_end in row_segments:
            for col_start, col_end in col_segments:
                panel_color = 0
                for r in range(row_start, row_end):
                    for c in range(col_start, col_end):
                        val = grid[r][c]
                        if val != 0 and val != separator_color:
                            panel_color = val
                if panel_color != 0:
                    if panel_color not in color_counts:
                        color_counts[panel_color] = 0
                    color_counts[panel_color] = color_counts[panel_color] + 1
 
        if len(color_counts) == 0:
            return None

        #build output grid. sort in ascending order,
        # get max freq for width of output, number
        # of colors is the length row wise
        frequencies = sorted(set(color_counts.values()))
        max_freq = frequencies[-1]
        height = len(frequencies)
        width = max_freq

        #build output grid according to new dimensions
        result = []
        for r in range(height):
            row = []
            for c in range(width):
                row.append(0)
            result.append(row)

        #for each row 0 to height, find the freq that 
        #corresponds to that row index. so row 0 = lowest freq
        #color, row 2 = third lowest freq color. then get the color
        #that corresponds to that freq, and paint the current row
        #from left pixel by pixel. 
        for row_index in range(height):
            freq = frequencies[row_index]
            color_for_freq = None
            for color in color_counts:
                if color_counts[color] == freq:
                    color_for_freq = color
                    break
            for c in range(freq):
                result[row_index][c] = color_for_freq
 
        return np.array(result)
    
    # wrapper: runs compute_panel_color_histogram against every training pair
    # to confirm the rule actually holds for this problem before trusting it.
    # if any training pair doesn't match (like wrong shape, no grid lines found,
    # wrong histogram), we bail out with None rather than risk a wrong guess.
    # only once every pair checks out do we apply the same computation to
    # the real test input.
    def try_panel_color_histogram(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            # run the histogram computation on this training input and see
            # if it matches the training output exactly
            result = self.compute_panel_color_histogram(in_grid)
            if result is None:
                return None
            if not np.array_equal(result, out_grid):
                return None
 
        return self.compute_panel_color_histogram(test_input)
 
    # helper: find a color whose cells form a rectangle outline
    # every cell on the perimeter of that color's own recatangle is that
    # color. returns (color, min_row, min_col, max_row, max_col) for the
    # first such color found, or None if no color forms a clean rectangle.
    def find_rectangle_border_color(self, grid):
        rows = len(grid)
        cols = len(grid[0])
        colors_present = {}

        # group every nonzero cell by its color, so for each color we have
        # a list of every (row, col) position it is at
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                if val == 0:
                    continue
                if val not in colors_present:
                    colors_present[val] = []
                colors_present[val].append((r, c))

        #find the box for each color, by searching for min/max row and col
        # to extract the shape
        for color in colors_present:
            cells = colors_present[color]
            min_row = cells[0][0]
            max_row = cells[0][0]
            min_col = cells[0][1]
            max_col = cells[0][1]
            for cell_r, cell_c in cells:
                if cell_r < min_row:
                    min_row = cell_r
                if cell_r > max_row:
                    max_row = cell_r
                if cell_c < min_col:
                    min_col = cell_c
                if cell_c > max_col:
                    max_col = cell_c

            # needs to be at least a 3x3 (border on
            # all sides plus at least one interior cell), so anything
            # smaller can't be what we're looking for
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            if height < 3 or width < 3:
                continue
            

            # check if this color forms a proper hollow rectangle: the top
            # and bottom rows have to be only this color, and the left and
            # right columns must be only this color. only check
            # these four edges on purpose, a hollow rectangle's outline
            # is enough to prove it's a rectangle, don't need to check
            # the interior, which should be empty or have other
            # colors
            is_rectangle = True
            for c in range(min_col, max_col + 1):
                if grid[min_row][c] != color or grid[max_row][c] != color:
                    is_rectangle = False
                    break
            if is_rectangle:
                for r in range(min_row, max_row + 1):
                    if grid[r][min_col] != color or grid[r][max_col] != color:
                        is_rectangle = False
                        break
 
            if is_rectangle:
                return color, min_row, min_col, max_row, max_col
 
        return None
 
    # "marker" color is scattered both inside and outside the border.
    # count only the marker dots only inside the border's interior,
    # then fill a fixed 3x3 output row wise with that many marker color
    # cells. exterior dots of the marker color don't matter at all.
    def compute_marker_count_grid(self, grid):
        border_info = self.find_rectangle_border_color(grid)
        if border_info is None:
            return None
 
        border_color, min_row, min_col, max_row, max_col = border_info
 
        marker_color = 0
        marker_count = 0

        # iterate only inside the border, not the border itself. starting
        # at (min_row + 1) skips the top/left border row and column, and
        # stopping at max_row/max_col will skip
        # the bottom/right border row and column.
        for r in range(min_row + 1, max_row):
            for c in range(min_col + 1, max_col):
                val = grid[r][c]
                if val != 0 and val != border_color:
                    marker_color = val
                    marker_count = marker_count + 1
 
        if marker_color == 0:
            return None

        #output here is always 3x3. so 
        #build temporary output grid of correct size
        grid_size = 3
        result = []
        for r in range(grid_size):
            row = []
            for c in range(grid_size):
                row.append(0)
            result.append(row)

        # fill the 3x3 grid row first then cols top to bottom
        # with marker_count amount of marker_color (left to right). remaining tracks how
        # many cells are still left to fill. it counts down by one each
        # time we place a color, and once it hits 0 every cell after that
        # stays 0
        remaining = marker_count
        for r in range(grid_size):
            for c in range(grid_size):
                if remaining > 0:
                    result[r][c] = marker_color
                    remaining = remaining - 1
 
        return np.array(result)
    
    # wrapper: runs compute_marker_count_grid against every training pair to
    # confirm the border plus marker count rule actually holds for this
    # problem before trusting it. if any training pair doesn't match (like no
    # rectangle border found, no marker color found, wrong count), exit
    # out with None instead of risking a wrong guess. only once every pair
    # checks out do we apply the same computation to the real test input.
    def try_marker_count_grid(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()
 
            result = self.compute_marker_count_grid(in_grid)
            if result is None:
                return None
            if not np.array_equal(result, out_grid):
                return None
 
        return self.compute_marker_count_grid(test_input)
    
    
    # crops out the shape then just swaps whichever two colors it finds.
    # basically reuses the bounding box logic we already have and adds a
    # color swap step on top
    def try_crop_and_swap_colors(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            cropped = self.try_bounding_box(in_grid)
            if cropped is None:
                return None
            if cropped.shape != out_grid.shape:
                return None

            crop_rows = len(cropped)
            crop_cols = len(cropped[0])

            # walk the cropped grid and grab the first two different
            # nonzero colors we see. those are our swap pair
            color_a = 0
            color_b = 0
            for r in range(crop_rows):
                for c in range(crop_cols):
                    current_val = cropped[r][c]
                    if current_val == 0:
                        continue
                    if color_a == 0:
                        color_a = current_val
                    elif current_val != color_a and color_b == 0:
                        color_b = current_val

            # if we didn't find two distinct colors this rule doesn't apply
            if color_a == 0 or color_b == 0:
                return None

            expected = cropped.copy()
            for r in range(crop_rows):
                for c in range(crop_cols):
                    if cropped[r][c] == color_a:
                        expected[r][c] = color_b
                    elif cropped[r][c] == color_b:
                        expected[r][c] = color_a

            if not np.array_equal(expected, out_grid):
                return None

        # same swap logic, just running it on the actual test input now
        cropped = self.try_bounding_box(test_input)
        if cropped is None:
            return None

        crop_rows = len(cropped)
        crop_cols = len(cropped[0])

        color_a = 0
        color_b = 0
        for r in range(crop_rows):
            for c in range(crop_cols):
                current_val = cropped[r][c]
                if current_val == 0:
                    continue
                if color_a == 0:
                    color_a = current_val
                elif current_val != color_a and color_b == 0:
                    color_b = current_val

        if color_a == 0 or color_b == 0:
            return None

        result = cropped.copy()
        for r in range(crop_rows):
            for c in range(crop_cols):
                if cropped[r][c] == color_a:
                    result[r][c] = color_b
                elif cropped[r][c] == color_b:
                    result[r][c] = color_a

        return result


    # takes any solid filled in rectangle block and hollows it out, just
    # leaves the outline behind. kinda like popping the middle out of a
    # cracker
    def compute_hollow_solid_rectangles(self, grid):
        result = grid.copy()
        components = self.get_connected_components(grid)

        for component in components:
            min_row = component['min_row']
            max_row = component['max_row']
            min_col = component['min_col']
            max_col = component['max_col']

            box_height = max_row - min_row + 1
            box_width = max_col - min_col + 1

            # a solid block should have exactly height times width cells,
            # no gaps anywhere. if the count is off this isn't solid so skip it
            total_cells_in_component = len(component['cells'])
            expected_cell_count_if_solid = box_height * box_width
            if total_cells_in_component != expected_cell_count_if_solid:
                continue

            # carve out everything except the outer ring
            for r in range(min_row + 1, max_row):
                for c in range(min_col + 1, max_col):
                    result[r][c] = 0

        return result

    # just runs the hollowing logic against training first to make sure
    # its actually the right rule before trusting it on the test grid
    def try_hollow_solid_rectangles(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            if in_grid.shape != out_grid.shape:
                return None

            expected = self.compute_hollow_solid_rectangles(in_grid)
            if not np.array_equal(expected, out_grid):
                return None

        return self.compute_hollow_solid_rectangles(test_input)


    # swaps a marker color for whatever other color is chilling in the
    # grid, and turns that other color into background. basically a two
    # color swap but one direction becomes blank
    def apply_marker_swap(self, grid, marker_color):
        grid_rows = len(grid)
        grid_cols = len(grid[0])

        # find whatever the "other" color is (not marker, not background)
        other_color = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                current_val = grid[r][c]
                if current_val != 0 and current_val != marker_color:
                    other_color = current_val

        if other_color == 0:
            return None

        result = []
        for r in range(grid_rows):
            new_row = []
            for c in range(grid_cols):
                current_val = grid[r][c]
                if current_val == marker_color:
                    new_row.append(other_color)
                else:
                    new_row.append(0)
            result.append(new_row)

        return np.array(result)

    # finds whichever color shows up in literally every training input,
    # thats our "marker". everything with the marker color gets swapped to
    # whatever the other color in that grid is, everything else goes blank.
    # figuring out the marker by intersecting colors across training pairs
    # is what makes this generalize instead of us hardcoding some color
    def try_swap_common_marker(self, training, test_input):
        common_colors = None
        for pair in training:
            in_grid = pair.get_input_data().data()
            grid_rows = len(in_grid)
            grid_cols = len(in_grid[0])

            colors_in_this_grid = []
            for r in range(grid_rows):
                for c in range(grid_cols):
                    current_val = in_grid[r][c]
                    if current_val != 0 and current_val not in colors_in_this_grid:
                        colors_in_this_grid.append(current_val)

            if common_colors is None:
                common_colors = colors_in_this_grid
            else:
                # only keep colors that show up in both the running list
                # and this grid's list
                still_common = []
                for color in common_colors:
                    if color in colors_in_this_grid:
                        still_common.append(color)
                common_colors = still_common

        # needs to be exactly one shared color for this rule to make sense
        if common_colors is None or len(common_colors) != 1:
            return None

        marker_color = common_colors[0]

        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            if in_grid.shape != out_grid.shape:
                return None

            expected = self.apply_marker_swap(in_grid, marker_color)
            if expected is None:
                return None
            if not np.array_equal(expected, out_grid):
                return None

        return self.apply_marker_swap(test_input, marker_color)


    # quick check, is this sub grid literally just all zeros
    def is_all_background(self, sub_grid):
        sub_rows = len(sub_grid)
        sub_cols = len(sub_grid[0])
        for r in range(sub_rows):
            for c in range(sub_cols):
                if sub_grid[r][c] != 0:
                    return False
        return True

    # given a grid thats split in half with one side totally blank, mirror
    # the populated half into the blank side. building the final grid by
    # hand row by row. 
    def apply_reflect_fill(self, grid, config):
        grid_rows = len(grid)
        grid_cols = len(grid[0])

        if config == 'top_blank' or config == 'bottom_blank':
            if grid_rows % 2 != 0:
                return None
            half_point = grid_rows // 2
            top_half = grid[0:half_point, :]
            bottom_half = grid[half_point:grid_rows, :]

            if config == 'top_blank':
                if not self.is_all_background(top_half):
                    return None
                # mirror bottom_half upside down to make the new top
                mirrored_top = np.flipud(bottom_half)

                result_rows = []
                for r in range(len(mirrored_top)):
                    result_rows.append(mirrored_top[r])
                for r in range(len(bottom_half)):
                    result_rows.append(bottom_half[r])

                return np.array(result_rows)
            
            else:
                if not self.is_all_background(bottom_half):
                    return None
                mirrored_bottom = np.flipud(top_half)

                result_rows = []
                for r in range(len(top_half)):
                    result_rows.append(top_half[r])
                for r in range(len(mirrored_bottom)):
                    result_rows.append(mirrored_bottom[r])

                return np.array(result_rows)

        if config == 'left_blank' or config == 'right_blank':
            if grid_cols % 2 != 0:
                return None
            half_point = grid_cols // 2
            left_half = grid[:, 0:half_point]
            right_half = grid[:, half_point:grid_cols]

            if config == 'left_blank':
                if not self.is_all_background(left_half):
                    return None
                mirrored_left = np.fliplr(right_half)

                # build result row by row, gluing the mirrored left piece
                # and the right half together for each row
                result_rows = []
                for r in range(len(right_half)):
                    combined_row = []
                    for c in range(len(mirrored_left[r])):
                        combined_row.append(mirrored_left[r][c])
                    for c in range(len(right_half[r])):
                        combined_row.append(right_half[r][c])
                    result_rows.append(combined_row)

                return np.array(result_rows)
            else:
                if not self.is_all_background(right_half):
                    return None
                mirrored_right = np.fliplr(left_half)

                result_rows = []
                for r in range(len(left_half)):
                    combined_row = []
                    for c in range(len(left_half[r])):
                        combined_row.append(left_half[r][c])
                    for c in range(len(mirrored_right[r])):
                        combined_row.append(mirrored_right[r][c])
                    result_rows.append(combined_row)

                return np.array(result_rows)

        return None

    # tries all four blank half configs (top, bottom, left, right) against
    # training, uses whichever one actually works consistently, then
    # applies that same config to the test grid
    def try_reflect_fill_half(self, training, test_input):
        matched_config = None
        config_options = ['top_blank', 'bottom_blank', 'left_blank', 'right_blank']

        for config in config_options:
            all_pairs_match = True

            for pair in training:
                in_grid = pair.get_input_data().data()
                out_grid = pair.get_output_data().data()

                if in_grid.shape != out_grid.shape:
                    all_pairs_match = False
                    break

                expected = self.apply_reflect_fill(in_grid, config)
                if expected is None:
                    all_pairs_match = False
                    break
                if not np.array_equal(expected, out_grid):
                    all_pairs_match = False
                    break

            if all_pairs_match:
                matched_config = config
                break

        if matched_config is None:
            return None

        return self.apply_reflect_fill(test_input, matched_config)
    
    # figures out what the "background" color actually is for this
    # specific grid, instead of just blindly assuming its always 0. some
    # problems use a totally different color as their blank canvas (like
    # b2862040 which uses 9 everywhere instead of 0). background is
    # whatever color shows up the most across the whole grid, since the
    # blank canvas color is basically always gonna dominate by a mile
    # compared to any of the actual shapes drawn on top of it
    def find_most_common_color(self, grid):
        grid_rows = len(grid)
        grid_cols = len(grid[0])

        # tally up how many times each color shows up, 0 included, we're
        # not filtering anything out here since we actually want to know
        # if 0 or something else wins
        color_counts = {}
        for r in range(grid_rows):
            for c in range(grid_cols):
                val = grid[r][c]
                if val not in color_counts:
                    color_counts[val] = 0
                color_counts[val] = color_counts[val] + 1

        # walk the tally and just keep track of whichever color has the
        # biggest count so far
        most_common_color = 0
        highest_count_seen = 0
        for color in color_counts:
            count_for_color = color_counts[color]
            if count_for_color > highest_count_seen:
                highest_count_seen = count_for_color
                most_common_color = color

        return most_common_color


    # this one's for finding shapes that "trap" a pocket of background
    # inside themselves, kinda like a donut vs a stick. flood fill in from
    # every background cell sitting on the outer edge of the grid, only
    # moving through background cells (never through shape cells). any
    # background cell that never gets touched by this flood fill is
    # "trapped" since there's no path from it back out to the edge without
    # crossing a shape. background param lets us pass in whatever color
    # this specific grid actually uses as its blank canvas color, since it
    # might not be 0
    def find_border_reachable_background(self, grid, background=0):
        grid_rows = len(grid)
        grid_cols = len(grid[0])

        reachable = []
        for r in range(grid_rows):
            reachable_row = []
            for c in range(grid_cols):
                reachable_row.append(False)
            reachable.append(reachable_row)

        # grab every background cell sitting right on the outer edge of
        # the grid, these are our starting points since anything touching
        # the actual border of the grid is definitely "outside" everything
        starting_cells = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                on_border = r == 0 or r == grid_rows - 1 or c == 0 or c == grid_cols - 1
                if on_border and grid[r][c] == background:
                    starting_cells.append((r, c))

        stack = []
        for cell in starting_cells:
            start_r = cell[0]
            start_c = cell[1]
            if not reachable[start_r][start_c]:
                reachable[start_r][start_c] = True
                stack.append((start_r, start_c))

        # standard flood fill with a stack, only allowed to step onto
        # cells that are also background color, shape cells act like walls
        while len(stack) > 0:
            r, c = stack.pop()

            up_r = r - 1
            up_c = c
            if up_r >= 0 and grid[up_r][up_c] == background and not reachable[up_r][up_c]:
                reachable[up_r][up_c] = True
                stack.append((up_r, up_c))

            down_r = r + 1
            down_c = c
            if down_r < grid_rows and grid[down_r][down_c] == background and not reachable[down_r][down_c]:
                reachable[down_r][down_c] = True
                stack.append((down_r, down_c))

            left_r = r
            left_c = c - 1
            if left_c >= 0 and grid[left_r][left_c] == background and not reachable[left_r][left_c]:
                reachable[left_r][left_c] = True
                stack.append((left_r, left_c))

            right_r = r
            right_c = c + 1
            if right_c < grid_cols and grid[right_r][right_c] == background and not reachable[right_r][right_c]:
                reachable[right_r][right_c] = True
                stack.append((right_r, right_c))

        return reachable


    # runs the border flood fill, then for every shape matching target_color
    # checks if its own bounding box has any background cell hiding inside
    # it that never got touched by the flood fill (meaning its trapped/
    # enclosed). if it finds a trapped pocket, that whole shape gets
    # swapped over to output_color. shapes with no trapped pocket (like
    # open/branching ones with gaps leading back out to the edge) get left
    # completely alone
    def compute_enclosed_shape_recolor(self, grid, target_color, output_color):
        # figure out this grid's actual background color instead of just
        # assuming its 0, since some problems (like b2862040) use a
        # totally different color as their blank canvas
        background_color = self.find_most_common_color(grid)

        reachable = self.find_border_reachable_background(grid, background_color)
        result = grid.copy()
        components = self.get_connected_components(grid, background_color)

        for component in components:
            if component['color'] != target_color:
                continue

            min_row = component['min_row']
            max_row = component['max_row']
            min_col = component['min_col']
            max_col = component['max_col']

            # scan the shape's own bounding box looking for any background
            # cell that the flood fill never reached, that's our trapped
            # pocket signal
            has_trapped_pocket = False
            for r in range(min_row, max_row + 1):
                for c in range(min_col, max_col + 1):
                    if grid[r][c] == background_color and not reachable[r][c]:
                        has_trapped_pocket = True

            if has_trapped_pocket:
                for cell in component['cells']:
                    cell_r = cell[0]
                    cell_c = cell[1]
                    result[cell_r][cell_c] = output_color

        return result


    # first figures out what color swap we're even looking for by diffing
    # every input cell against its matching output cell across training
    # (should be exactly one consistent target_color -> output_color swap
    # everywhere something changed). then double checks the enclosed shape
    # logic actually reproduces every single training pair exactly before
    # trusting it enough to run on the real test grid
    def try_enclosed_shape_recolor(self, training, test_input):
        target_color = 0
        output_color = 0

        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            if in_grid.shape != out_grid.shape:
                return None

            grid_rows = len(in_grid)
            grid_cols = len(in_grid[0])
            for r in range(grid_rows):
                for c in range(grid_cols):
                    if in_grid[r][c] != out_grid[r][c]:
                        if target_color == 0:
                            # first difference we've seen, lock in what the
                            # swap should be
                            target_color = in_grid[r][c]
                            output_color = out_grid[r][c]
                        else:
                            # any later difference has to match that exact
                            # same swap, otherwise this isn't a consistent
                            # rule and we bail
                            if in_grid[r][c] != target_color or out_grid[r][c] != output_color:
                                return None

        if target_color == 0 or output_color == 0:
            return None

        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            expected = self.compute_enclosed_shape_recolor(in_grid, target_color, output_color)
            if not np.array_equal(expected, out_grid):
                return None

        return self.compute_enclosed_shape_recolor(test_input, target_color, output_color)
    
    # splits a grid into 3 or more panels along whichever single axis has
    # separator lines going on (reuses find_grid_lines/get_panel_segments
    # from the histogram function up above), then does a priority merge
    # cell by cell -- first panel in line wins if its cell is nonzero,
    # otherwise fall through to the next panel, all the way down till we
    # hit a nonzero or run out of panels
    def compute_multi_panel_priority_merge(self, grid):
        row_lines, col_lines = self.find_grid_lines(grid)

        has_row_lines = len(row_lines) > 0
        has_col_lines = len(col_lines) > 0

        # only handle the case where exactly one axis has separator lines
        # going on, a grid with lines on both axes is too ambiguous for
        # this specific rule to make sense
        if has_row_lines and has_col_lines:
            return None
        if not has_row_lines and not has_col_lines:
            return None

        # slice out each panel along whichever axis actually had lines
        panels = []
        if has_col_lines:
            segments = self.get_panel_segments(len(grid[0]), col_lines)
            for segment in segments:
                seg_start = segment[0]
                seg_end = segment[1]
                panels.append(grid[:, seg_start:seg_end])
        else:
            segments = self.get_panel_segments(len(grid), row_lines)
            for segment in segments:
                seg_start = segment[0]
                seg_end = segment[1]
                panels.append(grid[seg_start:seg_end, :])

        # need at least 2 panels for a "merge" to even make sense here
        if len(panels) < 2:
            return None

        # every panel needs to be the exact same shape for this to work,
        # otherwise theres no consistent cell by cell comparison possible
        panel_rows = len(panels[0])
        panel_cols = len(panels[0][0])
        for panel in panels:
            if len(panel) != panel_rows or len(panel[0]) != panel_cols:
                return None

        # walk every cell position, and for each one check the panels in
        # order (left to right, or top to bottom), first nonzero value we
        # hit wins that cell
        result = []
        for r in range(panel_rows):
            result_row = []
            for c in range(panel_cols):
                winning_value = 0
                for panel in panels:
                    if panel[r][c] != 0:
                        winning_value = panel[r][c]
                        break
                result_row.append(winning_value)
            result.append(result_row)

        return np.array(result)

    # just runs the priority merge logic against training first to make
    # sure its actually the right rule (right shape, right values) before
    # trusting it enough to run on the real test grid
    def try_multi_panel_priority_merge(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            expected = self.compute_multi_panel_priority_merge(in_grid)
            if expected is None:
                return None
            if expected.shape != out_grid.shape:
                return None
            if not np.array_equal(expected, out_grid):
                return None

        return self.compute_multi_panel_priority_merge(test_input)


    # just counts up how many times each color shows up anywhere in the
    # grid, no panels or separators involved at all here, this is way
    # simpler than the histogram function up above. output ends up being a
    # bar chart, tallest bar (most common color) goes on the left, columns
    # get shorter heading right as we run out of frequency
    def compute_color_frequency_bars(self, grid):
        grid_rows = len(grid)
        grid_cols = len(grid[0])

        color_counts = {}
        for r in range(grid_rows):
            for c in range(grid_cols):
                val = grid[r][c]
                if val == 0:
                    continue
                if val not in color_counts:
                    color_counts[val] = 0
                color_counts[val] = color_counts[val] + 1

        if len(color_counts) == 0:
            return None

        # build a list of (count, color) pairs so we've got something we
        # can actually sort by count
        color_count_pairs = []
        for color in color_counts:
            count = color_counts[color]
            color_count_pairs.append((count, color))

        # manual descending bubble sort by count, list's tiny here (max 10
        # colors ever) so this is plenty fast, no need for anything fancy
        num_pairs = len(color_count_pairs)
        for i in range(num_pairs):
            for j in range(num_pairs - 1 - i):
                if color_count_pairs[j][0] < color_count_pairs[j + 1][0]:
                    temp_pair = color_count_pairs[j]
                    color_count_pairs[j] = color_count_pairs[j + 1]
                    color_count_pairs[j + 1] = temp_pair

        tallest_count = color_count_pairs[0][0]
        num_colors = len(color_count_pairs)

        # build a blank grid sized to fit the tallest bar (rows) by however
        # many distinct colors we found (cols)
        result = []
        for r in range(tallest_count):
            row = []
            for c in range(num_colors):
                row.append(0)
            result.append(row)

        # for each color's column, fill from the bottom up (row 0 down to
        # count-1) with that color, taller bars just naturally end up
        # filling more rows
        for col_index in range(num_colors):
            count_for_col = color_count_pairs[col_index][0]
            color_for_col = color_count_pairs[col_index][1]
            for r in range(count_for_col):
                result[r][col_index] = color_for_col

        return np.array(result)

    # just runs the color counting logic against training first to make
    # sure its actually the right rule before trusting it on the test grid
    def try_color_frequency_bars(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            expected = self.compute_color_frequency_bars(in_grid)
            if expected is None:
                return None
            if expected.shape != out_grid.shape:
                return None
            if not np.array_equal(expected, out_grid):
                return None

        return self.compute_color_frequency_bars(test_input)

    # this one's for problems where 4 lonely single-pixel "marker" dots
    # sit at the exact corners of an invisible rectangle, and theres a
    # shape of a different color living somewhere inside that rectangle.
    # the crop we want is the INTERIOR of that rectangle (one step in
    # from each corner, not counting the corner row/col itself), and we
    # repaint the shape using the marker color instead of its own color
    def compute_crop_shape_recolor_by_marker(self, grid):
        components = self.get_connected_components(grid)

        if len(components) == 0:
            return None

        # bucket every component by color again, same deal as before
        components_by_color = {}
        for component in components:
            color = component['color']
            if color not in components_by_color:
                components_by_color[color] = []
            components_by_color[color].append(component)

        # the marker color needs to form EXACTLY 4 components, and every
        # single one of them needs to be just 1 cell (four lonely dots).
        # walk through each color and check for that specific pattern
        marker_color = 0
        marker_positions = []
        for color in components_by_color:
            blob_list = components_by_color[color]
            if len(blob_list) == 4:
                all_single_cells = True
                for blob in blob_list:
                    if len(blob['cells']) != 1:
                        all_single_cells = False
                if all_single_cells:
                    marker_color = color
                    for blob in blob_list:
                        marker_positions.append(blob['cells'][0])

        if marker_color == 0:
            return None

        # shape color is just whatever the other nonzero color is
        shape_color = 0
        for color in components_by_color:
            if color != marker_color:
                shape_color = color

        if shape_color == 0:
            return None

        # find the min/max row and col across our 4 marker dots, this
        # gives us the rectangle they're sitting at the corners of
        frame_min_row = marker_positions[0][0]
        frame_max_row = marker_positions[0][0]
        frame_min_col = marker_positions[0][1]
        frame_max_col = marker_positions[0][1]
        for position in marker_positions:
            pos_row = position[0]
            pos_col = position[1]
            if pos_row < frame_min_row:
                frame_min_row = pos_row
            if pos_row > frame_max_row:
                frame_max_row = pos_row
            if pos_col < frame_min_col:
                frame_min_col = pos_col
            if pos_col > frame_max_col:
                frame_max_col = pos_col

        # double check the 4 dots are actually sitting at real corners
        # and not just scattered randomly, if the rectangle is too small
        # to even have an interior this rule doesn't apply either
        if frame_max_row - frame_min_row < 2:
            return None
        if frame_max_col - frame_min_col < 2:
            return None

        # the crop we actually want is the INTERIOR of that rectangle,
        # one step in from each corner row/col, not including the
        # corners themselves
        interior_min_row = frame_min_row + 1
        interior_max_row = frame_max_row - 1
        interior_min_col = frame_min_col + 1
        interior_max_col = frame_max_col - 1

        box_height = interior_max_row - interior_min_row + 1
        box_width = interior_max_col - interior_min_col + 1

        # build the crop by hand, plain python the whole way. wherever
        # we see the shape's original color inside the interior, paint
        # the marker color there instead. everything else stays 0
        result = []
        for r in range(box_height):
            result_row = []
            for c in range(box_width):
                grid_r = interior_min_row + r
                grid_c = interior_min_col + c
                if grid[grid_r][grid_c] == shape_color:
                    result_row.append(marker_color)
                else:
                    result_row.append(0)
            result.append(result_row)

        return np.array(result)

    # wrapper: checks the frame interior crop logic against every
    # training pair first, only trusts it enough to run on the real test
    # grid once every single pair lines up exactly
    def try_crop_shape_recolor_by_marker(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            expected = self.compute_crop_shape_recolor_by_marker(in_grid)
            if expected is None:
                return None
            if expected.shape != out_grid.shape:
                return None
            if not np.array_equal(expected, out_grid):
                return None

        return self.compute_crop_shape_recolor_by_marker(test_input)

    # for problems where a shape has one single cell inside it
    # swapped out for a different marker color, kinda like a
    # arrow poking into the shape telling us which way to shoot. we find
    # that marker cell, figure out which direction points from the
    # marker through the thickest part of the shape, then walk that
    # direction starting at the marker: hop over any shape colored cells
    # we bump into (thats us tunneling through the shape's body), and
    # once we pop out the other side into open background we start
    # filling marker color all the way out til we hit the edge of the
    # grid, like a laser beam shooting through and past the shape
    def compute_marker_ray_cast(self, grid):
        grid_rows = len(grid)
        grid_cols = len(grid[0])

        color_counts = {}
        color_positions = {}
        for r in range(grid_rows):
            for c in range(grid_cols):
                val = grid[r][c]
                if val == 0:
                    continue
                if val not in color_counts:
                    color_counts[val] = 0
                    color_positions[val] = []
                color_counts[val] = color_counts[val] + 1
                color_positions[val].append((r, c))

        marker_color = 0
        marker_row = 0
        marker_col = 0
        shape_color = 0
        for color in color_counts:
            if color_counts[color] == 1:
                marker_color = color
                marker_row = color_positions[color][0][0]
                marker_col = color_positions[color][0][1]
            else:
                shape_color = color

        if marker_color == 0 or shape_color == 0:
            return None

        shape_cells = color_positions[shape_color]
        total_row = 0
        total_col = 0
        for cell in shape_cells:
            total_row = total_row + cell[0]
            total_col = total_col + cell[1]
        centroid_row = total_row / len(shape_cells)
        centroid_col = total_col / len(shape_cells)

        row_offset = centroid_row - marker_row
        col_offset = centroid_col - marker_col

        direction_row = 0
        direction_col = 0
        if abs(row_offset) > abs(col_offset):
            if row_offset > 0:
                direction_row = 1
            else:
                direction_row = -1
        else:
            if col_offset > 0:
                direction_col = 1
            else:
                direction_col = -1

        result = []
        for r in range(grid_rows):
            result_row = []
            for c in range(grid_cols):
                result_row.append(grid[r][c])
            result.append(result_row)

        # fire off the shared ray caster starting at the marker, tunnel
        # through shape colored cells (skip_color), paint everything else
        self.cast_ray_and_paint(result, grid_rows, grid_cols, marker_row, marker_col, direction_row, direction_col, marker_color, shape_color)

        return np.array(result)

    # wrapper: runs the ray cast logic against every training pair first
    # to make sure its actually the right rule before we trust it enough
    # to run on the real test grid
    def try_marker_ray_cast(self, training, test_input):

        for pair in training:

            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            if in_grid.shape != out_grid.shape:
                return None

            expected = self.compute_marker_ray_cast(in_grid)
            if expected is None:
                return None
            if not np.array_equal(expected, out_grid):
                return None

        return self.compute_marker_ray_cast(test_input)
    
        # for problems with exactly two colored blocks sitting somewhere on
    # the grid. turns out the rule here has nothing to do with where the
    # blocks are relative to each other, its just based on which color
    # each block is. whichever block is the SMALLER color number shoots
    # a diagonal trail from its own top-left corner going up-left, and
    # whichever block is the BIGGER color number shoots a trail from its
    # own bottom-right corner going down-right. reuses the same shared
    # ray caster as the marker ray cast function above, just with a way
    # simpler (and kinda surprising) way of picking the starting corner
    # and direction
    def compute_two_block_diagonal_trails(self, grid):
        grid_rows = len(grid)
        grid_cols = len(grid[0])

        components = self.get_connected_components(grid)

        # this rule only makes sense with exactly 2 separate colored
        # blobs on the grid, bail if thats not the case
        if len(components) != 2:
            return None

        first_component = components[0]
        second_component = components[1]

        # figure out which component has the smaller color number and
        # which has the bigger one
        if first_component['color'] < second_component['color']:
            small_color_component = first_component
            big_color_component = second_component
        else:
            small_color_component = second_component
            big_color_component = first_component

        # build a plain python copy of the grid to work with
        result = []
        for r in range(grid_rows):
            result_row = []
            for c in range(grid_cols):
                result_row.append(grid[r][c])
            result.append(result_row)

        # smaller color: shoot from its top-left corner, heading up-left
        small_color = small_color_component['color']
        small_start_row = small_color_component['min_row']
        small_start_col = small_color_component['min_col']
        self.cast_ray_and_paint(result, grid_rows, grid_cols, small_start_row, small_start_col, -1, -1, small_color, None)

        # bigger color: shoot from its bottom-right corner, heading down-right
        big_color = big_color_component['color']
        big_start_row = big_color_component['max_row']
        big_start_col = big_color_component['max_col']
        self.cast_ray_and_paint(result, grid_rows, grid_cols, big_start_row, big_start_col, 1, 1, big_color, None)

        return np.array(result)

    # wrapper: runs the two block diagonal trail logic against every
    # training pair first to make sure its actually the right rule
    # before we trust it enough to run on the real test grid
    def try_two_block_diagonal_trails(self, training, test_input):
        for pair in training:
            in_grid = pair.get_input_data().data()
            out_grid = pair.get_output_data().data()

            if in_grid.shape != out_grid.shape:
                return None

            expected = self.compute_two_block_diagonal_trails(in_grid)
            if expected is None:
                return None
            if not np.array_equal(expected, out_grid):
                return None

        return self.compute_two_block_diagonal_trails(test_input)
    
        # this is the shared "walk in a straight line and paint stuff" logic
    # that both cardinal rays (straight up/down/left/right) and diagonal
    # rays (like corner to corner) can use. starts one step past
    # start_row/start_col in whatever direction we're given, and keeps
    # walking one step at a time til it falls off the edge of the grid.
    # if skip_color is set, cells matching that color get tunneled
    # through without getting painted (thats for shooting through a
    # shape's own body). if skip_color is None, literally every cell
    # along the way gets painted, no exceptions. mutates result in place
    # instead of returning a new grid, since result already exists as a
    # plain python list of lists by the time we call this
    def cast_ray_and_paint(self, result, grid_rows, grid_cols, start_row, start_col, direction_row, direction_col, paint_color, skip_color):
        current_row = start_row + direction_row
        current_col = start_col + direction_col

        while 0 <= current_row < grid_rows and 0 <= current_col < grid_cols:
            if skip_color is None or result[current_row][current_col] != skip_color:
                result[current_row][current_col] = paint_color
            current_row = current_row + direction_row
            current_col = current_col + direction_col



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
            self.try_panel_overlay(training, test_input),
            self.try_shape_fit_overlay(training, test_input),
            self.try_symmetry_repair_regions(training, test_input),
            self.try_panel_color_histogram(training, test_input),
            self.try_marker_count_grid(training, test_input),
            self.try_crop_and_swap_colors(training, test_input),
            self.try_hollow_solid_rectangles(training, test_input),
            self.try_swap_common_marker(training, test_input),
            self.try_reflect_fill_half(training, test_input),
            self.try_enclosed_shape_recolor(training, test_input),
            self.try_multi_panel_priority_merge(training, test_input),
            self.try_color_frequency_bars(training, test_input),
            self.try_crop_shape_recolor_by_marker(training, test_input),
            self.try_marker_ray_cast(training, test_input),
            self.try_two_block_diagonal_trails(training, test_input),
            self.try_bounding_box(test_input),
        ]

        for result in transform_attempts:
            if result is not None:
                predictions.append(result)
                return predictions

        return predictions