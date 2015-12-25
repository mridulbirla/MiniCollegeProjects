
'''
The program implements the game Rameses.

1) We have used Minimax algorithm with an utility function to find the next best move.
   Assuming that we are the MAX player in the algorithm, the utility function used by us is

   util = (number of available moves (i.e), playing at this position doesnt result into terminal state or loss)
                                                +
          (The parity of the depth of the node for which we are calculating the utility value)

   Since, the basic idea behind the next best move for both the players is the same, we use negation of the best
   value for MAX to determine the best value for min and thus successfully implementing the minimax algorithm
   successfully.

2) Structure of the CODE:
    a) We take the input from the user (input values : string combination of board, the value of n, and time cut off)

    b) We have made a Board class, which generalizes a board object and creates key value pair for each node. (x,y)

    c) We then create a board object for the input and pass it to our minimax algorithm .

    d) After checking whether this is the terminal state or not, the minimax algorithm proceeds and in a recursive loop
       tries to determine to go as far as possible in the tree, and the evaluates the util function and starts
       backtracking the value of util using minimax algorithm.

    e) After it has established the best path for itself, it prints the desired output.

    f) A check for if the initial state is a terminal state has also been placed.

    References used :
    Artificial Intelligence: A Modern Approach : russell norvig
    (book refered from site : http://aima.cs.berkeley.edu/)
'''
from copy import deepcopy
import time
import sys
import random

class Board:
    def __init__(self, n):
        self.filled = 'X'
        # self.opponent = 'X'
        self.empty = '.'
        self.size = n
        self.fields = {}
        for x in range(self.size):
            for y in range(self.size):
                self.fields[x, y] = ""

    def getnumberofx(self, n):
        xcount = 0
        for x in range(n):
            for y in range(n):
                if self.fields[x, y] == 'x':
                    xcount += 1
        return xcount
    def getavailablerows(self, n):
        rowcount = 0
        for x in range(n):
            count = 0
            for y in range(n):
                if self.fields[x, y] == '.':
                    count += 1
            if count > 1:
                rowcount += 1
        return rowcount
    def getavailablecolumns(self, n):
        columncount = 0
        for y in range(n):
            count = 0
            for x in range(n):
                if self.fields[x, y] == '.':
                    count += 1
            if count > 1:
                columncount += 1
        return columncount
    def getavailablediagonals(self, n):
        diagonalcount = 0
        count = 0
        count1 = 0
        for x in range(n):
            if self.fields[x, x] == '.':
                count += 1
        if count > 1:
            diagonalcount += 1

        for y in range(n):
            z = 1
            if self.fields[y, (n-z)] == '.':
                count1 += 1
        if count1 > 1:
            diagonalcount += 1
        z += 1
        return diagonalcount

    def get_available_positions(self, n):
        a = 10000
        b = 10000
        count = 0
        for x in range(n):
            for y in range(n):
                temp = deepcopy(self)
                if (temp.fields[x, y] == '.') and ([x, y] != [a, b]):
                    temp.fields[x, y] = 'x'
                    a = x
                    b = y
                    if not test_if_terminal_nodes(temp, n):
                        count += 1
        return count

    def numrowfilled(self, n):
        rowcount = 0
        for x in range(n):
            count = 0
            for y in range(n):
                if self.fields[x, y] == 'x':
                    count += 1
            if count == (n):
                rowcount += 1
        return rowcount
    def numcolumnfilled(self, n):
        columncount = 0
        for y in range(n):
            count = 0
            for x in range(n):
                if self.fields[x, y] == 'x':
                    count += 1
            if count == (n):
                columncount += 1
        return columncount
    def numdiagonalfilled(self, n):
        diagonalcount = 0
        count = 0
        count1 = 0
        for x in range(n):
            if self.fields[x, x] == 'x':
                count += 1
        if count == n:
            diagonalcount += 1

        for y in range(n):
            z = 1
            if self.fields[y, (n-z)] == 'x':
                count1 += 1
            z += 1
        if count1 == (n):
            diagonalcount += 1

        return diagonalcount

#This function indexes a given input string into a state [x,y]
def makeState(input_state, n):
    board = Board(n)
    i = 0
    for x in range(n):
        for y in range(n):
            board.fields[x, y] = input_state[i]
            i += 1
    return board

#This function makes the successor states for a given input state
def makeSuccessors(state, n,cut_off_time):
    a = 10000
    b = 10000
    s = []
    for x in range(n):
        for y in range(n):
            temp = deepcopy(state)
            if (temp.fields[x, y] == '.') and ([x, y] != [a, b]):
                temp.fields[x, y] = 'x'
                a = x
                b = y
                s.append(temp)
    return s

#This is function to check if the current state is an input state
def test_if_terminal_nodes(state, n):
    if state.numcolumnfilled(n) == 0 and state.numrowfilled(n) == 0 and state.numdiagonalfilled(n) == 0:
        return False
    return True

    return True

def utility_function(state, n, depth):
    if depth % 2 == 0:
        util = state.get_available_positions(n) + depth
    else:
        util = -(state.get_available_positions(n) + depth)
    return util

def cut_off_test_time(h):
    start = time.time()
    if time.time() < start + int(h):
        return False
    return True

def checking_the_final_output(final, initial, n):
    for a in range(n):
        for b in range(n):
            if initial.fields[a, b] != final.fields[a, b]:
                return [a, b]

def print_output(final_board,size_of_board):
    str=''
    for i in range(0,size_of_board):
        for j in range(0,size_of_board):
            str += final_board.fields[i, j]
    return str

#def minimax_decision(state, n):
def min_max_value(state, n, depth,game_state, cut_off_time, flag=0):
    t = {}
    if test_if_terminal_nodes(state, n) or cut_off_test_time(cut_off_time):
        p = utility_function(state, n, depth)
        return p
    if game_state == "max":
        v = -10000000
        current_depth = depth + 1
        for s in makeSuccessors(state, n, cut_off_time):
            temp_val = min_max_value(s, n, current_depth, "min", cut_off_time)
            v = max(v, temp_val)
            if flag == 1:
                t.setdefault(temp_val, []).append(s)
        if flag == 1:
            return t
        if flag == 0:
            return v
    if game_state == "min":
        v = 10000000
        current_depth = depth + 1
        for s in makeSuccessors(state, n, cut_off_time):
            v = min(v, min_max_value(s, n, current_depth, "max", cut_off_time))
        if flag == 0:
            return v

def main():

    initial_str_input = sys.argv[2]
    size_of_board = int(sys.argv[1])
    cut_off_time = int(sys.argv[3])

    initial_list_input = []
    initial_list_input += initial_str_input

    print "Board Size Entered : ", size_of_board
    print "Current Board State Entered : ", initial_str_input
    print "Thinking!... Please wait"

    initial_state = makeState(initial_list_input, size_of_board)

    if test_if_terminal_nodes(initial_state, size_of_board):
        print "The input board is a terminal state"

    else:
        coder = min_max_value(initial_state, size_of_board, 0, "max", cut_off_time, 1)

        all_optimal_outputs = []

        if len(coder.keys()) == 0:
            print "Couldn't find solution in given time"

        else:
            for i in coder[max(coder.keys())]:
                templist = []
                for a in range(size_of_board):
                    for b in range(size_of_board):
                        templist.append(i.fields[a, b])
                all_optimal_outputs.append(templist)

        final_optimal_outut = makeState(random.choice(all_optimal_outputs), size_of_board)

        row_coloumn = []
        row_coloumn = checking_the_final_output(final_optimal_outut, initial_state, size_of_board)

        print "Hmm I\' d recommend putting your pebble at row %d column %d "% ((row_coloumn[0]+1), (row_coloumn[1]+1))
        print print_output(final_optimal_outut, size_of_board)

main()