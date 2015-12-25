
# In this variant of 15 puzzle , we defined 2 classes one is node that ie a block and all the 16 blocks of a puzzle form
#  a class called state. This state class will have all the 16 node class objects ,its heuristic weight, update
#  heuristic. Now First I read all the nodes from the input file.and then insert it into the sates object.
# I push the state object into priority queue. When I pop it I generate the 8 nodes by using operations
# such as shifting row 1 by left ,shifting row 2 by right etc.
# Firstly i used simple manhattan distance divided by 4 to make it admissible as without dividing it by 4 it wont be
# admissible as if we move the goal state's row 1 in right the manhattan distance is 5 which is non admissible bu when
# we divide it by 4 it becomes admissible Later i tries misplaced no of blocks/4 but this was not giving output
# in shorter time. The I changed the manhattan distance calculation.
# The change is  if a number is supposed to be at the any the 4 edges of the puzzle and it at the opposite edge with row/column same.Then
# in actual scenario the move required is one whereas traditional manhattan distance would be 3.
# So for those tile i used manhattan  as one and for all the sum of manhattan i divided it by for 4 to make it admissible .And to verify this I
# ran code for both manhattan one divided by four other not divided by 4.And the number of loops were less when i divided the manhattan by4

import Queue
import copy
import sys


class Nodes(object):
    def __init__(self, val, r, w):
        if val / 4.0 <= 1:
            self.expected_row = 1
        elif val % 4 == 0:
            self.expected_row = val / 4
        elif val / 4.0 > 0:
            self.expected_row = val / 4 + 1

        self.value = val

        if val % 4 == 0:
            self.expected_column = 4
        else:
            self.expected_column = (val % 4)
        self.actual_row = r
        self.actual_column = w

    # Below functions checks if the nodes is at the goal state
    def at_correct(self):
        if self.expected_row != self.actual_row or self.actual_column != self.expected_column:
            return False
        else:
            return True


class State(object):
    def __init__(self):

        self.nodes = []
        self.heuristic = 0.0
        self.priority = 0.0
        self.state_pic = {}
        self.move = []

    def __cmp__(self, other):
        return cmp(self.priority, other.priority)

    # The below function calculates manhattan distance for a node

    def cal_manhattan(self, n):

        if n.actual_row == n.expected_row and n.actual_column != n.expected_column and n.expected_column == 4 and n.actual_column == 1:
            return 1

        if n.actual_row != n.expected_row and n.actual_column == n.expected_column and n.expected_row == 4 and n.actual_row == 1:
            return 1

        if n.actual_row == n.expected_row and n.actual_column != n.expected_column and n.expected_column == 1 and n.actual_column == 4:
            return 1

        if n.actual_row != n.expected_row and n.actual_column == n.expected_column and n.expected_row == 1 and n.actual_row == 4:
            return 1

        return abs(n.actual_row - n.expected_row) + abs(n.actual_column - n.expected_column)


    # Below are the other heuristic function which were used
    '''
    def permutation_inversion(self, st):
        val = 0
        for key, value in st.state_pic.iteritems():
            for k, v in st.state_pic.iteritems():
                if v < value and k > key:
                    val += 1
        return val


    def out_of_order(self, n):
        i = 0
        if n.actual_row != n.expected_row:
            i = 1
        if n.actual_column != n.expected_column:
            i += 1
        return i

    def misplaced(self, n):
        i = 0
        if n.actual_row != n.expected_row or n.actual_column != n.expected_column:
            i += 1
        return i
    '''
    def cal_heuristic(self):
        i = 0
        for box in self.nodes:
            i += self.cal_manhattan(box)
            # i += (self.out_of_order(box))
            # i += self.misplaced(box)
            # i=self.permutation_inversion(self)

        self.heuristic = i / 4 + 1.0
        self.priority = i / 4 + 1.0

    # Insert Node in a state object
    def insert_node(self, n):
        self.nodes.append(n)
        self.state_pic[(n.actual_row - 1) * 4 + n.actual_column] = n.value
        # self.heuristic += self.cal_heuristic()

    # For debugging created below function for printing state
    def print_state(self):
        print "\n"
        for i in range(1, 16, 4):
            print "%d %d %d %d" % (
                self.state_pic[i], self.state_pic[i + 1], self.state_pic[i + 2], self.state_pic[i + 3])

    # Below function updates the path of the state
    def update_move(self, parent_path, own_path):
        self.move = own_path + parent_path

# Below function reads file input


def read_input(all_states, file_name):
    #    input_nodes=[]
    try:
        temp = [line.split() for line in open(file_name, "r")]
    #   print temp
    except IOError:
        print "The file does not exist, exiting gracefully"
        sys.exit(0)

    initial_state = State()
    i = 0

    for item in temp:
        i += 1
        j = 0
        for n in item:
            j += 1
            #            print n
            initial_state.insert_node(Nodes(int(n), int(i), int(j)))

    initial_state.cal_heuristic()
    return initial_state


def goal_test(test):
    for n in test.nodes:
        if not n.at_correct():
            return False

    return True


def generate_node(in_state, param, rc_no, dir, parent_path):
    i = 1
    temp_state = copy.deepcopy(in_state)
    temp_state.update_move(parent_path, [dir + str(rc_no)])
    for node in temp_state.nodes:
        if param == "row" and node.actual_row == rc_no:

            if dir == "R" and node.actual_column % 4 == 0:
                node.actual_column = 1
                temp_state.state_pic[(node.actual_row - 1) * 4 + node.actual_column] = node.value
            elif dir == "L" and node.actual_column % 4 == 1:
                node.actual_column = 4
                temp_state.state_pic[(node.actual_row - 1) * 4 + node.actual_column] = node.value
            elif dir == "R":
                node.actual_column += 1
                temp_state.state_pic[(node.actual_row - 1) * 4 + node.actual_column] = node.value
            elif dir == "L":
                node.actual_column -= 1
                temp_state.state_pic[(node.actual_row - 1) * 4 + node.actual_column] = node.value

            i += 1

        elif param == "column" and node.actual_column == rc_no:

            if dir == "U" and node.actual_row % 4 == 1:
                node.actual_row = 4
                temp_state.state_pic[(node.actual_row - 1) * 4 + node.actual_column] = node.value
            elif dir == "D" and node.actual_row % 4 == 0:
                node.actual_row = 1
                temp_state.state_pic[(node.actual_row - 1) * 4 + node.actual_column] = node.value
            elif dir == "U":
                node.actual_row -= 1
                temp_state.state_pic[(node.actual_row - 1) * 4 + node.actual_column] = node.value
            elif dir == "D":
                node.actual_row += 1
                temp_state.state_pic[(node.actual_row - 1) * 4 + node.actual_column] = node.value
            i += 1
        if i >= 5:
            break
    temp_state.cal_heuristic()

    return temp_state


def check_visited(all_state, temp):
    for n in all_state:
        uncommon_nodes = set(n.state_pic.items()) ^ set(temp.state_pic.items())
        #  print len(common_nodes)
        if len(uncommon_nodes) == 0:
            return True

    return False



# Main code execution starts from here
def main():
        all_states = []
        initial_state = []
        # create priority queue
        q = Queue.PriorityQueue()
        try:
            q.put(read_input(all_states, sys.argv[1]))
        except IndexError:
            print "Please enter file name"
            sys.exit(0)

        for i in range(1, 90000):

            current_state = q.get()
            # Check if it is a goal state
            result = goal_test(current_state)

            if result:
                #   print "Success"
                current_state.move.reverse()
                print ' '.join(current_state.move)
                break
            # Check whether a particular state has been visited or not.If it has been visited then don't expand
            # For logging purpose we have stored all the generated states to an array called all_states
            if not check_visited(all_states, current_state):
               # print "Loop No %d" % i
                all_states.append(current_state)
                #   print "The Below state has f(n) %d " % current_state.priority
                #current_state.print_state()
            else:
                continue

            # Below code generates a state and push it to a priority queue

            temp = generate_node(current_state, "row", 1, "R", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "row", 1, "L", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "row", 2, "R", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "row", 2, "L", current_state.move)
            #   temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "row", 3, "L", current_state.move)
            #  temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "row", 3, "R", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "row", 4, "L", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "row", 4, "R", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "column", 1, "U", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "column", 1, "D", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "column", 2, "U", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "column", 2, "D", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "column", 3, "U", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "column", 3, "D", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "column", 4, "U", current_state.move)
            # temp.print_state()
            q.put(temp)

            temp = generate_node(current_state, "column", 4, "D", current_state.move)
            q.put(temp)

main()
