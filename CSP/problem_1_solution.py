__author__ = 'BlackPanther'

# Theory for problem: The problem comes under the category of Constraint Solving Problem.(CSP)

# To solve this problem, it can be considered that we have optimized the blind search (brute force method) at three levels.
#
# 1) We reduce the number of possible states which can be considered as initial states using all possible conditions
# given in the problem description.

# 2) We then reduce the number of possible states which can be considered as final states using all possible conditions
# given in the problem description.

# 3) There are some conditions in the problem which link the initial states to the final states. Using those conditions,
# now we use brute force method for each initial state to find a perfect goal state.
#
# References :
# http://aima.cs.berkeley.edu/2nd-ed/newchap05.pdf (After reading this, example studied of famous zebra puzzle)
# courses.cs.washington.edu/courses/.../CSP.ppt
# Examples referenced : N-Queen Problem and Map coloring problem

from itertools import permutations

import itertools

#initializing classes

class Name: elems = "F G H I J".split()
class Item: elems = "A B C D E".split()
class Street: elems = "K L M N O".split()

#indexing Them
for c in (Name, Item, Street):
    for i, e in enumerate(c.elems):
        exec "%s.%s = %d" % (c.__name__, e, i)

#filtering possible states for initial state
def is_possible_initialstate(item, street):

    if item and item[Name.F] != Item.D:
        return False
    if not item or not street:
        return True
    for i in xrange(5):
        if item[i] == Item.E and street[i] != Street.N:
            return False
        if item[i] == Item.A and street[i] != Street.M:
            return False
    return True

# filtering possible states for Final state
def is_possible_finalstate(item, street):

    if item and item[Name.F] == Item.D:
        return False
    if item and item[Name.I] == Item.B:
        return False
    if street and street[Name.G] == Street.K:
        return False
    if street and street[Name.H] == Street.O:
        return False

    if not item or not street:
        return True

    for i in xrange(5):
        if item[i] == Item.E and street[i] == Street.N:
            return False
        if item[i] == Item.A and street[i] == Street.M:
            return False
        if item[i] == Item.E and street[i] == Street.M:
            return False
    return True

# performing final checks to arrive at a final state given an initial state
def final_checks(initial_state, final_state):

            temp, temp1, temp2, temp3 = 0, 0, 0, 0

            for y in range(5):
                if initial_state[0][y] == 1 and final_state[0][y] != 2:
                    return False

            for x in range(5):
                if final_state[0][x] == 1 and initial_state[0][x] != final_state[0][3]:
                    return False

            if initial_state[0][4] != final_state[0][2]:
                return False

            for z in range(5):
                if final_state[1][z] == 2:
                    temp = z

            for w in range(5):
                if final_state[1][w] == 0:
                    temp1 = w

            for e in range(5):
                if initial_state[1][e] == 1:
                    temp2 = e

            if final_state[0][temp1] != initial_state[0][temp2]:
                return False

            for q in range(5):
                if final_state[0][q] == 4 and initial_state[0][q] != final_state[0][temp]:
                    return False

            for r in range(5):
                if initial_state[1][r] == 0 and final_state[0][1] != initial_state[0][r]:
                    return False

            for t in range(5):
                if final_state[1][t] == 4:
                    temp3 = t

            if final_state[0][temp3] != initial_state[0][2]:
                return False

            for p in range(5):
                if final_state[1][p] == 2 and initial_state[0][p] != 0:
                    return False
#assuming that the people dont move and remain at the same street
            if initial_state[1] != final_state[1]:
                return False

            return True

#function used to print output
def show_row(t, data):
  print "%6s: %12s%12s%12s%12s%12s" % (
    t.__name__, t.elems[data[0]],
    t.elems[data[1]], t.elems[data[2]],
    t.elems[data[3]], t.elems[data[4]])

#The main fuction()
def main():
#defining the state space
    perms = list(permutations(range(5)))
    initial_states = []
#reducing the initial state space and final state space
    for item in perms:
        if is_possible_initialstate(item, None):
            for street in perms:
                if is_possible_initialstate(item, street):
                    initial_states.append([item, street])
    final_states = []
    for item in perms:
        if is_possible_finalstate(item, None):
            for street in perms:
                if is_possible_finalstate(item, street):
                    final_states.append([item, street])

    new_initial_states = []
    new_final_states = []

#trying to find a goal state given an initial state

    for x in range(len(initial_states)):
        for y in range(len(final_states)):
            if (final_checks(initial_states[x], final_states[y])):
                new_initial_states.append(initial_states[x])
                new_final_states.append(final_states[y])

    #part of code could be used to check what input arrives in the final index list
    # print new_initial_states
    # print len(new_initial_states)
    #
    # print new_final_states
    # print len(new_final_states)

    print "The initial state is : when all items go to wrong people"
    show_row(Name, range(5))
    show_row(Item, new_initial_states[0][0])
    show_row(Street, new_initial_states[0][1])
    #
    print "The final state is : when all items go to correct people"
    show_row(Name, range(5))
    show_row(Item, new_final_states[0][0])
    show_row(Street, new_initial_states[0][1])

    print ("\n\nWhere A=Amplifier B=Banister C=Candelabrum D=Doorknob E=Elephant")
    print ("Where F=Frank G=George H=Heather I=Irene J=Jerry")
    print ("Where K=Kirkwood L=LakeAvenue M=Maxwell N=NorthAvenue O=OrangeDrive")
#calling the main function
main()

