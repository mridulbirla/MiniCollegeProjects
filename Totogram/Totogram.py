# We start with the state which not to bad and not too good in which we take the middle element as the root and then we
# find the maximum difference edge and try to replace child with the node which will decrease the difference and
# we put tha child with the score of the tree ( ie the max difference in the tree) in the priority queue and iterate by
#  dequeue from the queue which is more promising.



import copy
import Queue
import sys
import itertools
from Node import Node
import time
class Queue_class:

    def __init__(self,node_obj, s_dict, priority):
        self.priority = priority
        self.obj = node_obj
        self.s_dict = s_dict

    def __cmp__(self, other):
        return cmp(self.priority, other.priority)

# This function has combined various functionality namely finding the maximum difference child, BFS, creating dictionary
def traverse_tree(tree, flag):
    q = Queue.Queue()
    q.put([tree, ''])
    diff = 0
    max_node = {}
    tree_dict = {}
    lower_child_dict = {}
    root_val = tree.val
    bfs_tree = []
    while not q.empty():
        pair = q.get()
        t = pair[0]
        if "Dictionary" in flag:
            tree_dict[t.val] = t
        elif "Lower Child" in flag and t.val < root_val:
            lower_child_dict[t.val] = pair
        elif "BFS" in flag:
            bfs_tree.append(str(t.val))
        for k in t.children:
            if "Maximum" in flag:
                m = abs(t.val - k.val)
                if (m >= diff):
                    diff = m
                    max_node.setdefault(m, []).append([t, k])

                # Used below to avoid putting leaf node in Queue
                if k.children is not None:
                    q.put([k, t])
            else:
                q.put([k, t])
    return [max_node,tree_dict,lower_child_dict,bfs_tree]
    '''
    if flag == "Maximum": return max_node
    if flag == "Dictionary": return tree_dict
    if flag == "Lower Child": return lower_child_dict
    if flag == "BFS": return bfs_tree
    '''


# This is a recursive function to create a tree which is the start state for the program.

def create_b_tree(sub_tree,parent,count=0):
    mid_element = len(sub_tree) / 2
    if count==1: mid_element= len(sub_tree)-1
    if count==3: mid_element= 0
    new_root = Node(sub_tree.pop(mid_element),parent)
    if len(sub_tree) == 2:
        new_root.add_children(Node(sub_tree[0], new_root))
        new_root.add_children(Node(sub_tree[1], new_root))
    else:
        new_root.add_children(create_b_tree(sub_tree[0:len(sub_tree) / 2],new_root))
        new_root.add_children(create_b_tree(sub_tree[len(sub_tree) / 2: len(sub_tree)],new_root))
    return new_root


def create_initial_tree(initial_list_size):
    initial_list = range(1, (initial_list_size + 1))
    root = Node(initial_list.pop((initial_list_size / 2) - 1))
    step = len(initial_list) / 3
    chunks = [initial_list[x: (x + step)] for x in range(0, len(initial_list), step)]
    count=1
    # For each child of the root tree create Binary tree
    for i in chunks:
        root.add_children(create_b_tree(i,root,count))
        if count==2:
            first_sub_tree_max=i[len(i)-1]
            third_sub_tree_min=i[0]
        count+=1
    return root


def is_visited(v, c):
    return c in v


def swap_nodes(target_node, main_node, initial_node):

    # Change the parents of the swapping nodes

    p = target_node.parent
    target_node.parent = main_node

    # If the target node is the child of the initial node then in that case the target node becomes parent of initial
    # node.Else just swap the parents

    if initial_node == p:
        initial_node.parent = target_node
    else:
        initial_node.parent = p
        p.children.remove(target_node)
        p.children.append(initial_node)
        p.sort_children()

    # The children of initial node become children of the target node and like wise also if target and
    c = target_node.get_children()
    target_node.children = initial_node.children
    initial_node.children = c

    # If the initial node and target node are parent and child in that case for the child node remove the children
    # as itself

    if target_node in target_node.get_children():
        target_node.get_children().remove(target_node)
        target_node.children.append(initial_node)
        target_node.sort_children()
    # For the main node make the target node as its child and remove the initial node at its child and the sort it
    try:
        main_node.children.remove(initial_node)
        main_node.children.append(target_node)
    except ValueError:
        print "A"

    main_node.sort_children()
    for c in target_node.children:
        c.parent=target_node
    for c in initial_node.children:
        c.parent=initial_node


def generate_minimal_tree(tree,p,l):
    optimal_queue = Queue.PriorityQueue()
    visited_tree = []
    score_dict = (traverse_tree(tree, ["Maximum"]))[0]
    root_value=tree.val
    optimal_queue.put(Queue_class(tree,score_dict,max(score_dict.keys())))
    global_minima = {max(score_dict.keys()): [tree]}
    min_score_dict={}
    min_score_dict_max_size={}

    for i in range(1,1300):
        min_score_dict[i]=0
        min_score_dict_max_size[i]=400
    count =0
    while True:
        count+=1
        if optimal_queue.empty():
            break
        current_item = optimal_queue.get()
        current_tree = current_item.obj
        score_dict = current_item.s_dict
        current_score = max(score_dict.keys())
        max_dict = score_dict[current_score]
        travers_list = traverse_tree(current_tree, ["BFS"])
        if (l==3 and current_score==3) or (l==4 and current_score==4) or (l==5 and current_score==6):
            print current_score
            print ' '.join(travers_list[3])
            break
        if count >10:
            p=min(global_minima.keys())
            print p
            print (traverse_tree(global_minima[p][0], ["BFS"])[3    ])
            break
        # For each maximum edge
        for m in max_dict:
            if m[1].val > m[0].val:
                start_node=(m[0].val + 1)
                end_node=m[1].val

            elif m[1].val < m[0].val:
                start_node=m[1].val + 1
                end_node=m[0].val
            o=1
            for i in range(start_node, end_node):
                temp_max=1000000
                '''
                o += 1
                if o>10:
                   break
                '''
                if i==root_value:
                    continue
                temp_tree = copy.deepcopy(current_tree)
                tree_dict = (traverse_tree(temp_tree, ["Dictionary"]))[1]
                if i  in tree_dict[m[1].val].eligible_swap:
                    continue
                swap_nodes(tree_dict[i], tree_dict[m[0].val], tree_dict[m[1].val])
                travers_list = traverse_tree(temp_tree, ["Maximum", "BFS"])
                score_dict = travers_list[0]
                bfs_tree = travers_list[3]
                if is_visited(visited_tree, bfs_tree):
                    continue
                temp_max = max(score_dict.keys())
                visited_tree.append(bfs_tree)
                #print temp_max ;print bfs_tree
                if temp_max <= min(global_minima.keys()):  # and int(bfs_tree[len(bfs_tree)-1])==p:
                    if min_score_dict[temp_max] < min_score_dict_max_size[temp_max]:
                        # generate_swap_eligible(temp_tree)
                        # optimal_queue.put([temp_tree, score_dict],temp_max)
                        optimal_queue.put(Queue_class(temp_tree, score_dict, max(score_dict.keys())))
                        global_minima.setdefault(temp_max, []).append(temp_tree)
                        min_score_dict[temp_max]+=1
                       # print "%d :%d: %d " % (min(global_minima.keys()),temp_max, min_score_dict[temp_max])
                      #  print bfs_tree
                        # print min(global_minima.keys())


def generate_swap_eligible(root):

    t=[]
    for i in range(0,3):
        p=root.children[i]
        t.append(traverse_tree(p,["BFS"])[3])

    for i in range(0,3,2):
        temp_dict=traverse_tree(root.children[i],["Dictionary"])[1]
        for j in temp_dict.keys():
            temp_dict[j].eligible_swap=t[1]+t[i]



def main():
    l= int(sys.argv[1])
    if l==1:
        print 1
        return
    if l==2:
        o = list(xrange(1,5))
        min_diff=100
        min_diff_list=[]
        perm_iterator=itertools.permutations(o)
        for i in perm_iterator:
            key=i[0]
            local_diff=0
            for j in range(1,4):
                diff=abs(i[j]-i[0])
                if diff> local_diff:
                    local_diff=diff

            if local_diff< min_diff:
                min_diff=local_diff
                min_diff_list=i

        print min_diff
        print min_diff_list
    elif l > 2:
        # l=6
        k=3
        i=3
        p=4
        while i <= l:
           k=k*2
           p=p+k
           i+=1

        r = create_initial_tree(p)
        generate_swap_eligible(r)
        generate_minimal_tree(r,p,l)

main()
