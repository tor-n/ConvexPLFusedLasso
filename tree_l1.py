import numpy as np
import bisect
from collections import defaultdict

class treeNode():
    def __init__(self, depth=0, sol=[1], lambda_range=[0,float('inf')], lambda_range_type=['in','ex'], 
                 source_node_interval=None, step_move_to_sink=None, num_sink=0):
        self.depth = depth # the first level where this node splits from its parent
        self.sol = sol # cut solution at this node, len(sol) = N, each entry is either 1 (source node) or 0 (sink node)
        self.lambda_range = lambda_range
        self.lambda_range_type = lambda_range_type
        self.children = []
        self.source_node_interval = source_node_interval
        self.step_move_to_sink = step_move_to_sink # list of length N: i-th entry is j if graph node i is in S_j and T_{j+1} 
        self.num_sink = num_sink

class Tree():
 
    def __init__(self):
        self.root = treeNode(depth=0, sol=None)
        self.all_solutions = None
        self.all_lambda_ranges = None
        self.all_lambda_ranges_type = None
        '''
        self.sol = sol
        self.lambda_range = lambda_range
        self.depth = depth # the first level where this node splits from its parent
        self.depth_last = depth # the deepest level before this node brances out
        self.depth_max = depth_max # maximum depth of the tree that is rooted at this node
        self.child = []
        self.parent = None
        self.source_node_intervals = []
        '''
    
    def fit(self, weights, breakpoints):
        self.breakpoints = breakpoints
        self.weights = weights
        N = len(weights)
        A = np.zeros((N,N,N))
        order = np.argsort(breakpoints) # order[j] is the index of the j-th smallest breakpoint in the breakpoints array
        self.order = order
        for t in range(N):
            A[t:, :(order[t]+1), order[t]:] += weights[order[t]]
        self.arr = A
        self.max_depth = N
        self.root.sol = [1 for i in range(N)]
        self.root.source_node_interval = [0, N-1]
        self.root.step_move_to_sink = [N-1 for i in range(N)]
    
    def _solve(self, t, s_int):
        # input 1: t, depth in the tree (root: t=0), we will be solving the cut change from graph G_t to G_{t+1}
        # input 2: s_int, the s-interval of the graph G_t has s-interval containing node i_t being [s_int[0], s_int[1]]
        # get the solution (lambda ranges and different cut solutions (maybe the node intervals that move to T)) before intersecting them with each lambda interval
        # output: return 3 things
        # list sol : list of node intervals that move from source to sink
        # list lambda : list of threshold values of lambda for two different consecutive solutions (right end of each lambda range) (we omit the last border, which is infinity)
        # list lambda type : whether inclusive or exclusive
        # len(list sol) = len(list lambda)+1 = len(list lambda type)+1

        if s_int[0] > s_int[1]:
            return [[0,-1]], [], []

        i_t = self.order[t]
        
        if i_t == 0:
            return self._solve_1(t, s_int)
        elif i_t == self.max_depth-1:
            return self._solve_N(t, s_int)
        else:
            return self._solve_middle(t, s_int)

    def _solve_1(self, t, s_int):

        N = self.max_depth
        A = self.arr
        
        # the first node on the right that is in the sink
        r = s_int[1]+1
        #r = 1 + next((i for i, x in enumerate(prev_sol[1:]) if x==0), N-1) # N if does not exist
        
        if r == 1:
            # the node 0 moves in any case (for any lambda)
            return [[0,0]], [], []
        else:
            v_n = A[t][0,r-1] # nothing moves
            sol_n = [0,-1] # if nothing moves, we always use [0,-1]
            v_e = A[-1][0,r-1] - A[t][0,r-1] # everything in the s-int moves
            sol_e = [s_int[0], s_int[1]]
            list_v_r = np.array([A[-1][0,k] - A[t][0,k] + A[t][k+1,r-1] for k in range(r-1)]) # some but not all nodes move
            argmin_v_r = np.argmin(list_v_r) 
            v_r = list_v_r[argmin_v_r] 
            sol_r = [0,argmin_v_r]
        
        if r == N:
            list_multi = [0,1]
            list_sol = [[sol_n, sol_e][np.argmin([v_n, v_e])], sol_r] # here the order in argmin (v_n comes before v_e) is important
            # because if v_n = v_e, we would prefer having nothing moves (n) rather than everything moves (e) (for maximal source set)
            list_const = [np.min((v_n, v_e)), v_r]
        else:
            list_multi = [0,1,2]
            list_sol = [sol_e, sol_n, sol_r]
            list_const = [v_e, v_n, v_r]
        
        list_med_sol, list_th_lambda = self._pre_solve_lambda(list_sol, list_multi, list_const) # threshold values of lambda, len(list_th_lambda) = len(list_sol)-1
        list_new_sol, list_new_th, list_new_th_type = self._simplify_solution(list_med_sol[::-1], list_th_lambda) # condense cases where some solutions are never chosen
        # we use the reverse of list sol as input because the list sol was first sorted according to the multiplier of lambda in each case in asecending order
        # however, the solution with larger multiplier is prefered for small lambda, so we have to reverse it
        
        return list_new_sol, list_new_th, list_new_th_type
    
    def _solve_N(self, t, s_int):

        N = self.max_depth
        A = self.arr
        
        # the first node on the left that is in the sink
        l = s_int[0]-1
        #l = i_t - 1 - next((i for i, x in enumerate(prev_sol[:i_t][::-1]) if x==0), i_t) # -1 if does not exist
        
        if l == N-2:
            # the node N-1 moves for any lambda
            return [[N-1, N-1]], [], []
        else:
            v_n = A[t][l+1,N-1]
            sol_n = [0,-1]
            v_e = A[-1][l+1,N-1] - A[t][l+1,N-1]
            sol_e = [s_int[0], s_int[1]]
            list_v_l = np.array([A[t][l+1,k-1] + A[-1][k,N-1] - A[t][k,N-1] for k in range(l+2, N)]) # some but not all nodes move
            argmin_v_l = l+2+np.argmin(list_v_l)
            v_l = list_v_l[argmin_v_l-l-2]
            sol_l = [argmin_v_l, N-1]
        
        if l == -1:
            list_multi = [0,1]
            list_sol = [[sol_n, sol_e][np.argmin([v_n, v_e])], sol_l]
            list_const = [np.min((v_n, v_e)), v_l]
        else:
            list_multi = [0,1,2]
            list_sol = [sol_e, sol_n, sol_l]
            list_const = [v_e, v_n, v_l]
        
        list_med_sol, list_th_lambda = self._pre_solve_lambda(list_sol, list_multi, list_const) # threshold values of lambda, len(list_th_lambda) = len(list_sol)-1
        list_new_sol, list_new_th, list_new_th_type = self._simplify_solution(list_med_sol[::-1], list_th_lambda) # condense cases where some solutions are never chosen
        # we use the reverse of list sol as input because the list sol was first sorted according to the multiplier of lambda in each case in asecending order
        # however, the solution with larger multiplier is prefered for small lambda, so we have to reverse it

        return list_new_sol, list_new_th, list_new_th_type
    
    def _solve_middle(self, t, s_int):

        N = self.max_depth
        A = self.arr
        i_t = self.order[t]

        l = s_int[0]-1
        r = s_int[1]+1

        # the constant part of the case where nothing moves to T
        v_n = A[t][l+1,r-1]
        sol_n = [0,-1]
        v_e = A[-1][l+1,r-1] - A[t][l+1,r-1]
        sol_e = [s_int[0], s_int[1]]

        if i_t-l > 1:
            list_v_r = np.array([A[t][l+1,j-1] + A[-1][j,r-1] - A[t][j,r-1] for j in range(l+2,i_t+1)])
            argmin_v_r = l+2+np.argmin(list_v_r)
            v_r = list_v_r[argmin_v_r-l-2]
            sol_r = [argmin_v_r, s_int[1]]
        if r-i_t > 1:
            list_v_l = np.array([A[t][j+1,r-1] + A[-1][l+1,j] - A[t][l+1,j] for j in range(i_t,r-1)])
            argmin_v_l = i_t+np.argmin(list_v_l)
            v_l = list_v_l[argmin_v_l-i_t]
            sol_l = [s_int[0], argmin_v_l]
        if i_t-l > 1 and r-i_t > 1:
            list_v_b = np.array([[A[-1][j,k] - 2*A[t][j,k] for j in range(l+2, i_t+1)] for k in range(i_t,r-1)])
            argmin_v_b = np.unravel_index(np.argmin(list_v_b, axis=None), list_v_b.shape)
            argmin_v_b = [argmin_v_b[0] + i_t, argmin_v_b[1] + l+2] # [k,j]
            v_b = list_v_b[argmin_v_b[0] - i_t, argmin_v_b[1] - l-2] + A[t][l+1,r-1]
            sol_b = [argmin_v_b[1], argmin_v_b[0]]
        
        # case 4A
        if l >= 0 and r <= N-1:
            if i_t - l > 1:
                if r - i_t > 1:
                    list_multi = [0,2,4]
                    list_sol = [sol_e, [sol_n, sol_l, sol_r][np.argmin([v_n, v_l, v_r])], sol_b]
                    list_const = [v_e, np.min((v_n, v_l, v_r)), v_b]
                else:
                    list_multi = [0,2]
                    list_sol = [sol_e, [sol_n, sol_r][np.argmin([v_n, v_r])]]
                    list_const = [v_e, np.min([v_n, v_r])]
            else:
                if r - i_t > 1:
                    list_multi = [0,2]
                    list_sol = [sol_e, [sol_n, sol_l][np.argmin([v_n, v_l])]]
                    list_const = [v_e, np.min([v_n, v_l])]
                else:
                    list_multi = [0,2]
                    list_sol = [sol_e, sol_n]
                    list_const = [v_e, v_n]
        
        # case 4B (no L)
        elif l == -1 and r <= N-1:
            if r-i_t > 1:
                list_multi = [0,1,2,3]
                list_sol = [sol_e, [sol_n, sol_r][np.argmin([v_n, v_r])], sol_l, sol_b]
                list_const = [v_e, np.min([v_n, v_r]), v_l, v_b]
            else:
                list_multi = [0,1]
                list_sol = [sol_e, [sol_n, sol_r][np.argmin([v_n, v_r])]]
                list_const = [v_e, np.min([v_n, v_r])]
        # case 4C (no R)
        elif r == N and l >= 0:
            if i_t - l > 1:
                list_multi = [0,1,2,3]
                list_sol = [sol_e, [sol_n, sol_l][np.argmin([v_n, v_l])], sol_r, sol_b]
                list_const = [v_e, np.min([v_n, v_l]), v_r, v_b]
            else:
                list_multi = [0,1]
                list_sol = [sol_e, [sol_n, sol_l][np.argmin([v_n, v_l])]]
                list_const = [v_e, np.min([v_n, v_l])]
        # case 4D (no L and no R)
        else:
            list_multi = [0,1,2]
            list_sol = [[sol_n, sol_e][np.argmin([v_n, v_e])], [sol_l, sol_r][np.argmin([v_l, v_r])], sol_b]
            list_const = [np.min([v_n, v_e]), np.min([v_l, v_r]), v_b]

        list_med_sol, list_th_lambda = self._pre_solve_lambda(list_sol, list_multi, list_const) # threshold values of lambda, len(list_th_lambda) = len(list_sol)-1
        list_new_sol, list_new_th, list_new_th_type = self._simplify_solution(list_med_sol[::-1], list_th_lambda) # condense cases where some solutions are never chosen
        # we use the reverse of list sol as input because the list sol was first sorted according to the multiplier of lambda in each case in asecending order
        # however, the solution with larger multiplier is prefered for small lambda, so we have to reverse it

        return list_new_sol, list_new_th, list_new_th_type

    def _pre_solve_lambda(self, list_sol, list_multi_unsorted, list_const_unsorted):
        # input: list_multi_sorted = list of multipliers of lambdas (such as [0,1,2] or [0,2,4])
        # list_const_unsorted = list of the constant part of the cost, the index must correspond to that in the multi list
        
        if isinstance(list_multi_unsorted, list):
            list_multi_unsorted = np.array(list_multi_unsorted)
            
        if isinstance(list_const_unsorted, list):
            list_const_unsorted = np.array(list_const_unsorted)
        
        sorted_indices = np.argsort(list_multi_unsorted)
        
        list_multi = list_multi_unsorted[sorted_indices]
        list_const = list_const_unsorted[sorted_indices]
        
        if len(list_multi) == 2:
            return list_sol, [(list_const[0]-list_const[1])/(list_multi[1]-list_multi[0])]
        elif len(list_multi) == 3:
            ratio01 = (list_const[0]-list_const[1])/(list_multi[1]-list_multi[0])
            ratio02 = (list_const[0]-list_const[2])/(list_multi[2]-list_multi[0])
            ratio12 = (list_const[1]-list_const[2])/(list_multi[2]-list_multi[1])
            if ratio02 > ratio01:
                return [list_sol[0], list_sol[-1]], [ratio02]
            else:
                return list_sol, [ratio12, ratio01]

        elif len(list_multi) == 4:
            list_sol_pre, lambda_pre = self._pre_solve_lambda(list_sol[0:3], list_multi[0:3], list_const[0:3])
            if len(lambda_pre) == 1:
                return self._pre_solve_lambda([list_sol[0], list_sol[2], list_sol[3]], [list_multi[0], list_multi[2], list_multi[3]], [list_const[0], list_const[2], list_const[3]])
            else:
                list_sol_pre2, lambda_pre2 = self._pre_solve_lambda(list_sol[1:], list_multi[1:], list_const[1:])
                if len(lambda_pre2) == 1:
                    return [list_sol[0], list_sol[1], list_sol[3]], [lambda_pre2[0], lambda_pre[-1]]
                elif len(lambda_pre2) == 2 and lambda_pre2[1] == lambda_pre[0]:
                    return list_sol, [lambda_pre2[0], lambda_pre[0], lambda_pre[1]]
    
    def _simplify_solution(self, list_sol, list_th_lambda):
        # here we also compare the cuts that result in the same obj value; we take the cut with maximal source set
        # we also have to do this ^ at each threshold value to decide whether we take the solution from the left or right
        # return 
        # 1) list of cut sols (range of nodes that move to T), example: [[3,9], [0,-1], [5,5]]
        # 2) list of lambda threshold (the right border of each lambda range), example: [lambda1, lambda2, lambda3, ..., lambda m] in this case we will have m+1 lambda ranges with the last range being [lambda m, inf] 
        # 3) list of type of lambda threshold (type of right border of each range): example ['in','ex','ex','in'] implies [0,lambda1], (lambda1,lambda2),[lambda2,lambda3),... : in=inclusive, ex=exclusive

        # we don't handle the case where lambda threshold is negative in this function
        # we do it in cap_solution

        length_intervals = [sol[1]-sol[0] for sol in list_sol] # for comparison of the size of source set of two solutions
        index_sol_max_source = np.argmin(length_intervals)

        if len(list_th_lambda) == 1:
            return list_sol, list_th_lambda, ['in' if length_intervals[0] < length_intervals[1] else 'ex']
        
        if len(list_th_lambda) == 2:
            if list_th_lambda[0] == list_th_lambda[1]:
                # here all three lines intersect at one single point
                # we have to check which solution the threshold lambda will take
                if length_intervals[0] < length_intervals[1]:
                    return [list_sol[0], list_sol[-1]], [list_th_lambda[0]], ['in' if length_intervals[0] < length_intervals[2] else 'ex']
                else:
                    if length_intervals[1] < length_intervals[2]:
                        return list_sol, list_th_lambda, ['ex','in']
                    else:
                        return [list_sol[0], list_sol[-1]], [list_th_lambda[0]], ['ex']
            else:
                list_th_lambda_type = ['in' if length_intervals[i] < length_intervals[i+1] else 'ex' for i in range(2)]
                return list_sol, list_th_lambda, list_th_lambda_type
        
        if len(list_th_lambda) == 3:
            if list_th_lambda[0] == list_th_lambda[1]:
                if list_th_lambda[1] == list_th_lambda[2]:
                    if index_sol_max_source in {0,3}:
                        return [list_sol[0], list_sol[-1]], [list_th_lambda[0]], ['in' if index_sol_max_source == 0 else 'ex']
                    elif index_sol_max_source in {1,2}:
                        return [list_sol[0], list_sol[index_sol_max_source], list_sol[-1]], [list_th_lambda[0], list_th_lambda[0]], ['ex','in']
                else:
                    if length_intervals[1] < length_intervals[0] and length_intervals[1] < length_intervals[2]:
                        return list_sol, list_th_lambda, ['ex','in', 'in' if length_intervals[2] < length_intervals[3] else 'ex']
                    else:
                        return [list_sol[0], list_sol[2], list_sol[3]], [list_th_lambda[0],list_th_lambda[2]], \
                            ['in' if length_intervals[0] < length_intervals[2] else 'ex', 'in' if length_intervals[2] < length_intervals[3] else 'ex']

            else:
                if list_th_lambda[1] == list_th_lambda[2]:
                    if length_intervals[2] < length_intervals[1] and length_intervals[2] < length_intervals[3]:
                        return list_sol, list_th_lambda, ['in' if length_intervals[0] < length_intervals[1] else 'ex','ex', 'in']
                    else:
                        return [list_sol[0], list_sol[1], list_sol[3]], [list_th_lambda[0],list_th_lambda[1]], \
                            ['in' if length_intervals[0] < length_intervals[1] else 'ex', 'in' if length_intervals[1] < length_intervals[3] else 'ex']
                else:
                    return list_sol, list_th_lambda, ['in' if length_intervals[i] < length_intervals[i+1] else 'ex' for i in range(3)]

    def _intersect_solution(self, list_sol, list_lambda, list_lambda_type, lambda_range, lambda_range_type):

        if len(list_sol) == 1:
            return list_sol, [lambda_range], [lambda_range_type]

        # find which interval in the list_lambda the left end of the lambda range is in; also take into account the inclusive/exclusive type of the border
        left_index = bisect.bisect_left(list_lambda, lambda_range[0])
        if left_index < len(list_lambda) and list_lambda[left_index] == lambda_range[0]:
            if list_lambda_type[left_index] == 'ex':
                left_index += 1
                if left_index < len(list_lambda) and lambda_range_type[0] == 'ex' and list_lambda[left_index] == lambda_range[0]:
                    left_index += 1

        # find which interval in the list_lambda the right end of the lambda range is in
        right_index = bisect.bisect_left(list_lambda, lambda_range[1])
        if right_index < len(list_lambda) and list_lambda[right_index] == lambda_range[1]:
            if list_lambda_type[right_index] == 'ex':
                if lambda_range_type[1] == 'in':
                    right_index += 1
        
        list_capped_sol = list_sol[left_index:right_index+1]
        if left_index == right_index:
            list_capped_lambda = [[lambda_range[0], lambda_range[1]]]
            list_capped_lambda_type = [[lambda_range_type[0], lambda_range_type[1]]]
        else:
            list_capped_lambda = [[lambda_range[0], list_lambda[left_index]]] \
                + [[list_lambda[i-1],list_lambda[i]] for i in range(left_index+1,right_index)] +[[list_lambda[right_index-1], lambda_range[1]]]
            list_capped_lambda_type = [[lambda_range_type[0], list_lambda_type[left_index]]] \
                + [['in' if list_lambda_type[i-1]=='ex' else 'ex',list_lambda_type[i]] for i in range(left_index+1,right_index)] \
                    + [['in' if list_lambda_type[right_index-1]=='ex' else 'ex',lambda_range_type[1]]]
    
        return list_capped_sol, list_capped_lambda, list_capped_lambda_type

    def _find_s_int(self, depth, sol):
        current_node = self.order[depth]
        if sol[current_node] == 0:
            return (0,-1)
        r = current_node + next((i for i, x in enumerate(sol[(current_node+1):]) if x==0), len(sol)-1-current_node)
        l = current_node - next((i for i, x in enumerate(sol[:current_node][::-1]) if x==0), current_node)
        return (l,r)        
        
    
    def full_split(self):

        weights = self.weights
        breakpoints = self.breakpoints
        order = self.order
        N = len(weights)

        node_current_level = [self.root]
        node_next_level = []
        current_depth = 0

        while node_current_level:

            if current_depth == self.max_depth-1:
                # get the list of solutions for different lambda ranges
                all_solutions = np.zeros((len(node_current_level),N))
                all_lambda_ranges = np.zeros((len(node_current_level),2))
                all_lambda_ranges_type = np.empty((len(node_current_level),2), dtype='<U2')
                for i, node in enumerate(node_current_level):
                    for j in range(N):
                        all_solutions[i, j] = breakpoints[order[node.step_move_to_sink[j]]]
                    all_lambda_ranges[i, :] = node.lambda_range
                    all_lambda_ranges_type[i, :] = node.lambda_range_type
                self.all_solutions, self.all_lambda_ranges, self.all_lambda_ranges_type = all_solutions, all_lambda_ranges, all_lambda_ranges_type
                break

            node_current_level_dict = defaultdict(list)

            for tree_node in node_current_level:
                s_int = self._find_s_int(current_depth, tree_node.sol)
                tree_node.source_node_interval = s_int

                if tree_node.num_sink == N:
                    new_sol = [j for j in tree_node.sol]
                    new_lambda_range = [j for j in tree_node.lambda_range]
                    new_lambda_range_type = [j for j in tree_node.lambda_range_type]
                    new_step_move_to_sink = [j for j in tree_node.step_move_to_sink]
                    child = treeNode(depth=tree_node.depth+1, sol=new_sol, lambda_range=new_lambda_range, lambda_range_type=new_lambda_range_type, \
                                     step_move_to_sink=new_step_move_to_sink, num_sink=N)
                    tree_node.children = [child]
                    node_next_level.append(child)
                else:
                    node_current_level_dict[s_int].append(tree_node)

            for s_interval in node_current_level_dict:

                # solve for cut solutions given s-interval and depth
                list_sol, list_lambda, list_lambda_type = self._solve(current_depth, s_interval)

                for tree_node in node_current_level_dict[s_interval]:
                    # intersect list_lambda with each lambda_range 
                    list_final_sol, list_final_lambda, list_final_lambda_type = self._intersect_solution(list_sol, list_lambda, list_lambda_type, tree_node.lambda_range, tree_node.lambda_range_type)
                    # create children list and update other parameters
                    for i in range(len(list_final_sol)):
                        new_sol = [j for j in tree_node.sol]
                        new_step_move_to_sink = [j for j in tree_node.step_move_to_sink]
                        for j in range(list_final_sol[i][0], list_final_sol[i][1]+1):
                            new_sol[j] = 0
                            new_step_move_to_sink[j] = current_depth

                        child = treeNode(depth=current_depth+1, sol=new_sol, lambda_range=list_final_lambda[i], lambda_range_type=list_final_lambda_type[i])
                        child.step_move_to_sink = new_step_move_to_sink
                        child.num_sink = tree_node.num_sink + max(0, list_final_sol[i][1]-list_final_sol[i][0]+1)
                        tree_node.children.append(child)
                        node_next_level.append(child)

            node_current_level = node_next_level
            node_next_level = []
            current_depth += 1

        return self
    
    def draw(self):
        queue = [self.root]
        while len(queue) > 0:
            node = queue.pop()
            print("        "*(node.depth),"------", " ", np.round(np.array(node.lambda_range), 2), " : ", node.sol, " : ", node.depth)
            queue += node.children
        return None
    
    def getsize(self):
        queue = [self.root]
        size = [0 for i in range(self.max_depth)]
        while len(queue) > 0:
            node = queue.pop()
            size[node.depth] +=1 
            queue += node.children
        return size
    
    def get_s_int_all_depths(self):
        # get the s-intervals of nodes in each depth
        queue = [self.root]
        list_s_int = [[] for i in range(self.max_depth)]
        while len(queue) > 0:
            node = queue.pop()
            list_s_int[node.depth].append(node.source_node_interval) 
            queue += node.children
        return list_s_int
    
    def solve_lasso(self, lambda_value):
        # get the optimal solution to the original fused lasso problem

        if self.all_lambda_ranges is None:
            print("HAVE NOT BUILT THE TREE! -- run fit() and fill_split() first")
            return
        
        index = bisect.bisect_left(self.all_lambda_ranges, lambda_value)
        
        return
