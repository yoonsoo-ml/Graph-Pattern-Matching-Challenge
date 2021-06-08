import math
import random
from time import time
from collections import defaultdict, deque


class Backtracker:

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        self.do_print = do_print
        self.timer = timer
        self.data, self.data2label = self.read_graph(data_path)
        self.query, self.query2label = self.read_graph(query_path)
        self.cs = self.read_cs(cs_path)
        self.query_dag = self.build_dag(self.query, self.cs)
        self.max_matches = max_matches
        # limit data graph to only candidates
        candidate_nodes = set()
        for k, v in self.cs.items():
            candidate_nodes.update(v)
        self.candidate_data = {k: self.data[k] & candidate_nodes for k in candidate_nodes}

    @staticmethod
    def read_graph(path):
        """
        reads graph from path then
        return adjacency list type graph and dictionary that maps vertex to label
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            args = line.strip().split()
            if args[0] == 't':
                graph_id = int(args[1])
                graph = defaultdict(set)
                graph2label = dict()
            elif args[0] == 'v':
                graph2label[int(args[1])] = int(args[2])
            elif args[0] == 'e':
                graph[int(args[1])].add(int(args[2]))
                graph[int(args[2])].add(int(args[1]))
            else:
                raise KeyError
        return graph, graph2label

    @staticmethod
    def read_cs(path):
        """
        reads candidate space from path then
        return adjacency list type graph
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            args = line.strip().split()
            if args[0] == 't':
                graph_id = int(args[1])
                data = defaultdict(set)
            elif args[0] == 'c':
                data[int(args[1])].update([int(x) for x in args[3:]])
            else:
                raise KeyError
        return data

    def select_root(self, graph, cs):
        """
        from graph, select vertex that has maximum degree
        """
        min_value = float("inf")
        root = 0
        for k, v in graph.items():
            value = len(cs[k]) / len(v)
            if value < min_value:
                root = k
                min_value = value
        return root

    def build_dag(self, graph, cs):
        """
        from graph, build dag with a root that has maximum degree
        """
        root = self.select_root(graph, cs)
        query_dag = {k: {'parents': set(), 'children': set()} for k in graph.keys()}
        visited = []
        queue = deque([root])
        while queue:
            n = queue.popleft()
            if n not in visited:
                visited.append(n)
                cur_children = graph[n] - set(visited)
                queue += cur_children
                query_dag[n]['children'].update(cur_children)
                for child in cur_children:
                    query_dag[child]['parents'].add(n)
        return query_dag

    def convert_dict_to_str(self, d):
        """
        convert adjacency list graph to given output format
        """
        string = 'a'
        for i in sorted(self.query.keys()):
            string += f' {d[i]}'
        return string

    def parents_all_in_partial_embedding(self, vertex, partial_embedding):
        """
        check if parents of a vertex are all inside partial embedding
        """
        for parent in self.query_dag[vertex]['parents']:
            if parent not in partial_embedding.keys():
                return False
        return True

    def candidate_connected_to_all_parents(self, candidate, vertex, partial_embedding):
        """
        check if a candidate is conncetd to all parents of a vertex
        """
        for parent in self.query_dag[vertex]['parents']:
            if candidate not in self.candidate_data[partial_embedding[parent]]:
                return False
        return True

    def return_condition(self, unvisited_query_vertices, partial_embedding, recursion_depth):
        # if it runs for more than self.timer seconds, just return
        if self.timer:
            if time() - self.start_time > self.timer:
                if not self.printed_terminate:
                    print(f'Exceeded {self.timer} Seconds, Terminating...')
                    self.printed_terminate = True
                return True
        # if we've found max_matches, return
        if self.counter >= self.max_matches:
            return True

        # if all query vertices are visited, we've found a subgraph
        if len(unvisited_query_vertices) == 0:
            self.counter += 1
            if self.do_print:
                print(self.convert_dict_to_str(partial_embedding))
            else:
                self.result.append(partial_embedding)
            return True

        return False

    def find_extendable_vertices(self, unvisited_query_vertices, partial_embedding):
        extendable_vertices = set()
        for vertex in unvisited_query_vertices:
            if self.parents_all_in_partial_embedding(vertex, partial_embedding):
                extendable_vertices.add(vertex)
        return extendable_vertices

    def find_extendable_candidates(self, vertex, partial_embedding):
        extendable_candidates = set()
        for candidate in self.cs[vertex]:
            # check if candidate is not already in partial embedding, and is connected to all parents of vertex
            if candidate not in partial_embedding.values():
                if self.candidate_connected_to_all_parents(candidate, vertex, partial_embedding):
                    extendable_candidates.add(candidate)
        return extendable_candidates

    def check_prune(self, unvisited_query_vertices):
        """
        return n recursive calls if no find after m recursion, increase n by 1 after every return
        n_paths = (upper bound of path size)^a, where 0<a<1 is given as a hyperparameter
        m = n_paths.clip(10000, 1000000)
        """
        self.n_recursion += 1
        try:
            n_paths = math.prod([len(self.cs[x]) ** 0.16 for x in unvisited_query_vertices])
        except OverflowError:
            n_paths = 1000000
        if (self.n_recursion > max(min(1000000, n_paths), 10000)) & (self.counter == 0):
            self.force_return = self.n_returns
            self.n_returns += 1
            self.n_recursion = 0
            return True
        return False

    def backtrack(self, partial_embedding, unvisited_query_vertices):
        """
        should be implemented at child classes
        """
        pass

    def run(self):
        """
        run backtracking
        """
        self.result = []
        self.start_time = time()
        self.counter = 0
        self.printed_terminate = False
        self.n_recursion = 0
        self.force_return = 0
        self.n_returns = 0
        if self.do_print:
            print(f't {len(self.query)}')
        # sort unvisited_query_vertices by candidate size
        self.backtrack({}, sorted(list(self.query.keys()), key=lambda k: len(self.cs[k])))

    def check(self):
        """
        check if found embeddings are correct
        """
        assert not self.do_print
        print(f'{len(self.result)} embeddings found / Correctness: ', end='')
        for embedding in random.sample(self.result,
                                       min(len(self.result), 1000)):  # only check 1000 embeddings to save time
            if len(set(embedding.values())) < len(embedding):
                return False
            for k, v in embedding.items():
                if self.query2label[k] != self.data2label[v]:
                    return False
                if not set([embedding[x] for x in self.query[k]]).issubset(self.candidate_data[v]):
                    return False
        return True


class BacktrackerV1(Backtracker):
    """
    only consider candidate set size when selecting next vertex
    """

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        super().__init__(data_path, query_path, cs_path, do_print, timer, max_matches)

    def select_next_vertex(self, extendable_vertices):
        min_candidate_size = float("inf")
        next_vertex = None
        for vertex in extendable_vertices:
            if len(self.cs[vertex]) < min_candidate_size:
                next_vertex = vertex
                min_candidate_size = len(self.cs[vertex])
        return next_vertex

    def backtrack(self, partial_embedding, unvisited_query_vertices, recursion_depth=0):

        if self.return_condition(unvisited_query_vertices, partial_embedding, recursion_depth):
            return

        # find extendable vertices
        extendable_vertices = self.find_extendable_vertices(unvisited_query_vertices, partial_embedding)

        # select vertex to process
        vertex = self.select_next_vertex(extendable_vertices)
        unvisited_query_vertices.remove(vertex)

        # check if candidate is not already in partial embedding, and is connected to all parents of vertex
        for candidate in self.cs[vertex]:
            if candidate not in partial_embedding.values():
                if self.candidate_connected_to_all_parents(candidate, vertex, partial_embedding):
                    # add candidate to partial embedding then recurse
                    partial_embedding[vertex] = candidate
                    self.backtrack(partial_embedding.copy(), unvisited_query_vertices.copy(), recursion_depth + 1)

        return


class BacktrackerV2(Backtracker):
    """
    replicate lecture slide (minimum extendable candidate size order)
    """

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        super().__init__(data_path, query_path, cs_path, do_print, timer, max_matches)

    def backtrack(self, partial_embedding, unvisited_query_vertices, recursion_depth=0):
        if self.return_condition(unvisited_query_vertices, partial_embedding, recursion_depth):
            return

        # find extendable vertices
        extendable_vertices = self.find_extendable_vertices(unvisited_query_vertices, partial_embedding)

        # select next vertex and extendable candidates of that vertex
        min_extendable_candidates_size = float("inf")
        vertex = None
        extendable_candidates = set()
        for tmp_vertex in extendable_vertices:
            tmp_extendable_candidates = self.find_extendable_candidates(tmp_vertex, partial_embedding)
            if len(tmp_extendable_candidates) < min_extendable_candidates_size:
                min_extendable_candidates_size = len(tmp_extendable_candidates)
                vertex = tmp_vertex
                extendable_candidates = tmp_extendable_candidates

        unvisited_query_vertices.remove(vertex)

        # recurse
        for candidate in extendable_candidates:
            partial_embedding[vertex] = candidate
            self.backtrack(partial_embedding.copy(), unvisited_query_vertices.copy(), recursion_depth + 1)

        return


class BacktrackerV3(Backtracker):
    """
    no matching order
    """

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        super().__init__(data_path, query_path, cs_path, do_print, timer, max_matches)

    def find_extendable_vertex_and_remove(self, unvisited_query_vertices, partial_embedding):
        for i, vertex in enumerate(unvisited_query_vertices):
            if self.parents_all_in_partial_embedding(vertex, partial_embedding):
                unvisited_query_vertices.pop(i)
                return vertex

    def backtrack(self, partial_embedding, unvisited_query_vertices, recursion_depth=0):

        if self.return_condition(unvisited_query_vertices, partial_embedding, recursion_depth):
            return

        vertex = self.find_extendable_vertex_and_remove(unvisited_query_vertices, partial_embedding)

        # check if candidate is not already in partial embedding, and is connected to all parents of vertex
        for candidate in self.cs[vertex]:
            if candidate not in partial_embedding.values():
                if self.candidate_connected_to_all_parents(candidate, vertex, partial_embedding):
                    # add candidate to partial embedding then recurse
                    partial_embedding[vertex] = candidate
                    self.backtrack(partial_embedding.copy(), unvisited_query_vertices.copy(), recursion_depth + 1)
        return


class BacktrackerV4(Backtracker):
    """
    maximum extendable candidate size
    """

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        super().__init__(data_path, query_path, cs_path, do_print, timer, max_matches)

    def backtrack(self, partial_embedding, unvisited_query_vertices, recursion_depth=0):
        if self.return_condition(unvisited_query_vertices, partial_embedding, recursion_depth):
            return

        # find extendable vertices
        extendable_vertices = self.find_extendable_vertices(unvisited_query_vertices, partial_embedding)

        # select next vertex and extendable candidates of that vertex
        max_extendable_candidates_size = -float("inf")
        vertex = None
        extendable_candidates = set()
        for tmp_vertex in extendable_vertices:
            # find extendable candidates
            tmp_extendable_candidates = set()
            for candidate in self.cs[tmp_vertex]:
                # check if candidate is not already in partial embedding, and is connected to all parents of vertex
                if candidate not in partial_embedding.values():
                    if self.candidate_connected_to_all_parents(candidate, tmp_vertex, partial_embedding):
                        tmp_extendable_candidates.add(candidate)
            if len(tmp_extendable_candidates) > max_extendable_candidates_size:
                max_extendable_candidates_size = len(tmp_extendable_candidates)
                vertex = tmp_vertex
                extendable_candidates = tmp_extendable_candidates

        unvisited_query_vertices.remove(vertex)

        # recurse
        for candidate in extendable_candidates:
            partial_embedding[vertex] = candidate
            self.backtrack(partial_embedding.copy(), unvisited_query_vertices.copy(), recursion_depth + 1)

        return


class BacktrackerV5(Backtracker):
    """
    sum candidates of extendable candidates' children
    """

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        super().__init__(data_path, query_path, cs_path, do_print, timer, max_matches)

    def backtrack(self, partial_embedding, unvisited_query_vertices, recursion_depth=0):

        if self.return_condition(unvisited_query_vertices, partial_embedding, recursion_depth):
            return

        extendable_vertices = self.find_extendable_vertices(unvisited_query_vertices, partial_embedding)

        # adaptive matching order
        min_value = float("inf")
        vertex = None
        extendable_candidates = set()
        for tmp_vertex in extendable_vertices:
            tmp_extendable_candidates = self.find_extendable_candidates(tmp_vertex, partial_embedding)
            tmp_value = sum(
                [len(self.candidate_data[x] - set(partial_embedding.values())) for x in tmp_extendable_candidates])
            if tmp_value < min_value:
                min_value = tmp_value
                vertex = tmp_vertex
                extendable_candidates = tmp_extendable_candidates

        unvisited_query_vertices.remove(vertex)

        # recurse
        for candidate in extendable_candidates:
            partial_embedding[vertex] = candidate
            self.backtrack(partial_embedding.copy(), unvisited_query_vertices.copy(), recursion_depth + 1)

        return


class BacktrackerV6(Backtracker):
    """
    order extendable candidates
    """

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        super().__init__(data_path, query_path, cs_path, do_print, timer, max_matches)

    def backtrack(self, partial_embedding, unvisited_query_vertices, recursion_depth=0):

        if self.return_condition(unvisited_query_vertices, partial_embedding, recursion_depth):
            return

        extendable_vertices = self.find_extendable_vertices(unvisited_query_vertices, partial_embedding)

        # adaptive matching order
        min_value = float("inf")
        vertex = None
        extendable_candidates = set()
        for tmp_vertex in extendable_vertices:
            tmp_extendable_candidates = self.find_extendable_candidates(tmp_vertex, partial_embedding)
            if len(tmp_extendable_candidates) > 0:
                tmp_value = len(tmp_extendable_candidates)
                if tmp_value < min_value:
                    min_value = tmp_value
                    vertex = tmp_vertex
                    extendable_candidates = tmp_extendable_candidates

        if len(extendable_candidates) == 0:
            return

        unvisited_query_vertices.remove(vertex)

        # order extendable candidates
        visited_candidates = set(partial_embedding.values())
        extendable_candidates = sorted(list(extendable_candidates),
                                       key=lambda x: len(self.candidate_data[x] - visited_candidates))

        # recurse
        for candidate in extendable_candidates:
            partial_embedding[vertex] = candidate
            self.backtrack(partial_embedding.copy(), unvisited_query_vertices.copy(), recursion_depth + 1)

        return


class BacktrackerV7(Backtracker):
    """
    precompute matching order according to candidate size order
    """

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        super().__init__(data_path, query_path, cs_path, do_print, timer, max_matches)

    def find_extendable_vertex_and_remove(self, unvisited_query_vertices, partial_embedding):
        for i, vertex in enumerate(unvisited_query_vertices):
            if self.parents_all_in_partial_embedding(vertex, partial_embedding):
                unvisited_query_vertices.pop(i)
                return vertex

    def backtrack(self, partial_embedding, unvisited_query_vertices, recursion_depth=0):

        if self.return_condition(unvisited_query_vertices, partial_embedding, recursion_depth):
            return

        # select first extendable vertex
        vertex = self.find_extendable_vertex_and_remove(unvisited_query_vertices, partial_embedding)

        # check if candidate is not already in partial embedding, and is connected to all parents of vertex
        for candidate in self.cs[vertex]:
            if candidate not in partial_embedding.values():
                if self.candidate_connected_to_all_parents(candidate, vertex, partial_embedding):
                    # add candidate to partial embedding then recurse
                    partial_embedding[vertex] = candidate
                    self.backtrack(partial_embedding.copy(), unvisited_query_vertices.copy(), recursion_depth + 1)

        return


class BacktrackerV8(Backtracker):
    """
    consider size of candidate set not already in partial embedding when selecting next vertex
    """

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        super().__init__(data_path, query_path, cs_path, do_print, timer, max_matches)

    def select_next_vertex(self, extendable_vertices, visited_data_nodes):
        min_candidate_size = float("inf")
        next_vertex = None
        for vertex in extendable_vertices:
            if len(self.cs[vertex] - visited_data_nodes) < min_candidate_size:
                next_vertex = vertex
                min_candidate_size = len(self.cs[vertex])
        return next_vertex

    def backtrack(self, partial_embedding, unvisited_query_vertices, recursion_depth=0):

        if self.return_condition(unvisited_query_vertices, partial_embedding, recursion_depth):
            return

        # find extendable vertices
        extendable_vertices = self.find_extendable_vertices(unvisited_query_vertices, partial_embedding)

        # select vertex to process
        vertex = self.select_next_vertex(extendable_vertices, set(partial_embedding.values()))
        unvisited_query_vertices.remove(vertex)

        # check if candidate is not already in partial embedding, and is connected to all parents of vertex
        for candidate in self.cs[vertex]:
            if candidate not in partial_embedding.values():
                if self.candidate_connected_to_all_parents(candidate, vertex, partial_embedding):
                    # add candidate to partial embedding then recurse
                    partial_embedding[vertex] = candidate
                    self.backtrack(partial_embedding.copy(), unvisited_query_vertices.copy(), recursion_depth + 1)

        return


class BacktrackerV9(Backtracker):
    """
    V7 + prune
    """

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        super().__init__(data_path, query_path, cs_path, do_print, timer, max_matches)

    def find_extendable_vertex_and_remove(self, unvisited_query_vertices, partial_embedding):
        for i, vertex in enumerate(unvisited_query_vertices):
            if self.parents_all_in_partial_embedding(vertex, partial_embedding):
                unvisited_query_vertices.pop(i)
                return vertex

    def backtrack(self, partial_embedding, unvisited_query_vertices, recursion_depth=0):

        if self.return_condition(unvisited_query_vertices, partial_embedding, recursion_depth):
            return

        if self.check_prune(unvisited_query_vertices):
            return

        # select first extendable vertex
        vertex = self.find_extendable_vertex_and_remove(unvisited_query_vertices, partial_embedding)

        # check if candidate is not already in partial embedding, and is connected to all parents of vertex
        for candidate in self.cs[vertex]:
            if candidate not in partial_embedding.values():
                if self.candidate_connected_to_all_parents(candidate, vertex, partial_embedding):
                    # add candidate to partial embedding then recurse
                    partial_embedding[vertex] = candidate
                    self.backtrack(partial_embedding.copy(), unvisited_query_vertices.copy(), recursion_depth + 1)
                    if self.force_return:
                        self.force_return -= 1
                        return

        return


class BacktrackerV10(Backtracker):
    """
    V2 + prune
    """

    def __init__(self, data_path, query_path, cs_path, do_print=True, timer=False, max_matches=100000):
        super().__init__(data_path, query_path, cs_path, do_print, timer, max_matches)

    def backtrack(self, partial_embedding, unvisited_query_vertices, recursion_depth=0):

        if self.return_condition(unvisited_query_vertices, partial_embedding, recursion_depth):
            return

        if self.check_prune(unvisited_query_vertices):
            return

        # find extendable vertices
        extendable_vertices = self.find_extendable_vertices(unvisited_query_vertices, partial_embedding)

        # select next vertex and extendable candidates of that vertex
        min_extendable_candidates_size = float("inf")
        vertex = None
        extendable_candidates = set()
        for tmp_vertex in extendable_vertices:
            tmp_extendable_candidates = self.find_extendable_candidates(tmp_vertex, partial_embedding)
            if len(tmp_extendable_candidates) < min_extendable_candidates_size:
                min_extendable_candidates_size = len(tmp_extendable_candidates)
                vertex = tmp_vertex
                extendable_candidates = tmp_extendable_candidates

        unvisited_query_vertices.remove(vertex)

        # recurse
        for candidate in extendable_candidates:
            partial_embedding[vertex] = candidate
            self.backtrack(partial_embedding.copy(), unvisited_query_vertices.copy(), recursion_depth + 1)
            if self.force_return:
                self.force_return -= 1
                return
        return
