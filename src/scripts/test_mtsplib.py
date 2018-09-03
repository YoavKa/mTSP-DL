from argparse import ArgumentParser
from collections import OrderedDict
import os

import pandas as pd
from tqdm import tqdm

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from ..logging import pretty_print, time_str
from ..mtsp import MTSPModel, MTSPLibDataset


META_HEURISTIC = OrderedDict({
    'greedy_descent': routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    'guided_local_search': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    'simulated_annealing': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
    'tabu_search': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    'objective_tabu_search': routing_enums_pb2.LocalSearchMetaheuristic.OBJECTIVE_TABU_SEARCH
})


FIRST_STRATEGY = OrderedDict({
    'path_cheapest_arc': routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    'path_most_constrained_arc': routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
    'christofides': routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
    'parallel_cheapest_insertion': routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    'local_cheapest_insertion': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
    'global_cheapest_arc': routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
    'local_cheapest_arc': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
    'first_unbound_min_value': routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
})


class CitiesDistancesWrapper(object):
    SCALE = 10**10

    def __init__(self, cities):
        from_cities = cities.unsqueeze(1).expand(cities.size(0), cities.size(0), 2)
        to_cities = cities.unsqueeze(0).expand_as(from_cities)
        self.dists = (from_cities - to_cities).norm(p=2, dim=2)

    def distance(self, from_city, to_city):
        return self.dists[from_city, to_city]

    def approx_distance(self, from_city, to_city):
        return int(self.distance(from_city, to_city) * self.SCALE)


def connectivity_matrix_to_routes(con_matrix):
    # con_matrix:  cities x cities
    routes = []
    for route_number, cur in enumerate(con_matrix[0].nonzero().squeeze(1)):
        route = []
        while cur != 0:
            route.append(cur)
            cur = con_matrix[cur].nonzero()[0, 0]
        routes.append(route)
    return routes


def solve_mtsp(cities, groups, meta_heuristic, first_strategy, solution_limit):
    routing = pywrapcp.RoutingModel(cities.size(0), groups, 0)

    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    if solution_limit > 0:
        search_parameters.solution_limit = solution_limit
    search_parameters.local_search_metaheuristic = meta_heuristic

    distances_wrapper = CitiesDistancesWrapper(cities)
    distance_callback = distances_wrapper.approx_distance

    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)

    # add a dummy dimension where each transit adds 1 to the cumul value
    routing.AddConstantDimension(1, cities.size(0), True, 'dummy')
    dummy_dimension = routing.GetDimensionOrDie('dummy')
    # add a lower bound to the cumul value of the dummy dimension, to make sure each group has a non-empty route
    for route_number in range(groups):
        end_node = routing.End(route_number)
        dummy_dimension.CumulVar(end_node).SetMin(2)

    if not isinstance(first_strategy, int):  # custom first strategy:
        # The following line is only necessary if you have added search parameters different from the
        # default parameters. If you omit this line, the search uses default search parameters.
        routing.CloseModelWithParameters(search_parameters)
        initial_routes = connectivity_matrix_to_routes(first_strategy)
        initial_assignment = routing.ReadAssignmentFromRoutes(initial_routes, True)
        if solution_limit == 1:
            assignment = initial_assignment
        else:
            assignment = routing.SolveFromAssignmentWithParameters(initial_assignment, search_parameters)
    else:
        assert first_strategy >= 0, 'a custom first strategy must be inserted using a connectivity matrix!'
        search_parameters.first_solution_strategy = first_strategy
        assignment = routing.SolveWithParameters(search_parameters)

    if assignment is None:
        pretty_print('ERROR: An assignment could not be created!')
        return None

    # calculate actual distance of the assignment
    length = 0.0
    actual_distance = CitiesDistancesWrapper(cities).distance
    for route_number in range(groups):
        path_length = 0.0
        nodes_count = 2
        node = routing.Start(route_number)
        next_node = assignment.Value(routing.NextVar(node))
        while not routing.IsEnd(next_node):
            nodes_count += 1
            path_length += actual_distance(routing.IndexToNode(node), routing.IndexToNode(next_node))
            node = next_node
            next_node = assignment.Value(routing.NextVar(node))
        if nodes_count < 3:
            pretty_print('ERROR: Every route should include at least one city which is not the depot!')
            return None
        path_length += actual_distance(routing.IndexToNode(node), routing.IndexToNode(next_node))
        length += path_length
    return length


def main(*args):
    parser = ArgumentParser()
    parser.add_argument('weights')
    parser.add_argument('tsplib_dir')
    parser.add_argument('output_file')
    parser.add_argument('meta', choices=META_HEURISTIC.keys())
    parser.add_argument('top', type=int)
    parser.add_argument('beam', type=int)
    parser.add_argument('-o', '--other-args')
    args = parser.parse_args(args)

    if args.other_args is not None:
        base_args = list(filter(lambda item: item, args.other_args.split()))
    else:
        base_args = []
    model = MTSPModel.run(base_args + [
        '--load-weights', args.weights
    ], epochs=0)
    model.net.eval()
    model.net.memory_efficient = True

    results = pd.DataFrame()

    name = os.path.basename(args.weights)
    for cities, instance_name, groups in tqdm(list(MTSPLibDataset(args.tsplib_dir)),
                                                desc=time_str() + ': Solving mTSPLib',
                                                leave=False,
                                                unit='instance'):
        first = model.solve(cities, groups, beam_size=args.top)
        length = solve_mtsp(cities, groups, META_HEURISTIC[args.meta], first, args.beam - args.top + 1)
        if length is not None:
            combination_name = name + '[' + str(args.top) + '] | ' + args.meta + '[' + str(args.beam) + ']'
            results.at[combination_name, instance_name] = length

    results.to_csv(args.output_file)

    if not results.empty:
        summary = results.min().to_frame('length').join(results.idxmin().to_frame('configuration'))
        print('Summary:')
        print(summary.to_string())
    else:
        print('No results were obtained...')
