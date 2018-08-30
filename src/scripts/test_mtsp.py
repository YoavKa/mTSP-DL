from argparse import ArgumentParser

from ..mtsp import MTSPModel


def main(*args):
    parser = ArgumentParser()
    parser.add_argument('weights')
    parser.add_argument('dataset')
    parser.add_argument('beam_sizes', type=int, nargs='+')
    parser.add_argument('-o', '--other-args')
    args = parser.parse_args(args)

    if args.other_args is not None:
        base_args = list(filter(lambda item: item, args.other_args.split()))
    else:
        base_args = []
    model = MTSPModel.run(base_args + [
        '--load-weights', args.weights,
        '--val-paths', args.dataset,
        '--beam-sizes', *list(map(str, args.beam_sizes)),
    ], epochs=0)

    return model.val()
