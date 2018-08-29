import abc
from collections import defaultdict, OrderedDict
import configparser
import json
import os
import time
import argparse
import sys
from glob import glob

import torch
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from tensorboardX import SummaryWriter

from ..loaders import RepeatIterator
from ..logging import pretty_print, time_str, log_tensorboard
from ..utils import USE_GPU, set_seed, is_number, torch_load, torch_save


class CustomModel(abc.ABC):
    def __init__(self, **kwargs):
        pretty_print('Model', self.type_name, 'setup:')
        pretty_print('\tUsing GPU:', USE_GPU)
        if USE_GPU:
            pretty_print(f'\tCUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')

        for key, value in self.default_kwargs().items():
            if key not in kwargs:
                kwargs[key] = value

        self.start_args = kwargs

        if kwargs['seed'] > 0:
            set_seed(kwargs['seed'])

        self.net = self.init_net()
        if USE_GPU and self.net is not None:
            self.net = self.net.cuda()

        self.criterion = self.init_criterion()
        if USE_GPU and self.criterion is not None:
            self.criterion = self.criterion.cuda()

        self.optimizer = self.init_optimizer()
        # noinspection PyTypeChecker
        self.scheduler = StepLR(self.optimizer, kwargs['lr_step'], kwargs['lr_gamma'])

        if kwargs['load_weights'] is not None:
            self.load(kwargs['load_weights'])
        pretty_print('\tNet setup completed')

        # noinspection PyNoneFunctionAssignment
        self.train_dataset = self.get_dataset(kwargs['train_paths'], is_train=True)
        if self.train_dataset is not None:
            pretty_print(f'\tTrain dataset loaded: {len(self.train_dataset)} samples found')
            self.train_loader = RepeatIterator(self.get_loader(self.train_dataset, is_train=True))
        else:
            self.train_loader = None

        self.val_datasets = OrderedDict()
        self.val_loaders = OrderedDict()
        for path in kwargs['val_paths'].split():
            # noinspection PyNoneFunctionAssignment
            dataset = self.get_dataset(path, is_train=False)
            if dataset is not None:
                pretty_print(f'\tVal dataset loaded: {len(dataset)} samples found in {path}')
                self.val_datasets[path] = dataset
                # noinspection PyNoneFunctionAssignment
                self.val_loaders[path] = self.get_loader(dataset, is_train=False)
        if len(self.val_datasets) == 0:
            self.val_datasets = None
        if len(self.val_loaders) == 0:
            self.val_loaders = None

        self.save_dir = kwargs['save_dir']
        self.epochs_done = 0
        self.beginning = time.time()
        self.norm = kwargs['norm']
        self.log_weight_norm = kwargs['log_weight_norm']
        self.log_grad_norm = kwargs['log_grad_norm']
        self.grad_clip = kwargs['grad_clip']

        self.save_last = kwargs['save_last']
        self.save_best = kwargs['save_best']
        if self.save_dir is None:
            assert not self.save_last, 'Cannot save only last checkpoint if no save dir is defined!'
            assert self.save_best == '', 'Cannot save best checkpoint if no save dir is defined!'
        self.best_value = None

        beginning_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        if kwargs['name'] is not None:
            self.name = beginning_name + '_' + kwargs['name']
        else:
            self.name = beginning_name

        if kwargs['log_dir'] is not None:
            self.logger = SummaryWriter(os.path.join(kwargs['log_dir'], self.name))
            pretty_print('\tLogging to Tensorboard')
        else:
            self.logger = None

        self.stats_constants = self.get_stats_constants()
        pretty_print('\tCalculated run constants')

        pretty_print('Setup completed')

    @classmethod
    def default_kwargs(cls):
        return {
            'lr_gamma': 1.0,
            'lr_step': 1,
            'name': None,
            'log_dir': None,
            'save_dir': None,
            'norm': 2.0,
            'log_weight_norm': False,
            'log_grad_norm': False,
            'load_weights': None,
            'grad_clip': 0.0,
            'save_last': False,
            'save_best': '',
            'seed': -1,
            'train_paths': None,
            'val_paths': ''
        }

    @classmethod
    def add_args(cls, parser):
        return parser

    @abc.abstractmethod
    def init_net(self):
        pass

    # noinspection PyUnusedLocal
    @abc.abstractmethod
    def get_dataset(self, paths, is_train):
        return None

    # noinspection PyUnusedLocal
    @abc.abstractmethod
    def get_loader(self, dataset, is_train):
        return None

    @abc.abstractmethod
    def init_criterion(self):
        pass

    @abc.abstractmethod
    def init_optimizer(self):
        pass

    def epoch(self, lr=None, train_minibatches=0):
        self.epochs_done += 1

        if self.epochs_done == 1:   # add stats constants to the first epoch
            scalars_summary = self.stats_constants.copy()
        else:
            scalars_summary = {}
        scalars_summary['Iteration'] = self.epochs_done

        train_scalars = self.train(lr=lr, minibatches=train_minibatches)
        for key, value in train_scalars.items():
            scalars_summary['Train/' + key] = value

        if self.save_dir is not None:
            if self.save_last:
                self.save(os.path.join(self.save_dir, self.name + '_last.pt'))
            else:
                self.save(os.path.join(self.save_dir, self.name + '_{}.pt'.format(self.epochs_done)))

        val_scalars = self.val()
        for key, value in val_scalars.items():
            scalars_summary['Validate/' + key] = value

        if lr is None:
            self.scheduler.step()

        scalars_summary['Time'] = time.time() - self.beginning

        # move text scalars to a different dict
        text_summary = {}
        for key in list(scalars_summary.keys()):
            value = scalars_summary[key]
            if isinstance(value, str):
                text_summary[key] = value
                del scalars_summary[key]

        # save best model
        if self.save_best != '':
            if self.save_best not in scalars_summary:
                pretty_print("ERROR: The value {} does not appear in the epoch's scalars!".format(self.save_best))
                pretty_print('ERROR: No best model is saved!')
            else:
                cur_value = scalars_summary[self.save_best]
                if self.best_value is None or cur_value < self.best_value:
                    self.save(os.path.join(self.save_dir, self.name + '_best.pt'))
                    pretty_print('New best value: {}'.format(cur_value))
                    self.best_value = cur_value
            pretty_print('Best value so far: {}'.format(self.best_value))

        if self.logger is not None:
            try:
                log_tensorboard(self.logger, self.epochs_done, scalars=scalars_summary, texts=text_summary)
            except OSError as e:
                pretty_print('ERROR saving to log:', e)

        pretty_print('Epoch', str(self.epochs_done) + ':')
        pretty_print('\tScalars:', json.dumps(scalars_summary, sort_keys=True))
        pretty_print('\tTexts:', json.dumps(text_summary, sort_keys=True))

        return scalars_summary

    def train(self, lr=None, minibatches=0):
        if self.train_loader is None:
            pretty_print('Cannot train; no train loader set up')
            return {}

        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        if minibatches <= 0:
            minibatches = len(self.train_loader)

        self.net.train()

        train_scores = defaultdict(float)
        train_tensors = defaultdict(list)
        for items in tqdm(self.train_loader.get(minibatches), desc=time_str() + ': Training', leave=False,
                          unit='batches', total=minibatches):
            self.optimizer.zero_grad()
            loss, scores = self.get_train_scores(*items)
            loss.backward()

            if self.grad_clip > 0:
                clip_grad_norm(self.net.parameters(), self.grad_clip)

            if self.log_grad_norm or self.log_weight_norm:
                for child_name, child in self.net.named_children():
                    for param_name, param in child.named_parameters():
                        if self.log_grad_norm and param.grad is not None:
                            grad_norm = param.grad.norm(self.norm).data
                            train_tensors['Gradient norm'].append(grad_norm)
                            train_tensors['Gradient norm/' + child_name].append(grad_norm)
                        if self.log_weight_norm:
                            weight_norm = param.norm(self.norm).data
                            train_tensors['Weight norm'].append(weight_norm)
                            train_tensors['Weight norm/' + child_name].append(weight_norm)

            self.optimizer.step()

            train_tensors['Loss'].append(loss.data)
            for score, value in scores.items():
                if is_number(value):
                    train_scores[score] += value
                else:
                    train_tensors[score].append(value)

        for score, value in train_scores.items():
            train_scores[score] = value / len(self.train_dataset)

        for score, tensors in train_tensors.items():
            train_scores[score] = torch.cat(tensors).sum() / minibatches

        pretty_print('Train:', json.dumps(train_scores, sort_keys=True))

        return train_scores

    def val(self):
        if self.val_loaders is None:
            pretty_print('Cannot validate; no val loaders set up')
            return {}

        self.net.eval()

        val_scores = defaultdict(float)
        sum_scores = defaultdict(float)

        for dataset, (loader_name, loader) in zip(self.val_datasets.values(), self.val_loaders.items()):
            loader_scores = defaultdict(float)

            for items in tqdm(loader, desc=time_str() + ': Validating ' + loader_name, leave=False, unit='batches'):
                # noinspection PyNoneFunctionAssignment
                scores = self.get_val_scores(*items)
                for score, value in scores.items():
                    loader_scores[score] += value

            for score, value in loader_scores.items():
                if score == 'Loss':
                    loader_scores[score] = value / len(loader)
                else:
                    loader_scores[score] = value / len(dataset)
                val_scores[loader_name + '/' + score] = loader_scores[score]
                sum_scores[score] += loader_scores[score]

        for score, value in sum_scores.items():
            val_scores[score] = value / len(self.val_loaders)

        pretty_print('Validate:', json.dumps(val_scores, sort_keys=True))

        return val_scores

    # noinspection PyUnusedLocal
    @abc.abstractmethod
    def get_train_scores(self, *items):
        return None, None

    def get_val_scores(self, *items):
        loss, result = self.get_train_scores(*items)
        result['Loss'] = loss.data[0]
        return result

    def get_stats_constants(self):
        results = dict()

        results['CMD'] = ' '.join(map(str, sys.argv))
        for key, value in self.start_args.items():
            results['CMD/' + key] = value

        return results

    @property
    def type_name(self):
        return type(self).__name__

    def save(self, path):
        try:
            torch_save(path, self.net.state_dict())
            pretty_print('Model', self.type_name, 'saved to', path)
        except OSError as e:
            pretty_print("ERROR: Couldn't save model!")
            pretty_print('\tMessage:', e)

    def load(self, path):
        paths = glob(path, recursive=True)
        if len(paths) != 1:
            raise RuntimeError(f'The pattern of the saved weights should match exactly one file! '
                               f'({len(paths)} matches found for the pattern "{path}": {paths})')
        path = paths[0]

        self.net.load_state_dict(torch_load(path))

        pretty_print('Model', self.type_name, 'loaded from', path)

    @classmethod
    def get_args_parser(cls):
        parser = argparse.ArgumentParser()
        for key, value in cls.default_kwargs().items():
            opt = '--' + key.replace('_', '-')
            if value is None:
                parser.add_argument(opt)
            elif value is True:
                parser.add_argument(opt, dest=key, action='store_false')
            elif value is False:
                parser.add_argument(opt, dest=key, action='store_true')
            else:
                parser.add_argument(opt, type=type(value))
        new_parser = cls.add_args(parser)
        if new_parser is not None:
            parser = new_parser
        return parser

    @classmethod
    def parse_defaults_from_file(cls, args):
        # based on https://stackoverflow.com/a/5826167
        # Parse any defaults_file specification
        # We make this parser with add_help=False so that
        # it doesn't parse -h and print help.
        defaults_parser = argparse.ArgumentParser(
            description=__doc__,  # printed with -h/--help
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # Turn off help, so we print all options in response to -h
            add_help=False
        )
        defaults_parser.add_argument('-d', '--defaults-file', default=None, metavar='FILE')
        args, _ = defaults_parser.parse_known_args(args)

        defaults_file = args.defaults_file

        if defaults_file is None:
            return cls.default_kwargs()
        else:
            config = configparser.ConfigParser()
            files_read = config.read([defaults_file])

            if len(files_read) != 1:
                raise RuntimeError('Could not read defaults file: ' + defaults_file)

            defaults = cls.default_kwargs()

            # override from file
            for section in config.sections():
                for key in config.options(section):
                    if key in defaults and type(defaults[key]) == bool:
                        defaults[key] = config.getboolean(section, key)
                    else:
                        defaults[key] = config.get(section, key)

            return defaults

    @classmethod
    def run(cls, args, epochs=None):
        # create parser and original defaults
        parser = cls.get_args_parser()
        parser.add_argument('--epochs', default=0, type=int)
        parser.add_argument('--minibatches', default=0, type=int)
        parser.add_argument('-d', '--defaults-file', default=None, metavar='FILE')  # add to make it visible upon -h

        # read defaults from file
        defaults = cls.parse_defaults_from_file(args)

        # update parser defaults
        parser.set_defaults(**defaults)

        # parse the rest of the command line
        kwargs = vars(parser.parse_args(args=args))
        if epochs is None:
            epochs = kwargs['epochs']
        minibatches = kwargs['minibatches']
        del kwargs['epochs']
        del kwargs['defaults_file']
        del kwargs['minibatches']
        model = cls(**kwargs)
        if epochs is not None:
            for _ in range(epochs):
                model.epoch(train_minibatches=minibatches)
        return model
