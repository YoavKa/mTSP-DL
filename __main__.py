import importlib
import os
import sys
import runpy


DIR_NAME = os.path.basename(os.path.dirname(__file__))
SCRIPTS_PACKAGE = 'src.scripts'


def main(args=None):
    # load sys args if none are given
    if args is None:
        args = sys.argv[1:]

    if len(args) == 0:
        script_name, args = '', []
    else:
        script_name, args = '.' + args[0], args[1:]

    print(f'Invoking script "{script_name}" with args: {args}')

    # try load script
    try:
        entry_point = importlib.import_module(f'.{SCRIPTS_PACKAGE}{script_name}', package=DIR_NAME)
    except ImportError as e:
        print(f'Script "{script_name}" not found: {e.msg}')
        return 1

    # try execute entry point
    if not hasattr(entry_point, 'main'):
        print(f'Script {script_name} has no entry point main')
        return 1
    else:
        return entry_point.main(*args)


if __name__ == '__main__':
    # if we were run as a package
    if __package__:
        exit_code = main()
        if exit_code is not None:
            sys.exit(exit_code)

    # else, run self as package
    else:
        sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
        runpy.run_module(DIR_NAME, run_name='__main__', alter_sys=True)
