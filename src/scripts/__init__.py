from ..utils import PROJECT_DIR


def main(*_):
    # if we reached here, the user did not specify a specific script
    print(f'USAGE: python {PROJECT_DIR} script_name [args [...]]')
    return 1
