import os
from pathlib import Path
from icecream import ic

tree_str = 'cello_performance'
def generate_tree(pathname, n=0):
    global tree_str
    if pathname.is_file():
        tree_str += '    |' * n + '-' * 4 + pathname.name + '\n'
    elif pathname.is_dir():
        tree_str += '    |' * n + '-' * 4 + \
            str(pathname.relative_to(pathname.parent)) + '\\' + '\n'
        for cp in sorted(pathname.iterdir()):
            if str(cp).startswith('.'):
                continue
            if cp.is_dir() and len(os.listdir(str(cp))) > 30:
                tree_str += '    |' * (n+1) + '-' * 4 + str(cp.relative_to(cp.parent)) + '\\' + '  ...' + '\n'
                continue
            generate_tree(cp, n + 1)

if __name__ == '__main__':
    path = './'
    generate_tree(Path(path), 0)
    with open('./path_tree.txt', 'w') as f:
        f.write(tree_str)
    print(tree_str)