import os

def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)