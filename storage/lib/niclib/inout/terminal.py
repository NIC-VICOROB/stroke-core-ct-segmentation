import sys


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 25, fill = '='):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)

    bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

    print('\r{} [{}] {}% {}'.format(prefix, bar, percent, suffix), end='\r')
    sys.stdout.flush()

    # Print New Line on Complete
    if iteration == total:
        print(' ')

