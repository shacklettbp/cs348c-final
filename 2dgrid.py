import matplotlib as mpl
from matplotlib import pyplot
import numpy as np
import sys

dim = 256

zvals = np.zeros((dim, dim))

cubes = []
with open(sys.argv[1]) as f:
    for line in f:
        x, y = line.split()
        x = int(x)
        y = int(y)
        zvals[y, x] += 1

# make a color map of fixed colors
cmap = mpl.colors.ListedColormap(['blue','black','red'])

frame1 = pyplot.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.xaxis.set_ticks([])
frame1.axes.yaxis.set_ticks([])


# tell imshow about color map so that only set colors are used
img = pyplot.imshow(zvals,interpolation='nearest',
                    cmap=pyplot.get_cmap('viridis'))

pyplot.savefig("dmp/2d_out.png", bbox_inches='tight')
