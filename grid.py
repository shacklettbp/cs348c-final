from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
import sys

# Adjust appearance of cubes
alpha = 1
facecolor = "red"
axis_length = 32

# x, y, z: define front bottom left coordinate of unit cube
# ax     : the figure to plot to
def plot_cube(ax, x, y, z):
    v = np.array([[x, y, z],     [x, y, z+1],
                  [x, y+1, z],   [x, y+1, z+1],
                  [x+1, y, z],   [x+1, y, z+1],
                  [x+1, y+1, z], [x+1, y+1, z+1]])
    # Plot the vertices of the cube
    ax.scatter3D(v[:,0], v[:,1], v[:,2], marker="")

    # Define the six cube faces using the calculated vertices
    faces = [[v[0],v[2],v[3],v[1]], [v[4],v[6],v[7],v[5]],
             [v[0],v[4],v[5],v[1]], [v[3],v[2],v[6],v[7]],
             [v[0],v[4],v[6],v[2]], [v[1],v[5],v[7],v[3]]]
    pc = Poly3DCollection(faces, linewidths=0)
    pc.set_alpha(alpha)
    pc.set_facecolor(facecolor)
    ax.add_collection(pc)

# vertex_list: list of vertices represented as tuples
def plot_all_cubes(vertex_list):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(0, axis_length)
    ax.set_ylim(0, axis_length)
    ax.set_aspect("equal") # Looks nice if given a cube

    for vertex in vertex_list:
        x, y, z = vertex
        plot_cube(ax, x, y, z)

    plt.show()
        
cubes = []
with open(sys.argv[1]) as f:
    for line in f:
        x, y, z = line.split()
        x = int(x)
        y = int(y)
        z = axis_length - int(z)
        vertex = (x, y, z)
        cubes.append(vertex)

print("Plotting\n")

plot_all_cubes(cubes)
