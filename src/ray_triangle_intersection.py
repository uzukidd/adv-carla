'''

This code is modified from the following link:
https://github.com/johnnovak/raytriangle-test

'''


import math
import random
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def sub(self, v):
        return Vec3(self.x - v.x,
                    self.y - v.y,
                    self.z - v.z)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v):
        return Vec3(self.y * v.z - self.z * v.y,
                    self.z * v.x - self.x * v.z,
                    self.x * v.y - self.y * v.x)

    def length(self):
        return math.sqrt(self.x * self.x +
                         self.y * self.y +
                         self.z * self.z)

    def normalize(self):
        l = self.length()
        return Vec3(self.x / l, self.y / l, self.z / l)


class Ray:
    def __init__(self, orig=None, direction=None):
        self.orig = orig
        self.direction = direction


def ray_triangle_intersect(r, v0, v1, v2):
    v0v1 = v1.sub(v0)
    v0v2 = v2.sub(v0)
    pvec = r.direction.cross(v0v2)

    det = v0v1.dot(pvec)

    if det < 0.000001:
        return float('-inf')

    invDet = 1.0 / det
    tvec = r.orig.sub(v0)
    u = tvec.dot(pvec) * invDet

    if u < 0 or u > 1:
        return float('-inf')

    qvec = tvec.cross(v0v1)
    v = r.direction.dot(qvec) * invDet

    if v < 0 or u + v > 1:
        return float('-inf')

    return v0v2.dot(qvec) * invDet


def random_vertex():
  return Vec3(random.random() * 2.0 - 1.0,
              random.random() * 2.0 - 1.0,
              random.random() * 2.0 - 1.0)


def generate_random_triangles(numTriangles):
    vertices = [None] * numTriangles * 3

    for i in range(0, numTriangles):
        vertices[i*3 + 0] = random_vertex()
        vertices[i*3 + 1] = random_vertex()
        vertices[i*3 + 2] = random_vertex()

    return vertices


def random_sphere():
    # limit [-1, 1] for x, y, z
    r1 = random.random()
    r2 = random.random()
    lat = math.acos(2*r1 - 1) - math.pi/2
    lon = 2*math.pi * r2

    return Vec3(math.cos(lat) * math.cos(lon),
                math.cos(lat) * math.sin(lon),
                math.sin(lat))

def calculate_intersection_point(r, t, v0, v1, v2):
    # Compute the intersection point using ray parameter t
    intersection_point = Vec3(r.orig.x + t * r.direction.x,
                               r.orig.y + t * r.direction.y,
                               r.orig.z + t * r.direction.z)
    
    # Print or use the intersection_point as needed
    print("Intersection point:", intersection_point.x, intersection_point.y, intersection_point.z)
    return intersection_point

def plot_ray_triangle_intersection(intersection_point, r, t, v0, v1, v2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the triangle
    triangle_vertices = [v0, v1, v2, v0]
    triangle_vertices = np.array([[v.x, v.y, v.z] for v in triangle_vertices])
    ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2], color='blue')

    # Plot the intersection point if it's valid
    if not math.isinf(t):
        ax.scatter(intersection_point.x, intersection_point.y, intersection_point.z, color='red', s=100)

    # Plot the ray
    ray_end_point = Vec3(r.orig.x + r.direction.x, r.orig.y + r.direction.y, r.orig.z + r.direction.z)
    ray_vertices = np.array([[r.orig.x, r.orig.y, r.orig.z], [ray_end_point.x, ray_end_point.y, ray_end_point.z]])
    ax.plot(ray_vertices[:, 0], ray_vertices[:, 1], ray_vertices[:, 2], color='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

NUM_RAYS = 64
NUM_TRIANGLES = 100 * 100

random.seed()

vertices = generate_random_triangles(NUM_TRIANGLES)
num_vertices = NUM_TRIANGLES * 3

num_hit = 0
num_miss = 0

r = Ray()

t_start = timer()

for i in range(0, NUM_RAYS):
    r.orig = random_sphere()
    r.direction = random_sphere().sub(r.orig).normalize()

    for j in range(0, int(num_vertices / 3)):
        t = ray_triangle_intersect(r, vertices[j*3 + 0],
                                      vertices[j*3 + 1],
                                      vertices[j*3 + 2])
        if t >= 0:
          num_hit += 1
          intersection_point = calculate_intersection_point(r, t, vertices[j*3 + 0],
                                       vertices[j*3 + 1],
                                       vertices[j*3 + 2])
          plot_ray_triangle_intersection(intersection_point, r, t, vertices[j*3 + 0],
                                       vertices[j*3 + 1],
                                       vertices[j*3 + 2])
        else:
          num_miss += 1

t_end = timer()
t_total = t_end - t_start

num_tests = NUM_RAYS * NUM_TRIANGLES
hit_perc  = float(num_hit) / num_tests * 100
miss_perc = float(num_miss) / num_tests * 100
mtests_per_second = float(num_tests) / t_total / 1000000

#print 'Total intersection tests:  %11d' % (num_tests)
#print '  Hits:\t\t\t   %11d (%5.2f%%)' % (num_hit, hit_perc)
#print '  Misses:\t\t   %11d (%5.2f%%)' % (num_miss, miss_perc)
#print
#print '  Total time:\t\t\t  %6.2f seconds' % (t_total)
#print '  Millions of tests per second:\t  %6.2f' % (mtests_per_second)
print(mtests_per_second)
print(t_total)

