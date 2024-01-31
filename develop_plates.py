#!/usr/bin/env python3

from collections import OrderedDict
import logging
import argparse
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import functools
import math
import enum

import abc


def sequence(vertices, indices):
    return np.array([vertices[index] for index in indices])
    
def invert(indices):
    result = np.full(np.max(indices) + 1, -1, dtype=np.int32)
    for index, value in enumerate(indices):
        result[value] = index
        
    for value in result:
        if value == -1:
            logging.warning("Non-homomorphic indicies only partially inverted: %s -> %s", indices, result)
            break
            
    return result
    
def unsequence(vertices, indices):
    shape = list(np.shape(vertices))
    shape[0] = np.max(indices) + 1
    result = np.full(shape, np.NAN, dtype=vertices.dtype)
    for index, value in enumerate(indices):
        result[value] = vertices[index]
        
    for value in result:
        if value is np.NAN:
            logging.warning("Non-homomorphic indicies only partially unsequenced: %s", indices)
            break
    return result
    
def project(indices, ref_indices):
    inverted = invert(ref_indices)
    result = np.array([inverted[index] for index in indices], dtype=np.int32)
    for value in result:
        if value == -1:
            raise ValueError("Cannot project %s through non-homomorphic indices %s!", indices, ref_indices)
    return result

def index_of(vertices, value):
    for index, element in enumerate(vertices[:]):
        if (value == element).all():
            return index
    raise ValueError(f"Not found in array: {value}, {vertices}")
    
    
def unit(vector):
    return vector * 1/np.linalg.norm(vector)
    
def sign(number):
    return -1 if number < 0 else 1

def n_wise(iterable, n, stride=1):
    '''
    iterate over the iterable, N elements at a time e.g. pairwise or more.
    '''
    qty = len(iterable)
    iterables = ( iterable[x:qty+x+1-n:stride] for x in range(n) )
    yield from zip(*iterables)
    
def flatten(vertices, origin=None, basis = None, tolerance = 0.01):
    '''
    Vertices is a sequence of vertices that form a triangle strip (should be
    really familiar for open GL users).
    '''
    
    if origin is None:
        origin = vertices[0, :2].copy()
    
    # 2nd most recent vertex--anchor for flattening vertex by distance
    v2 = vertices[0]
    p2 = origin

    # 1st most recent vertex--anchor for flattening vertex by distance
    v1 = vertices[1]
    
    if basis is None:
        basis = v1 - v2
        basis = unit(basis[:2])
    p1 = origin + basis *np.linalg.norm(v2 - v1)
    
    # running error statistics
    max_error = 0
    total_error = 0
    
    result = [p2, p1]
    even = False
    for v0 in vertices[2:]:
        v2_1 = v1 - v2
        v2_0 = v0 - v2
        v1_0 = v0 - v1
        plane_u = unit(v2_1)
        
        # Establish a unit perpendicular to plane_u in the same plane
        plane_normal = np.cross(plane_u, v2_0)
        plane_v = unit(np.cross(plane_u, plane_normal))
        
        # Project length onto u, v vectors
        du_2_0 = np.dot(plane_u, v2_0)
        dv_2_0 = np.dot(plane_v, v2_0)
        du_1_0 = np.dot(plane_u, v1_0)
        dv_1_0 = np.dot(plane_v, v1_0)
        
        # Establish u, v vectors in xy point space and project
        #
        p2_1 = p1 - p2
        xy_u = unit(p2_1)
        if even:
            xy_v = np.array((xy_u[1], -xy_u[0]))
        else:
            xy_v = np.array((-xy_u[1], xy_u[0]))
        p0 = p2 + xy_u * du_2_0 + xy_v * dv_2_0
        p0_check = p1 + xy_u * du_1_0 + xy_v * dv_1_0
        
        # Calculate error
        #
        expected_d2_0 = np.linalg.norm(v0 - v2)
        expected_d1_0 = np.linalg.norm(v0 - v1)
        
        actual_d2_0 = np.linalg.norm(p0 - p2)
        actual_d1_0 = np.linalg.norm(p0 - p1)
        
        error_d2_0 = abs(expected_d2_0 - actual_d2_0)
        error_d1_0 = abs(expected_d1_0 - actual_d1_0)
        
        max_error = max(error_d2_0, error_d1_0, max_error)
        total_error += error_d2_0 + error_d1_0
        
        assert(np.linalg.norm(p0 - p0_check) < tolerance)
        assert(max_error < tolerance)
        
        # iterate
        #
        result.append(p0)
        
        v2 = v1
        p2 = p1
        
        v1 = v0
        p1 = p0
        even = not(even)
        
    # TODO: error is always more or less zero here. Instead we should look at unmap error(e.g. if points are reused).
        
    return np.array(result), total_error, max_error  

def reorient(origin, x_crossing, points):
    '''
    move points so that they originate at the specified location and the x_axis is aligned
    '''
    pointer = x_crossing - origin
    angle = np.arctan2(pointer[1], pointer[0])
    cos = np.cos(angle)
    sin = np.sin(angle)
    r = np.full((2, 2), cos)
    r[0, 1] = -sin
    r[1, 0] = sin
    
    return np.matmul(points - origin, r)
    
    
class Line():
    
    __slots__ = ("offset", "unit", "scale")
    
    @classmethod
    def axis(cls, total, axis, dtype=np.double):
        unit = np.zeros(total, dtype=dtype)
        unit[axis] = 1.0
        return cls(np.zeros(total, dtype=dtype), unit, 1.0)
    
    def __init__(self, offset, unit, scale=None):
        self.offset = offset
        if scale is None:
            self.scale = np.linalg.norm(unit)
            self.unit = unit / self.scale
        else:
            self.scale = scale
            self.unit = unit
            
    @classmethod
    def fit(cls, points):
        a = np.vstack((points[:, 0], np.ones(len(points))))
        a = np.transpose(a)
        b = points[:, 1:]
        m, c = np.linalg.lstsq(a, b)[0]
        # TODO: handle 3d lines
        return cls(np.array((1, c)), np.array((1, m)))
        
    def unmap(self, t):
        return self.offset + self.unit * self.scale * t
        
    def map(self, point, check=1e-16):
        v = point - self.offset
        t = np.dot(v, self.unit)
        if check:
            assert(abs(np.linalg.norm(v) - abs(t)) < check)
        return t / self.scale
        
    def solve(self, dim, value):
        return (value - self.offset[dim])/(self.scale * self.unit[dim])
        
    def where(self, dim, value):
        t = self.solve(dim, value)
        return self.unmap(t)
        
    @classmethod
    def between(cls, a, b, normalize=True):
        v = b - a
        scale = np.linalg.norm(v)
        v /= scale
        
        if normalize:
            scale = 1.0
            
        return cls(a, v, scale)
        
    def intersect(self, plane):
        # solve T for dot(t*U + Ol - Op, N) = 0
        #
        # Let Ol - Op = C
        # t*dot(U,N)  + dot(C,N) = 0
        # t*dot(U,N) = - dot(C, N)
        # t = - dot(C,N)/dot(U,N) 
        c = self.offset - plane.offset
        return - np.dot(c, plane.normal) / (self.scale * np.dot(self.unit, plane.normal))
        
    def nearest(self, other):
        '''
        find the pair of points closest to intersection
        '''
        # U1 x U2 = N
        # C = O1 - O2
        # t1 = - dot(C,N)/dot(U1*s1,N) 
        # t2 = + dot(C,N)/dot(U2*s2,N) 
        normal = np.cross(self.unit, other.unit)
        c = self.offset - other.offset
        t1 = - np.dot(c, plane.normal) / (self.scale * np.dot(self.unit, normal))
        t2 =   np.dot(c, plane.normal) / (other.scale * np.dot(other.unit, normal))
        
        return (t1, t2)
    
class Plane():

    __slots__ = ("uv", "offset", "normal")
    
    def __init__(self, uv, offset, normal):
        self.uv = uv
        self.offset = offset
        self.normal = normal
        
    @classmethod
    def from_points(cls, a, b, c):
        vec1 = a - b
        vec2 = c - b
        normal = np.cross(vec1, vec2)

        u = unit(vec1)
        v = unit(np.cross(vec1, normal))
        uv = np.stack((u,v))
        return cls(uv, b, normal)
        
    def map(self, point, check=1e-16):
        rel = point - self.offset
        if check:
            assert(abs(np.dot(rel, self.normal)) < check)
        return np.matmul(self.uv, rel)
        
    def unmap(self, point):
        return self.offset + np.matmul(point, self.uv)
        
    @classmethod
    def from_lines(cls, u, v, offset=None):
        uv = np.stack((u.unit, v.unit))
        normal = np.cross(u.unit, v.unit)
        if offset is None:
            offset = u.offset
        return cls(uv, offset, normal)
        
    def distance(self, point):
        return np.dot(self.normal, point - self.offset)
        
class Arc():

    def __init__(self, a, b, c):
        self.points = (a, b, c)
        self.plane = Plane.from_points(a, b, c)
        
        p0 = self.plane.map(a) # (||a-b||, 0)
        p1 = self.plane.map(b) # (0, 0)
        p2 = self.plane.map(c)
        
        # ||(p0 - center)|| = r**2
        # ||(p1 - center)|| = r**2
        # ||(p2 - center)|| = r**2
        #
        # ||(p0 - center)|| = ||(p1 - center)||
        #
        # Expand ||pN|| == pN_x**2  + pN_y**2
        #
        # (p0_x - center_x)**2 + (p0_y - center_y)**2 = (p1_y - center_y)**2 + (p1_y - center_y)**2
        #
        # Solve for center_x
        #
        #    p0_x**2 - 2*p0_x*center_x + center_x**2 + p0_y**2 - 2*p0_y*center_y + center_y**2
        # == p1_x**2 - 2*p1_x*center_x + center_x**2 + p1_y**2 - 2*p1_y*center_y + center_y**2
        #
        # Note center_x**2 and center_y**2 cancel out
        #
        #    p0_x**2 - 2*p0_x*center_x + p0_y**2 - 2*p0_y*center_y
        # == p1_x**2 - 2*p1_x*center_x + p1_y**2 - 2*p1_y*center_y
        #
        # Group center_x and center y on each side
        #
        #    p0_x**2 - 2*p0_x*center_x - p1_x**2 + 2*p1_x*center_x
        # == p1_y**2 - 2*p1_y*center_y - p0_y**2 + 2*p0_y*center_y
        #
        # Factor out 2 center_x/y
        #
        #    p0_x**2 - p1_x**2 + 2*center_x*(p1_x - p0_x) 
        # == p1_y**2 - p0_y**2 + 2*center_y*(p0_y - p1_y)
        #
        #    2*center_x*(p1_x - p0_x) 
        # == (p1_y**2 + p1_x**2 - p0_y**2 - p0_x**2) + 2*center_y*(p0_y - p1_y)
        #
        #    2*center_x*(p1_x - p0_x)
        # == (||p1|| - ||p0||) + 2*center_y*(p0_y - p1_y)
        #
        #    2*center_x
        # == (||p1|| - ||p0||)/(p1_x - p0_x) + 2*center_y*(p0_y - p1_y)/(p1_x - p0_x)
        # == (||p1|| - ||p0||)/(p1_x - p0_x) - 2*center_y*(p1_y - p0_y)/(p1_x - p0_x)
        #
        # Substitute p2, p1 for p1, p0 to find second solution for 2*center_x
        #
        #    2*center_x
        # == (||p2|| - ||p1||)/(p2_x - p1_x) - 2*center_y*(p2_y - p1_y)/(p2_x - p1_x)
        #
        # Solve for center_y
        #
        #    (||p1|| - ||p0||)/(p1_x - p0_x) - 2*center_y*(p1_y - p0_y)/(p1_x - p0_x)
        # == (||p2|| - ||p1||)/(p2_x - p1_x) - 2*center_y*(p2_y - p1_y)/(p2_x - p1_x)
        #
        #    2*center_y*(p2_y - p1_y)/(p2_x - p1_x) - 2*center_y*(p1_y - p0_y)/(p1_x - p0_x)
        # == (||p2|| - ||p1||)/(p2_x - p1_x) - (||p1|| - ||p0||)/(p1_x - p0_x)
        #
        #    2*center_y*((p2_y - p1_y)/(p2_x - p1_x) - (p1_y - p0_y)/(p1_x - p0_x))
        # == (||p2|| - ||p1||)/(p2_x - p1_x) - (||p1|| - ||p0||)/(p1_x - p0_x)
        #
        # Make shared denominators (p2_x - p1_x)*(p1_x - p0_x) and cancel
        #
        #    2*center_y*((p2_y - p1_y)*(p1_x - p0_x) - (p1_y - p0_y)*(p2_x - p1_x))/((p2_x - p1_x)*(p1_x - p0_x))
        # == ((||p2|| - ||p1||)*(p1_x - p0_x) - (||p1|| - ||p0||)*(p2_x - p1_x))/((p2_x - p1_x)*(p1_x - p0_x))
        #
        #    2*center_y*((p2_y - p1_y)*(p1_x - p0_x) - (p1_y - p0_y)*(p2_x - p1_x))
        # == (||p2|| - ||p1||)*(p1_x - p0_x) - (||p1|| - ||p0||)*(p2_x - p1_x)
        #
        #    2*center_y
        # == (||p2|| - ||p1||)*(p1_x - p0_x) - (||p1|| - ||p0||)*(p2_x - p1_x)
        #  / ((p2_y - p1_y)*(p1_x - p0_x) - (p1_y - p0_y)*(p2_x - p1_x))
        #
        # Expect symmetry
        #
        #    2*center_x
        # == (||p2|| - ||p1||)*(p1_y - p0_y) - (||p1|| - ||p0||)*(p2_y - p1_y)
        #  / ((p2_x - p1_x)*(p1_y - p0_y) - (p1_x - p0_x)*(p2_y - p1_y))
        #
        # Check
        #
        #    2*center_x
        # == (||p1|| - ||p0||)/(p1_x - p0_x) - 2*center_y*(p1_y - p0_y)/(p1_x - p0_x)
        # == (||p1|| - ||p0|| - 2*center_y*(p1_y - p0_y))/(p1_x - p0_x)
        #
        #    Q
        # == ||p1|| - ||p0|| - 2*center_y*(p1_y - p0_y)
        # == ||p1|| - ||p0|| - (p1_y - p0_y)*((||p2|| - ||p1||)*(p1_x - p0_x) - (||p1|| - ||p0||)*(p2_x - p1_x)) / ((p2_y - p1_y)*(p1_x - p0_x) - (p1_y - p0_y)*(p2_x - p1_x))
        #
        #    Q*((p2_y - p1_y)*(p1_x - p0_x) - (p1_y - p0_y)*(p2_x - p1_x))
        # == (||p1|| - ||p0||)*((p2_y - p1_y)*(p1_x - p0_x) - (p1_y - p0_y)*(p2_x - p1_x))) 
        #  - (p1_y - p0_y)*((||p2|| - ||p1||)*(p1_x - p0_x) - (||p1|| - ||p0||)*(p2_x - p1_x))
        # == (||p1|| - ||p0||)*(p2_y - p1_y)*(p1_x - p0_x) - (||p1|| - ||p0||)*(p1_y - p0_y)*(p2_x - p1_x))
        #  + (||p1|| - ||p0||)*(p1_y - p0_y)*(p2_x - p1_x) - (||p2|| - ||p1||)*(p1_y - p0_y)*(p1_x - p0_x)
        # == (||p1|| - ||p0||)*(p2_y - p1_y)*(p1_x - p0_x) - (||p2|| - ||p1||)*(p1_y - p0_y)*(p1_x - p0_x)
        # == -(p1_x - p0_x)*((||p2|| - ||p1||)*(p1_y - p0_y) - (||p1|| - ||p0||)*(p2_y - p1_y))
        #
        #    Q
        # == -(p1_x - p0_x)*((||p2|| - ||p1||)*(p1_y - p0_y) - (||p1|| - ||p0||)*(p2_y - p1_y))
        #  / ((p2_y - p1_y)*(p1_x - p0_x) - (p1_y - p0_y)*(p2_x - p1_x))
        #
        #    2*center_x
        # == -(p1_x - p0_x)*((||p2|| - ||p1||)*(p1_y - p0_y) - (||p1|| - ||p0||)*(p2_y - p1_y))
        #  / ((p2_y - p1_y)*(p1_x - p0_x) - (p1_y - p0_y)*(p2_x - p1_x))
        #  / (p1_x - p0_x)
        # == -((||p2|| - ||p1||)*(p1_y - p0_y) - (||p1|| - ||p0||)*(p2_y - p1_y))
        #  / ((p2_y - p1_y)*(p1_x - p0_x) - (p1_y - p0_y)*(p2_x - p1_x)))
        # == ((||p2|| - ||p1||)*(p1_y - p0_y) - (||p1|| - ||p0||)*(p2_y - p1_y))
        #  / ((p1_y - p0_y)*(p2_x - p1_x) - (p2_y - p1_y)*(p1_x - p0_x))
        #
        # !!!
        #
        l2_0 = np.dot(p0, p0)
        l2_1 = np.dot(p1, p1)
        l2_2 = np.dot(p2, p2)
        p0_1 = p1 - p0
        p1_2 = p2 - p1
        
        numerator_y = (l2_2 - l2_1) * p0_1[0] - (l2_1 - l2_0) * p1_2[0]
        denominator = (p0_1[1] * p1_2[0] - p1_2[1]*  p0_1[0] )*2
        
        center_y = -numerator_y / denominator
        
        numerator_x = (l2_2 - l2_1) * p0_1[1] - (l2_1 - l2_0) * p1_2[1]
        
        center_x = numerator_x / denominator
        
        center = np.array((center_x, center_y))
        
        r0 = np.linalg.norm(p0 - center)
        r1 = np.linalg.norm(p0 - center)
        r2 = np.linalg.norm(p0 - center)
        assert(r0 == r1)
        assert(r1 == r2)
        
        self.radius = r0
        self.center = center
        #print(f"{self.center} {self.radius}")
        
        a1 = np.arctan2((p0 - center)[1], (p0 - center)[0])
        a2 = np.arctan2((p1 - center)[1], (p1 - center)[0])
        a3 = np.arctan2((p2 - center)[1], (p2 - center)[0])
        
        d12 = a2 - a1
        d13 = a3 - a1
        
        if sign(d12) != sign(d13):
            d13 -= sign(d13) * 2 * np.pi
            
        if abs(d12) > abs(d13):
            # need to go to d2 the opposite direction
            d13 = sign(d13) * 2 * np.pi - d13
            
        self.angles = (a1, a2, a3)
        self.distances = (d12, d13)
            
        #print(f"{np.degrees(a1)} {np.degrees(d12)} {np.degrees(d13)}")
        
    def unmap(self, angle):
        ray = np.array((np.cos(angle), np.sin(angle))) * self.radius
        return self.plane.unmap(self.center + ray)

def arc_interpolate(vertices, count):
    '''
    Generate count points between vertices, using trig with continuity
    '''
    v0, v1 = vertices[:2]
    arc_0 = None
    
    result = []
    
    segments = count + 1
    
    for v2 in vertices[2:]:
        arc_1 = Arc(v0, v1, v2)
        result.append(v0)
        
        step_1 = arc_1.distances[0] / segments
        offset_1 = arc_1.angles[0]
        
        if arc_0:
            step_0 = (arc_0.distances[1] - arc_0.distances[0]) / segments
            offset_0 = arc_0.angles[0] + arc_0.distances[0]
            
        for segment in range(1, segments):
            point_1 = arc_1.unmap(offset_1 + step_1 * segment)
            if arc_0:
                point_0 = arc_0.unmap(offset_0 + step_0 * segment)
                
                # Angular weighting
                k = step_0 / (step_0 + step_1)
                w1 = segment / segments
                w0 = 1.0 - w1
                w0 *= 1.0 - k
                w1 *= k
                
                point = point_1 * w1 + point_0 * w0
                point /= w0 + w1
            else:
                point = point_1
            result.append(point)
        
        arc_0 = arc_1
        v0 = v1
        v1 = v2
        
    result.append(v0)

    step_0 = (arc_0.distances[1] - arc_0.distances[0]) / segments
    offset_0 = arc_0.angles[0] + arc_0.distances[0]
    for segment in range(1, segments):
        point = arc_0.unmap(offset_0 + step_0 * segment)
        result.append(point)
        
    result.append(v1)
    return np.array(result)
    
class Panel():
    '''
    Single panel to be developed
    '''

    __slots__ = ('vertices', 'triangles', 'perimeters', 'points', 'quantity', 'mirrors')
    
    def __init__(self, vertices, triangles, perimeters, quantity=1, points=None, mirrors=tuple()):
        self.vertices = vertices
        self.triangles = triangles
        self.perimeters = perimeters
        self.quantity = quantity
        self.points = points
        self.mirrors = [np.ones(3)]
        self.mirrors.extend(mirrors)
        
    def flatten(self, tolerance, offset=None, mirror=True):
        triangle_strip = sequence(self.vertices, self.triangles)
        points, total_error, max_error = flatten(triangle_strip, tolerance=tolerance)
        logging.info("Total error: %f, max error: %f", total_error, max_error)
        #center = np.mean(points, axis=0)
        #axis = np.sum(np.absolute(points - center), axis=0)
        #points = reorient(center, center + axis, points)
        
        linear_regression = Line.fit(points)
        points = reorient(linear_regression.unmap(points[0, 0]), linear_regression.unmap(points[0, 0] + 1), points)
        
        
        #points = reorient(points[0], points[-1] + points[-2], points)
        if mirror:
            points *= np.array((1.0, -1.0))
        if offset:
            points += offset
        else:
            points += triangle_strip[0][:2]
        self.points = unsequence(points, self.triangles)
        
        
    def area(self):
        if self.points is None:
            raise ValueError("Cannot plot non-existant points!")
        triangles = sequence(self.points, self.triangles)
        p0, p1 = triangles[:2]
        
        area = 0
        for p2 in triangles[2:]:
            area += 0.5 * abs(p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]))
            p0 = p1
            p1 = p2
            
        return area
        
    def plot_points(self, ax, z=False, offset=None, scale=1):
        if self.points is None:
            raise ValueError("Cannot plot non-existant points!")
            
        # guess an offset
        if offset is None:
            z_guess = 0 #-np.mean(self.vertices[:, 2])
            x_guess = 0 #-np.min(self.points[:, 0]) #-abs(np.mean(self.vertices[:, 1]))
            offset = np.array((x_guess, 0, z_guess))
            
        points_2d = sequence(self.points, self.triangles) * scale + offset[:2]
        perimeters = [sequence(self.points, perimeter) * scale  + offset[:2] for perimeter in self.perimeters]
        if z:
            ax.plot(points_2d[:, 0], points_2d[:, 1], np.full(np.shape(points_2d)[0], offset[2]))
            for perimeter in perimeters:
                ax.plot(perimeter[:, 0], perimeter[:, 1], np.full(np.shape(perimeter)[0], offset[2]))
        else:
            ax.plot(points_2d[:, 0], points_2d[:, 1])
            for perimeter in perimeters:
                ax.plot(perimeter[:, 0], perimeter[:, 1])
        
    
    def plot_vertices(self, ax, z=False, scale=1):
        triangle_strip = sequence(self.vertices, self.triangles) * scale
        perimeters = [sequence(self.vertices, perimeter) * scale for perimeter in self.perimeters]
        if z:
            if len(triangle_strip):
                ax.plot(triangle_strip[:, 0], triangle_strip[:, 1], triangle_strip[:, 2])
            for perimeter in perimeters:
                ax.plot(perimeter[:, 0], perimeter[:, 1], perimeter[:, 2])
        else:
            if len(triangle_strip):
                ax.plot(triangle_strip[:, 0], triangle_strip[:, 1])
            for perimeter in perimeters:
                ax.plot(perimeter[:, 0], perimeter[:, 1])
                
def coaming_front_curve(plane, bounding_box, samples, offset=(0,0)):
    vertices = []
    scale = bounding_box[:2] * np.array((1.0, 0.5))
    for index in range(samples):
        angle = index * np.pi / (2 * samples)
        point = offset + scale * (np.sin(angle), np.cos(angle))
        vertices.append(plane.unmap(point))
        
    return vertices

def coaming_back_curve(plane, bounding_box, samples, offset=(0,0)):
    vertices = []
    scale = bounding_box[:2] * np.array((1, 0.25))
    offset = np.array((scale[0], bounding_box[1] * 0.25)) + offset
    for index in range(samples):
        angle = index * np.pi / (2 * samples)
        point = offset + scale * (-np.cos(angle), np.sin(angle))
        vertices.append(plane.unmap(point))
        
    return vertices
    
def coaming_ring(plane, bounding_box, samples, split_x, diameter, qty=1, offset=(0,0)):
        """
        Creates a ring with the backmost inner opening at offset and spanning
        the bounding box in the x axis of the plane. The outer ring is offset
        by diameter in each direction. split_x defines the transition between
        front and back curves (relative to bounding box)
        """
        mirror = np.array((1, -1, 1), dtype=np.double)
        
        coaming_inner_back_shape = np.array((split_x, bounding_box[1]))
        coaming_inner_back_offset = np.zeros(2, dtype=np.double) + offset
        coaming_inner_back_samples = samples # match for triangulation
        coaming_inner_back = coaming_back_curve(plane, coaming_inner_back_shape, coaming_inner_back_samples, coaming_inner_back_offset)
        
        coaming_inner_front_shape = np.array((bounding_box[0] - split_x, bounding_box[1]))
        coaming_inner_front_offset = np.array((split_x, 0)) + offset
        coaming_inner_front_samples = 2 * samples # just a guess
        coaming_inner_front = coaming_front_curve(plane, coaming_inner_front_shape, coaming_inner_front_samples, coaming_inner_front_offset)
        
        coaming_inner_peak = plane.unmap((bounding_box[0], 0))
        
        coaming_outer_back_shape = np.array((split_x, bounding_box[1])) + (diameter, diameter)
        coaming_outer_back_offset = np.array((-diameter, 0)) + offset
        coaming_outer_back_samples = samples # match for triangulation
        coaming_outer_back = coaming_back_curve(plane, coaming_outer_back_shape, coaming_outer_back_samples, coaming_outer_back_offset)
        
        coaming_outer_front_shape = np.array((bounding_box[0] - split_x, bounding_box[1])) + (diameter, diameter)
        coaming_outer_front_offset = np.array((split_x, 0)) + offset
        coaming_outer_front_samples = 2 * samples # just a guess
        coaming_outer_front = coaming_front_curve(plane, coaming_outer_front_shape, coaming_outer_front_samples, coaming_outer_front_offset)
        
        coaming_outer_peak = plane.unmap((bounding_box[0] + diameter, 0))
        
        internal_vertices = np.concatenate((coaming_inner_back, coaming_inner_front))
        internal_vertices = np.concatenate((internal_vertices, (coaming_inner_peak,), np.flip(internal_vertices, 0) * mirror))
        
        external_vertices = np.concatenate((coaming_outer_back, coaming_outer_front))
        external_vertices = np.concatenate((external_vertices, (coaming_outer_peak,), np.flip(external_vertices, 0) * mirror))
        
        assert(len(internal_vertices) == len(external_vertices))
        
        vertices = np.concatenate((external_vertices, internal_vertices))
        
        indices = []
        for index in range(len(external_vertices)):
            indices.append(index)
            indices.append(len(external_vertices) + index)
        
        # Patch in last triangle
        indices.append(0)
        indices.append(len(external_vertices))
        
        external_perimeter = list(range(len(external_vertices)))
        external_perimeter.append(0)
        
        internal_perimeter = list(range(len(external_vertices), len(vertices)))
        internal_perimeter.append(len(external_vertices))
        perimeters = [ np.array(external_perimeter), np.array(internal_perimeter) ]

        return Panel(vertices, np.array(indices, dtype=np.int32), perimeters, qty)


class Part():
    """
    A shape to be machined, or an intermediate step on a shape to be machined
    """
    __slots__ = ("perimeter", "holes")
    
    def __init__(self, perimeter, holes = None):
        self.perimeter = perimeter
        self.holes = holes or []
        
        
    def translate(self, offset):
        perimeter = self.perimeter + offset
        holes = [ hole + offset for hole in self.holes ]
        return Part(perimeter, holes)
        
    def rotate(self, angle, center=np.array((0, 0))):
        r = np.full((2, 2), np.cos(angle))
        sin_a = np.sin(angle)
        r = -sin_a
        r = sin_a
        
        perimeter = (self.perimeter - center) * r + center
        holes = [ (hole - center) * r + center for hole in holes ]
        return Part(perimeter, holes)
        
class Node(abc.ABC):
    """
    An operation that produces zero or more parts.
    """
    
    __slots__ = ("parent", "children")
    
    def __init__(self, parent):
        self.parent = parent
        self.children = []
        parent.children.append(self)
        
    @abc.abstractmethod
    def parts(self):
        return None

class Cache(Node):
    __slots__ = ("cached_parts",)
    def __init__(self, parent, offset):
        super().__init__(parent)
        self.cached_parts = None
        
    def parts(self):
        if self.cached_parts is None:
            self.cached_parts = self.parent.parts()
        return self.cached_parts
        
class Flatten(Node):
    """
    Maps a panel into a 2d part
    """
    
    __slots__ = ("panel", "tolerance")
    
    def __init__(self, panel, tolerance):
        super().__init__(None)
        self.panel = panel
        self.tolerance = tolerance
        
    def parts(self):
        self.panel.flatten(self.tolerance)
        perimeter = sequence(self.panel.vertices, self.panel.perimeters[0])
        holes = [sequence(self.panel.vertices, perimeter) for perimeter in self.panel.perimeters[1:]]
        return [Part(perimeter, holes)]
        
class Translate(Node):

    __slots__ = ("offset",)
    
    def __init__(self, parent, offset):
        super().__init__(parent)
        self.offset = offset
        
    def parts(self):
        return [part.translate(self.offset) for part in self.parent.parts()]
        
class Rotate(Node):

    __slots__ = ("angle",)
    
    def __init__(self, parent, angle):
        super().__init__(parent)
        self.angle = angle
        
    def parts(self):
        return [part.rotate(self.angle) for part in self.parent.parts()]

def finger_joint(begin, end, length, width, samples_per_arc):
    """
    Creates a finger joint
    """
    
    
def find_plane_intersections(plane, perimeter, tolerance = 1e-16):
    """
    Points likely come out in linear chain for a perimeter of an intersection
    """
    points = []
    for v0, v1 in n_wise(perimeter,2):
        d0 = plane.distance(v0)
        d1 = plane.distance(v1)
        
        if (d0 < -tolerance and d1 < -tolerance) or (d0 > tolerance and d1 > tolerance):
            continue
            
        if abs(d1) < tolerance:
            points.append(v1)
            continue
            
        if abs(d0) < tolerance:
            if not points or np.linalg.norm(points[-1] - v0) > tolerance:
                points.append(v0)
            continue
            
        line = Line.between(v0, v1)
        t = line.intersect(plane)
        point = line.unmap(t)
        assert(not np.isnan(point).any())
        points.append(point)
        
    # handle perimeter wrap (e.g vertices[0] == vertices[-1])
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < tolerance:
        points.pop()
    
    return points
    

class Bulkhead():

    def __init__(self, plane, tolerance=1e-16):
        self.plane = plane
        self.segments = []
        self.tolerance = tolerance
        
    def add_panel(self, panel):
        for index, perimeter in enumerate(panel.perimeters):
            vertices = sequence(panel.vertices, perimeter)
            points = find_plane_intersections(self.plane, vertices)
            if index and points:
                logging.warning("Ignoring presumed hole!")
            if not points:
                continue
            
            if len(points) & 1:
                raise NotImplementedError(f"Unsupported odd incidence number: {len(points)}")
            
            # Determine point pairing (longest pairwise distance is invalid)
            normal_longest = 0
            for v0, v1 in n_wise(points, 2, stride=2):
                normal_longest = max(normal_longest, np.linalg.norm(v1 - v0))
                
            alternate = [points[-1]]
            alternate.extend(points[:-1])
            
            alternate_longest = 0
            for v0, v1 in n_wise(alternate, 2, stride=2):
                alternate_longest = max(alternate_longest, np.linalg.norm(v1 - v0))
            
            if alternate_longest < normal_longest:
                points = alternate
                
            for v0, v1 in n_wise(points, 2, stride=2):
                for mirror in panel.mirrors:
                    self.segments.append([v0 * mirror, v1 * mirror])
                                
    def perimeter(self):
        options = []
        combined = self.segments.pop()
        while self.segments:
            unmatched = []
            for segment in self.segments:
                if np.linalg.norm(segment[0] - combined[0]) < self.tolerance:
                    combined = list(reversed(combined))
                    combined.extend(segment[1:])
                    
                elif np.linalg.norm(segment[0] - combined[-1]) < self.tolerance:
                    combined.extend(segment[1:])
                    
                elif np.linalg.norm(segment[-1] - combined[0]) < self.tolerance:
                    segment.extend(combined[1:])
                    combined = segment
                    
                elif np.linalg.norm(segment[-1] - combined[-1]) < self.tolerance:
                    combined.extend(reversed(segment[:-1]))
                    
                else:
                    unmatched.append(segment)
            
            if len(unmatched) == len(self.segments):
                options.append(combined)
                combined = unmatched.pop()
                
            self.segments = unmatched
        
        options.append(combined)
        
        for option in options:
            if len(option) > len(combined):
                combined = option
                
        if len(options) > 1:
            logging.warning("Ignoring options: %s", options)
            
        return np.array(combined)
        
    def to_panel(self):
        # TODO: setback for material thickness
        # TODO: prune duplicate 0 or -1 element.
        vertices = self.perimeter()
        triangles = np.array([], dtype=np.int32)
        perimeter = np.array(list(range(len(vertices))), dtype=np.int32)
        points = np.array([self.plane.map(vertex, self.tolerance) for vertex in vertices])
        return Panel(vertices, triangles, [perimeter], 1, points)
            
class Features(enum.Enum):
    BOTTOM = enum.auto()
    SIDE = enum.auto()
    COAMING_BASE = enum.auto()
    RECESS = enum.auto()
    FRONT_DECK = enum.auto()
    COAMING_SPACER = enum.auto()
    COAMING_RIM = enum.auto()
    KNEE_DECK = enum.auto()
    BACK_DECK = enum.auto()
    COCKPIT_BACK = enum.auto()
    COCKPIT_FRONT = enum.auto()
    DAY_BULKHEAD = enum.auto()
    FRONT_BULKHEAD = enum.auto()
    BACK_BULKHEAD =  enum.auto()
    
body_features = (Features.BOTTOM, Features.SIDE, Features.COAMING_BASE, Features.RECESS, Features.FRONT_DECK, Features.KNEE_DECK, Features.BACK_DECK) 
    
class Model():
    ref_gunwale = np.array([
        [0.0, 0.0, 0.75],
        [3.0, 0.448, 0.603],    # knee panel start
        [4.5, 0.601, 0.551],    
        [6.0, 0.705, 0.512],    
        [8.25, 0.762, 0.487],   
        [10.05, 0.746, 0.481],  # cockpit back
        [11.233, 0.692, 0.491],
        [12.617, 0.591, 0.513],
        [14.0, 0.448, 0.548],
        [17.0, 0.0, 0.666]])
    
    ref_chine = np.array([
        [2.0, 0.0, 0.220],
        [3.0, 0.175, 0.175],
        [4.5, 0.387, 0.135],
        [6.0, 0.528, 0.108],
        [8.25, 0.603, 0.094],
        [10.05, 0.584, 0.097],
        [11.233, 0.517, 0.110],
        [12.617, 0.396, 0.133],
        [14.0, 0.229, 0.165],
        [15.5, 0.0, 0.21]])

    ref_keel = np.array([
        [2.0, 0.0, 0.220],
        [3.0, 0.0, 0.088],
        [4.5, 0.0, 0.044],
        [6.0, 0.0, 0.015],
        [8.25, 0.0, 0.0],
        [10.05, 0.0, 0.009],
        [11.233, 0.0, 0.025],
        [12.617, 0.0, 0.052],
        [14.0, 0.0, 0.089],
        [15.5, 0.0, 0.21]])
        
    #ref_front_deck_seam = np.concatenate((([0.0, 0.0, 0.75],), np.linspace([3.0, 0.448, 0.603], [7.55, 0.583, .833], 4)))
        #[0.0, 0.0, 0.75]
        #[3.0, 0.448, 0.603],    # 1
        #[4.5, 0.505, 0.677],    # 3
        #[6.0, 0.562, .75],      # 5
        #[7.55, 0.583, .833]])   # 7

    knee_panel_start = np.array([3.0, 0.448, 0.603])
    knee_panel_end = np.array([7.55, 0.583, .833])
    cockpit_back = np.array([10.05, 0.746, 0.481])

    mirror = np.array([1.0, -1.0, 1.0])
        
    def __init__(self, samples=3, tolerance=1e-6):
        self.features = {}
        
        # interpolate lines from reference model for improved output detail
        #
        self.keel = arc_interpolate(self.ref_keel, samples)
        self.chine = arc_interpolate(self.ref_chine, samples)
        self.gunwale = arc_interpolate(self.ref_gunwale, samples)
        #self.front_deck_seam = arc_interpolate(self.ref_front_deck_seam, samples) 
        
        # Find indices of key geometries for constructing panels
        #
        #knee_panel_start = self.knee_panel_start * (samples+1)
        #cockpit_back = self.cockpit_back * (samples+1)

        knee_panel_start = index_of(self.gunwale, self.knee_panel_start)
        cockpit_back = index_of(self.gunwale, self.cockpit_back)
        
        
        # Construct hull bottom
        #
        vertices = np.concatenate((self.chine, self.keel[1:-1]))
        triangles = [0]
        for index in range(1, len(self.chine) - 1):
            triangles.append(index)
            triangles.append(len(self.chine) + index - 1)
        triangles.append(len(self.chine)-1)
        
        perimeter = list(range(len(self.chine)))
        offset = len(perimeter)
        perimeter.extend(reversed(range(offset, offset + len(self.keel) - 2)))
        perimeter.append(0)
        
        self.bottom = Panel(vertices, np.array(triangles, dtype=np.int32), [np.array(perimeter, dtype=np.int32)], 2, mirrors=(self.mirror,))
        self.features[Features.BOTTOM] = self.bottom
        
        # Construct hull side
        #
        vertices = np.concatenate((self.gunwale, self.chine))
        indices = []
        for index in range(len(self.gunwale)):
            indices.append(index)
            indices.append(len(self.gunwale) + index)
            
        perimeter = list(range(len(self.chine)))
        offset = len(perimeter)
        perimeter.extend(reversed(range(offset, offset + len(self.gunwale))))
        perimeter.append(0)
        
        self.side = Panel(vertices, np.array(indices, dtype=np.int32), [np.array(perimeter, dtype=np.int32)], 2, mirrors=(self.mirror,))
        self.features[Features.SIDE] = self.side
        
        # Construct coaming base plate
        #
        coaming_peak = 10/12
        coaming_shape = np.array((29, 16, 1.0)) / 12
        coaming_setback = 1/12
        coaming_spacer_width = 0.20/12
        coaming_back_center = self.cockpit_back.copy()
        coaming_back_center[1] = 0
        coaming_back_center[2] -= coaming_shape[2]
        
        coaming_front_center = coaming_back_center.copy()
        coaming_front_center[0] -= coaming_shape[0]
        coaming_front_center[2] = coaming_peak
        
        coaming_base_line = Line.between(coaming_back_center, coaming_front_center)
        coaming_base_plane = Plane.from_lines(coaming_base_line, Line.axis(3, 1))

        recess_gunwale_crossing = coaming_base_line.solve(2, self.cockpit_back[2])
        recess_start_x = coaming_base_line.unmap(recess_gunwale_crossing)[0]
        for v0, v1 in zip(self.gunwale[:-1], self.gunwale[1:]):
            if v0[0] > recess_start_x or v1[0] < recess_start_x:
                continue
                
            line = Line.between(v0, v1)
            recess_start = Line.between(v0, v1).where(0, recess_start_x)
            break
        
        # Round back for recess
        coaming_base_back_shape = np.array((recess_gunwale_crossing + coaming_setback, recess_start[1] * 2))
        coaming_base_back_offset = np.array((-coaming_setback, 0))
        coaming_base_back_samples = len(self.gunwale) - cockpit_back # match back_deck_scallop_samples
        coaming_base_back = coaming_back_curve(coaming_base_plane, coaming_base_back_shape, coaming_base_back_samples, coaming_base_back_offset)
        
        coaming_cut_back_shape = np.array((recess_gunwale_crossing, coaming_shape[1]))
        coaming_cut_back_offset = np.zeros(2, dtype=np.double)
        coaming_cut_back_samples = coaming_base_back_samples # match for triangulation
        coaming_cut_back = coaming_back_curve(coaming_base_plane, coaming_cut_back_shape, coaming_cut_back_samples, coaming_cut_back_offset)
        
        # TODO thigh braces. 22" from back and 8-9" opening for knees. if we're going to round them, they should be wider! 2-3" deep after rounding
        coaming_cut_front_shape = np.array((coaming_shape[0] - recess_gunwale_crossing, coaming_shape[1]))
        coaming_cut_front_offset = np.array((recess_gunwale_crossing, 0))
        coaming_cut_front_samples = 2 * coaming_cut_back_samples # just a guess
        coaming_cut_front = coaming_front_curve(coaming_base_plane, coaming_cut_front_shape, coaming_cut_front_samples, coaming_cut_front_offset)
        
        coaming_cut_peak = coaming_base_plane.unmap((coaming_shape[0], 0))
                
        coaming_base_front =  coaming_base_line.unmap(coaming_shape[0] + coaming_setback)
        coaming_base_front[1] = self.knee_panel_end[1]
        
        external_vertices = np.concatenate((coaming_base_back, np.linspace(recess_start, coaming_base_front, coaming_cut_front_samples)))
        external_vertices = np.concatenate((external_vertices, np.flip(external_vertices, 0) * self.mirror))
        
        internal_vertices = np.concatenate((coaming_cut_back, coaming_cut_front))
        coaming_peak_index = len(internal_vertices)
        internal_vertices = np.concatenate((internal_vertices, (coaming_cut_peak,), np.flip(internal_vertices, 0) * self.mirror))
        
        assert(len(external_vertices) + 1 == len(internal_vertices))
        
        vertices = np.concatenate((external_vertices, internal_vertices))
        
        indices = []
        for index in range(len(external_vertices)):
            indices.append(len(external_vertices) + index)
            indices.append(index)
        
        # Patch in last triangle
        indices.append(len(vertices) - 1)
        indices.append(0)
        indices.append(len(external_vertices))
        
        external_perimeter = list(range(len(external_vertices)))
        external_perimeter.append(0)
        
        internal_perimeter = list(range(len(external_vertices), len(vertices)))
        internal_perimeter.append(len(external_vertices))
        perimeters = [ np.array(external_perimeter), np.array(internal_perimeter) ]

        self.features[Features.COAMING_BASE] = Panel(vertices, np.array(indices, dtype=np.int32), perimeters, 1)
        
        # Construct coaming rim and spacer
        #
        coaming_rim_plane = Plane(coaming_base_plane.uv, coaming_base_plane.offset + coaming_base_plane.normal * -coaming_shape[2], coaming_base_plane.normal)
        
        self.features[Features.COAMING_RIM] = coaming_ring(coaming_rim_plane, coaming_shape, coaming_base_back_samples, recess_gunwale_crossing, coaming_setback, 2)
        
        
        coaming_spacer_plane = Plane(coaming_base_plane.uv, coaming_base_plane.offset + coaming_base_plane.normal * -coaming_shape[2] * 0.5, coaming_base_plane.normal)
        
        self.features[Features.COAMING_SPACER] = coaming_ring(coaming_spacer_plane, coaming_shape, coaming_base_back_samples, recess_gunwale_crossing, coaming_spacer_width, 6)
         
        # Ensure gunwale and front deck match before knee panel, then mirror
        #
        for knee_gunwell_end, v in enumerate(self.gunwale):
            if v[0] > recess_start[0]:
                break
        knee_points = knee_gunwell_end - knee_panel_start + 2
        knee_line = np.linspace(self.knee_panel_start, coaming_base_front, knee_points)
        self.front_deck_seam = np.concatenate((self.gunwale[:knee_panel_start], knee_line))
        self.front_deck_seam_mirror = self.front_deck_seam * self.mirror
        
        # Construct front deck
        #
        vertices = np.concatenate((self.front_deck_seam, self.front_deck_seam_mirror))
        indices = [0]
        for index in range(1, len(self.front_deck_seam)):
            indices.append(index)
            indices.append(len(self.front_deck_seam) + index)
            
        perimeter = list(range(len(self.front_deck_seam)))
        offset = len(perimeter)
        perimeter.extend(reversed(range(offset, offset + len(self.front_deck_seam))))
        perimeter[-1] = 0
        
        self.front_deck = Panel(vertices, np.array(indices, dtype=np.int32), [np.array(perimeter, dtype=np.int32)], 1)
        self.features[Features.FRONT_DECK] = self.front_deck
        
        # Construct knee deck inbetween gunwale and front deck seam
        #
        vertices = np.concatenate((self.gunwale[knee_panel_start: knee_gunwell_end + 1], self.front_deck_seam[knee_panel_start + 1:]))
        vertices[knee_gunwell_end - knee_panel_start] = recess_start
        indices = [0]
        for index in range(1, knee_gunwell_end - knee_panel_start + 1):
            indices.append(index)
            indices.append(knee_gunwell_end - knee_panel_start + index)
            
        perimeter = list(range(knee_gunwell_end - knee_panel_start + 1))
        offset = len(perimeter)
        perimeter.extend(reversed(range(offset, len(vertices))))
        perimeter.append(0)
        
        self.knee_deck = Panel(vertices, np.array(indices, dtype=np.int32), [np.array(perimeter, dtype=np.int32)], 2, mirrors=(self.mirror,))
        self.features[Features.KNEE_DECK] = self.knee_deck
        
        # Back deck
        #
        self.gunwale_mirror = self.gunwale * self.mirror
        back_deck_center_line = Line.between(self.gunwale[cockpit_back] * (1, 0, 1), recess_start * (1, 0, 1))
        back_deck_center_line.offset = back_deck_center_line.where(0, coaming_back_center[0])
        back_deck_scallop_plane = Plane.from_lines(back_deck_center_line, Line.axis(3, 1))
        back_deck_scallop_shape = coaming_base_back_shape + (coaming_setback * 2, 0)
        back_deck_scallop_offset = coaming_base_back_offset - (coaming_setback * 2, 0)
        back_deck_scallop_samples = len(self.gunwale) - cockpit_back
        back_deck_scallop = coaming_back_curve(back_deck_scallop_plane, back_deck_scallop_shape, back_deck_scallop_samples, back_deck_scallop_offset)
        
        vertices = list(back_deck_scallop)
        vertices.append(recess_start.copy())
        vertices = np.concatenate((vertices, self.gunwale[cockpit_back:]))
        count = len(vertices)
        vertices = np.concatenate((vertices, vertices[:-1] * self.mirror))
        indices = [back_deck_scallop_samples]
        for index in reversed(range(back_deck_scallop_samples)):
            indices.append(index)
            indices.append(count - index - 1)
        
        for index in range(back_deck_scallop_samples):
            indices.append(count + index)
            indices.append(len(vertices) - index - 1)
            
        perimeter = list(range(count))
        offset = len(perimeter)
        perimeter.extend(reversed(range(offset, offset + count - 1)))
        perimeter.append(0)
        
        self.back_deck = Panel(vertices, np.array(indices, dtype=np.int32), [np.array(perimeter, dtype=np.int32)], 1)
        self.features[Features.BACK_DECK] = self.back_deck
        
        # Coaming recess
        #
        assert(coaming_base_back_samples == back_deck_scallop_samples)
        vertices = list(coaming_base_back)
        vertices.append(recess_start)
        vertices.extend(reversed(back_deck_scallop))
        count = len(vertices)
        vertices = np.concatenate((vertices, vertices * self.mirror))
        
        indices = [len(coaming_base_back)]
        
        for index in reversed(range(len(coaming_base_back))):
            indices.append(index)
            indices.append(count - index - 1)
            
        for index in range(len(coaming_base_back)):
            indices.append(count + index)
            indices.append(len(vertices) - index - 1)
            
        indices.append(count + len(coaming_base_back))
            
        perimeter = list(range(count))
        offset = len(perimeter)
        perimeter.extend(reversed(range(offset, offset + count)))
        perimeter.append(0)
        
        recess_panel = Panel(vertices, np.array(indices, dtype=np.int32), [np.array(perimeter, dtype=np.int32)], 1)
        self.features[Features.RECESS] = recess_panel
        
        # Bulkheads
        #
        # TODO: tilt
        cockpit_back_offset = coaming_back_center.copy()
        cockpit_back_plane = Plane.from_lines(Line.axis(3, 1), Line.axis(3, 2), cockpit_back_offset)
        cockpit_front_offset = cockpit_back_offset - np.array((48/12, 0, 0))
        cockpit_front_plane = Plane.from_lines(Line.axis(3, 1), Line.axis(3, 2), cockpit_front_offset)
        day_bulkhead_offset = cockpit_back_offset + np.array((12/12, 0, 0))
        day_bulkhead_plane = Plane.from_lines(Line.axis(3, 1), Line.axis(3, 2), day_bulkhead_offset)
        front_bulkhead_offset = cockpit_front_offset - np.array((36/12, 0, 0))
        front_bulkhead_plane = Plane.from_lines(Line.axis(3, 1), Line.axis(3, 2), front_bulkhead_offset)
        back_bulkhead_offset = day_bulkhead_offset + np.array((36/12, 0, 0))
        back_bulkhead_plane = Plane.from_lines(Line.axis(3, 1), Line.axis(3, 2), back_bulkhead_offset)
        
        self.bulkheads = {
            Features.BACK_BULKHEAD: Bulkhead(back_bulkhead_plane, tolerance),
            Features.FRONT_BULKHEAD: Bulkhead(front_bulkhead_plane, tolerance),
            Features.DAY_BULKHEAD: Bulkhead(day_bulkhead_plane, tolerance),
            Features.COCKPIT_BACK: Bulkhead(cockpit_back_plane, tolerance),
            Features.COCKPIT_FRONT: Bulkhead(cockpit_front_plane, tolerance)
        }
        
        for feature in body_features:
            for bulkhead in self.bulkheads.values():
                bulkhead.add_panel(self.features[feature])
        
           
        for feature, bulkhead in self.bulkheads.items():
            self.features[feature] = bulkhead.to_panel()

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
if __name__ == "__main__":
        
    a = Arc(np.array((1, 0, 0)), np.array((0, 1, 0)), np.array((0, -1, 0)))
    a = Arc(np.array((-1, 0, 0)), np.array((0, 1, 0)), np.array((0, -1, 0))) 
    a = Arc(np.array((2, 0, 0)), np.array((1, 1, 0)), np.array((0, 0, 0))) 
    a = Arc(np.array((10, 0, 0)), np.array((6, 4, 0)), np.array((2, 0, 0)))
    parser = argparse.ArgumentParser("Flatten geometry for tortured plywood construction")
    parser.add_argument("-l", "--log-level", default="warn", choices=("warn", "info", "error", "critical", "debug"), help="Log level")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging mode")
    parser.add_argument("-t", "--tolerance", type=float, default=0.01, help="Max error tolerance")
    parser.add_argument("-i", "--interpolate", type=int, default=0, help="Number of segments to interpolate")
    parser.add_argument("-p", "--points", action="store_true", help ="Graph points")
    parser.add_argument("-v", "--vertices", action="store_true", help ="Graph vertices")
    parser.add_argument("-m", "--mirror", action="store_true", help ="Graph both sides")
    parser.add_argument("-f", "--feature", action="append", choices=[feature.name.lower() for feature in Features], help="Feature(s) to modify")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level)
    
    model = Model(args.interpolate, args.tolerance)
    
    ax = plt.figure().add_subplot(projection='3d')
    
    if args.feature:
        features = (Features[feature.upper()] for feature in args.feature)
    else:
        features = tuple(Features)
    
    #ax = plt.gca()
    #ax.set_aspect('equal', adjustable='box')
    area = 0.0
    for feature in features:
        panel = model.features[feature]
        if panel.points is None:
            panel.flatten(args.tolerance)
        if args.vertices:
            panel.plot_vertices(ax, z=True)
            if args.mirror and panel.quantity == 2:
                panel.plot_vertices(ax, z=True, scale=model.mirror)
        if args.points:
            panel.plot_points(ax, z=True)
            if args.mirror and panel.quantity == 2:
                panel.plot_points(ax, z=True, scale=model.mirror)
        #area += panel.area() * panel.quantity
        
    #bulkhead = np.concatenate(model.cockpit_back.segments)
    #ax.scatter(bulkhead[:, 0], bulkhead[:, 1], bulkhead[:, 2])
    
    logging.info("Area: %f", area)
    set_axes_equal(ax)
    plt.show()
    
