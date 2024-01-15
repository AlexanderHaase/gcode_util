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

from abc import ABC


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
                point = point_1 + point_0
                point *= 0.5
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
    
    __slots__ = ('vertices', 'triangles', 'perimeter', 'points', 'quantity')
    
    def __init__(self, vertices, triangles, perimeter, quantity=1, points=None):
        self.vertices = vertices
        self.triangles = triangles
        self.perimeter = perimeter
        self.quantity = quantity
        self.points = points
        
    def flatten(self, tolerance, offset=None, mirror=True):
        triangle_strip = sequence(self.vertices, self.triangles)
        points, total_error, max_error = flatten(triangle_strip, tolerance=tolerance)
        logging.info("Total error: %f, max error: %f", total_error, max_error)
        points = reorient(points[0], points[-1] + points[-2], points)
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
            z_guess = -np.mean(self.vertices[:, 2])
            x_guess = -abs(np.mean(self.vertices[:, 1]))
            offset = np.array((x_guess, 0, z_guess))
            
        points_2d = sequence(self.points, self.triangles) * scale + offset[:2]
        perim_2d = sequence(self.points, self.perimeter) * scale + offset[:2]
        if z:
            ax.plot(points_2d[:, 0], points_2d[:, 1], np.full(np.shape(points_2d)[0], offset[2]))
            ax.plot(perim_2d[:, 0], perim_2d[:, 1], np.full(np.shape(perim_2d)[0], offset[2]))
        else:
            ax.plot(points_2d[:, 0], points_2d[:, 1])
            ax.plot(perim_2d[:, 0], perim_2d[:, 1])
        
    
    def plot_vertices(self, ax, z=False, scale=1):
        triangle_strip = sequence(self.vertices, self.triangles) * scale
        perim_3d = sequence(self.vertices, self.perimeter) * scale
        if z:
            ax.plot(triangle_strip[:, 0], triangle_strip[:, 1], triangle_strip[:, 2])
            ax.plot(perim_3d[:, 0], perim_3d[:, 1], perim_3d[:, 2])
        else:
            ax.plot(triangle_strip[:, 0], triangle_strip[:, 1])
            ax.plot(perim_3d[:, 0], perim_3d[:, 1])
            
class Features(enum.Enum):
    BOTTOM = enum.auto()
    SIDE = enum.auto()
    FRONT_DECK = enum.auto()
    KNEE_DECK = enum.auto()
    BACK_DECK = enum.auto()
    COAMING_BASE = enum.auto()

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
        
    def __init__(self, samples=3):
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
        
        # Ensure gunwale and front deck match before knee panel, then mirror
        #
        knee_line = np.linspace(self.knee_panel_start, self.knee_panel_end, 5 + 4 * samples)
        self.front_deck_seam = np.concatenate((self.gunwale[:knee_panel_start], knee_line))
        self.front_deck_seam_mirror = self.front_deck_seam * self.mirror
        
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
        
        self.bottom = Panel(vertices, np.array(triangles, dtype=np.int32), np.array(perimeter, dtype=np.int32), 2)
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
        
        self.side = Panel(vertices, np.array(indices, dtype=np.int32), np.array(perimeter, dtype=np.int32), 2)
        self.features[Features.SIDE] = self.side
        
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
        
        self.front_deck = Panel(vertices, np.array(indices, dtype=np.int32), np.array(perimeter, dtype=np.int32), 1)
        self.features[Features.FRONT_DECK] = self.front_deck
        
        # Construct coaming base plate
        #
        base_back_center = self.cockpit_back.copy()
        
        vertices = np.array((self.knee_panel_end, self.cockpit_back))
        vertices = np.concatenate((vertices, self.mirror*vertices))
        indices = [0, 1, 2, 3]
        perimeter = [0, 1, 3, 2, 0]
        self.features[Features.COAMING_BASE] = Panel(vertices, np.array(indices, dtype=np.int32), np.array(perimeter, dtype=np.int32), 1)
        
        # Construct knee deck inbetween gunwale and front deck seam
        #
        #vertices = np.concatenate((self.front_deck_seam[knee_panel_start:], self.gunwale[knee_panel_start + 1: cockpit_back + 1]))
        vertices = np.concatenate((self.gunwale[knee_panel_start: cockpit_back + 1], self.front_deck_seam[knee_panel_start + 1:]))
        indices = [0]
        for index in range(1, cockpit_back - knee_panel_start + 1):
            indices.append(index)
            indices.append(cockpit_back - knee_panel_start + index)
            
        perimeter = list(range(cockpit_back - knee_panel_start + 1))
        offset = len(perimeter)
        perimeter.extend(reversed(range(offset, offset + len(self.front_deck_seam) - knee_panel_start - 1)))
        perimeter.append(0)
        
        self.knee_deck = Panel(vertices, np.array(indices, dtype=np.int32), np.array(perimeter, dtype=np.int32), 2)
        self.features[Features.KNEE_DECK] = self.knee_deck
        
        # Back deck
        #
        # Construct front deck
        self.gunwale_mirror = self.gunwale * self.mirror
        vertices = np.concatenate((self.gunwale[cockpit_back:], self.gunwale_mirror[cockpit_back:-1]))
        indices = []
        for index in range(len(self.gunwale) - cockpit_back - 1):
            indices.append(index)
            indices.append(len(self.gunwale) - cockpit_back + index)
        indices.append(len(self.gunwale) - cockpit_back - 1)
            
        perimeter = list(range(len(self.gunwale) - cockpit_back))
        offset = len(perimeter)
        perimeter.extend(reversed(range(offset, offset + len(self.gunwale) - cockpit_back - 1)))
        perimeter.append(0)
        
        self.back_deck = Panel(vertices, np.array(indices, dtype=np.int32), np.array(perimeter, dtype=np.int32), 1)
        self.features[Features.BACK_DECK] = self.back_deck

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
    
    model = Model(args.interpolate)
    #triangle_strip = sequence(model.front_deck.vertices, model.front_deck.triangles)
    #points, total_error, max_error = flatten(triangle_strip, tolerance=args.tolerance)
    #logging.info("Total error: %f, max error: %f", total_error, max_error)
    
    #points = reorient(points[0], points[-1] + points[-2], points)
    #points *= np.array((1.0, -1.0))
    #points += np.array((0.0, 2.0))
    
    ax = plt.figure().add_subplot(projection='3d')
    
    #ax.plot(points[:, 0], points[:, 1], np.zeros(np.shape(points)[0]))
    #ax.plot(triangle_strip[:, 0], triangle_strip[:, 1], triangle_strip[:, 2])
    
    #perim_2d = sequence(points, project(model.front_deck.perimeter, model.front_deck.triangles))
    #ax.plot(perim_2d[:, 0], perim_2d[:, 1], np.zeros(np.shape(perim_2d)[0]))
    
    #perim_3d = sequence(model.front_deck.vertices, model.front_deck.perimeter)
    #ax.plot(perim_3d[:, 0], perim_3d[:, 1], perim_3d[:, 2])
    
    if args.feature:
        features = (Features[feature.upper()] for feature in args.feature)
    else:
        features = tuple(Features)
    
    #ax = plt.gca()
    #ax.set_aspect('equal', adjustable='box')
    area = 0.0
    for feature in features:
        panel = model.features[feature]
        panel.flatten(args.tolerance)
        if args.vertices:
            panel.plot_vertices(ax, z=True)
            if args.mirror and panel.quantity == 2:
                panel.plot_vertices(ax, z=True, scale=model.mirror)
        if args.points:
            panel.plot_points(ax, z=True)
            if args.mirror and panel.quantity == 2:
                panel.plot_points(ax, z=True, scale=model.mirror)
        area += panel.area() * panel.quantity
        
    logging.info("Area: %f", area)
    set_axes_equal(ax)
    plt.show()
    
