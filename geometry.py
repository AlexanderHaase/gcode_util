#!/usr/bin/env python3

from collections import OrderedDict
import logging
import numpy as np
import math

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
        #
        # Minimize ||t1*U1 + O1 - t2*U2 - O2|| wrt t0, t1
        # F(t1, t2) = (t1*u1x - t2*u2x + cx)**2 + ...
        # dF/dt1 = 2*u1x*(t1*u1x - t2*u2x + cx) + ... = 0
        # 0 = 2*(t1*dot(U1, U1) - t2*dot(U1, U2) + dot(C, U1))
        #
        # ==> t1 = (t2*dot(U1, U2) - dot(C, U1)) / dot(U1, U1)
        #
        # dF/dt2 = -2*u2x*(t1*u1x - t2*u2x + cx) + ... = 0
        # 0 = t2*dot(U2, U2) -  t1*dot(U1, U2) - dot(C, U2)
        #
        # substitute t1
        #
        # 0 = t2*dot(U2, U2) -  (t2*dot(U1, U2) - dot(C, U1))*dot(U1, U2)/dot(U1, U1) - dot(C, U2)
        # 0 = t2*dot(U2, U2)*dot(U1, U1) - t2*dot(U1, U2)**2 + dot(C, U1)*dot(U1, U2) - dot(C, U2)*dot(U1, U1)
        # 0 = t2*(dot(U2, U2)*dot(U1, U1) - dot(U1, U2)**2) + dot(C, U1)*dot(U1, U2) - dot(C, U2)*dot(U1, U1)
        # t2*(dot(U2, U2)*dot(U1, U1) - dot(U1, U2)**2) = - dot(C, U1)*dot(U1, U2) + dot(C, U2)*dot(U1, U1)
        # t2 = (dot(C, U2)*dot(U1, U1) - dot(C, U1)*dot(U1, U2)) / (dot(U2, U2)*dot(U1, U1) - dot(U1, U2)**2)
        #
        # Solve for t1
        #
        # t1 = (t2*dot(U1, U2) - dot(C, U1)) / dot(U1, U1)
        # t1*dot(U1, U1) + dot(C, U1) = t2*dot(U1, U2)
        # t1*dot(U1, U1) + dot(C, U1) = dot(U1, U2) * (dot(C, U2)*dot(U1, U1) - dot(C, U1)*dot(U1, U2)) / (dot(U2, U2)*dot(U1, U1) - dot(U1, U2)**2)
        # t1*dot(U1, U1) + dot(C, U1) = (dot(C, U2)*dot(U1, U1)*dot(U1, U2) - dot(C, U1)*dot(U1, U2)**2) / (dot(U2, U2)*dot(U1, U1) - dot(U1, U2)**2)
        # t1*dot(U1, U1) = (dot(C, U2)*dot(U1, U1)*dot(U1, U2) - dot(C, U1)*dot(U1, U2)**2 - dot(C, U1)*(dot(U2, U2)*dot(U1, U1) - dot(U1, U2)**2)) / (dot(U2, U2)*dot(U1, U1) - dot(U1, U2)**2)
        # t1*dot(U1, U1) = (dot(C, U2)*dot(U1, U1)*dot(U1, U2) - dot(C, U1)*dot(U2, U2)*dot(U1, U1)) / (dot(U2, U2)*dot(U1, U1) - dot(U1, U2)**2)
        # t1 = (dot(C, U2)*dot(U1, U2) - dot(C, U1)*dot(U2, U2)) / (dot(U2, U2)*dot(U1, U1) - dot(U1, U2)**2)
        
        c = self.offset - other.offset
        u1 = self.unit * self.scale
        u2 = other.unit * other.scale
        u1u1 = np.dot(u1, u1)
        u1u2 = np.dot(u1, u2)
        u2u2 = np.dot(u2, u2)
        u1c = np.dot(u1, c)
        u2c = np.dot(u2, c)
        
        t2 = (u2c*u1u1 - u1c*u1u2)/(u2u2*u1u1 - u1u2**2)
        t1 = (t2*u1u2 - u1c)/u1u1
        
        return (t1, t2)
    
class Plane():

    __slots__ = ("uv", "offset", "normal")
    
    def __init__(self, uv, offset, normal):
        self.uv = uv
        self.offset = offset
        self.normal = normal / np.linalg.norm(normal)
        
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
    
