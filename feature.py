#!/usr/bin/env python3

import logging
import itertools
import numpy as np
import math
import abc

from geometry import *
    
class Panel():
    '''
    Single panel to be developed
    '''

    __slots__ = ('vertices', 'triangles', 'perimeters', 'points', 'quantity', 'mirrors', 'align_axis')
    
    def __init__(self, vertices, triangles, perimeters, quantity=1, points=None, mirrors=tuple(), align_axis=tuple()):
        self.vertices = vertices
        self.triangles = triangles
        self.perimeters = perimeters
        self.quantity = quantity
        self.points = points
        self.mirrors = [np.ones(3)]
        self.mirrors.extend(mirrors)
        self.align_axis = align_axis
        
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
        # TODO: figure out bulkhead triangulation
        #
        if not len(self.triangles):
            return 0 
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
            
        if len(self.triangles):
            points_2d = sequence(self.points, self.triangles) * scale + offset[:2]
        
        perimeters = [sequence(self.points, perimeter) * scale  + offset[:2] for perimeter in self.perimeters]
        if z:
            if len(self.triangles):
                ax.plot(points_2d[:, 0], points_2d[:, 1], np.full(np.shape(points_2d)[0], offset[2]))
            for perimeter in perimeters:
                ax.plot(perimeter[:, 0], perimeter[:, 1], np.full(np.shape(perimeter)[0], offset[2]))
        else:
            if len(self.triangles):
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
        
        align_axis = np.array([index_of(vertices, coaming_inner_peak), index_of(vertices, coaming_outer_peak)], dtype=np.int32)

        return Panel(vertices, np.array(indices, dtype=np.int32), perimeters, qty, align_axis=align_axis)


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

    def __init__(self, plane, setback=0, tolerance=1e-10):
        self.plane = plane
        self.segments = []
        self.setback = setback
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
                    self.segments.append(np.array((v0 * mirror, v1 * mirror)))
                                
    def solve(self):
        # Combine all coincident line segments into "options"
        #
        options = []
        combined = self.segments.pop()
        while self.segments:
            unmatched = []
            for segment in self.segments:
                if np.linalg.norm(segment[0] - combined[0]) < self.tolerance:
                    combined = np.concatenate((np.flip(combined, 0), segment[1:]))
                    
                elif np.linalg.norm(segment[0] - combined[-1]) < self.tolerance:
                    combined = np.concatenate((combined, segment[1:]))
                    
                elif np.linalg.norm(segment[-1] - combined[0]) < self.tolerance:
                    combined = np.concatenate((segment, combined[1:]))
                    
                elif np.linalg.norm(segment[-1] - combined[-1]) < self.tolerance:
                    combined = np.concatenate((combined, np.flip(segment[:-1], 0)))
                    
                else:
                    unmatched.append(segment)
            
            if len(unmatched) == len(self.segments):
                # Reject non-cyclic line segments
                #
                if np.linalg.norm(combined[0] - combined[-1]) > self.tolerance:
                    logging.debug("Discarding incomplete bulkhead option:\n%s", combined)
                else:
                    options.append(combined)
                
                combined = unmatched.pop()
                
            self.segments = unmatched
            

        if np.linalg.norm(combined[0] - combined[-1]) > self.tolerance:
            # Reject non-cyclic line segments
            #
            logging.debug("Discarding incomplete bulkhead option:\n%s", combined)
            combined = options.pop()
        
        # Accept the most complex perimeter as presumably the best (but warn the humans)
        #
        for option in options:
            if len(option) > len(combined):
                logging.info("Ignoring bulkhead option:\n%s", combined)
                combined = option
            else:
                logging.info("Ignoring bulkhead option:\n%s", option)
            
        self.vertices = combined[:-1]
        
        
    def sort_ccw(self, axis=2):
        '''
        Re-order the vertices in ccw order, setup for graham's scan convex hull
        '''
        stack = []
        
        # Scan for the min_x point and determine the ordering of the vertices relative to the plane
        #
        v0 = self.vertices[-2]
        v1 = self.vertices[-1]
        
        min_value = self.vertices[0]
        min_index = 0
        
        angle = 0
        
        for index, v2 in enumerate(self.vertices):
            if v2[axis] < min_value[axis]:
                min_value = v2
                min_index = index
                
            corner = self.Corner(v0, v1, v2, self.plane.normal)
            angle += corner.angle()
            
            stack.append(corner)
            
            v0 = v1
            v1 = v2
            
        # Reverse order if needed
        #
        if angle < 0:
            self.vertices = np.flip(self.vertices, 0)
            stack.reverse()
            min_index = len(self.vertices) - 1 - min_index
 
        # Reorder points from min_x
        #
        if min_index > 0:
            self.vertices = np.concatenate((self.vertices[min_index:], self.vertices[:min_index]))
            new_stack = list(stack[min_index:])
            new_stack.extend(stack[:min_index])
            stack = new_stack
        
        return stack
        
    class Corner():
        __slots__ = ('v0', 'v1', 'v2', 'l10', 'l12', 'cp', 'mag')
        
        def __init__(self, v0, v1, v2, normal):
            self.v0 = v0
            self.v1 = v1
            self.v2 = v2
            self.l10 = v0 - v1
            self.l12 = v2 - v1
            self.cp = np.cross(self.l10, self.l12)
            self.mag = np.dot(self.cp, normal)
         
        def angle(self):
            d10 = np.linalg.norm(self.l10)
            d12 = np.linalg.norm(self.l12)
          
            return np.arcsin(self.mag/(d10 * d12))
        
    def convex_hull(self):
        '''
        Graham's scan, given that we already have sorted points and clockwise
        is sign(cross product * normal).
        TODO: move this into a more generic scope
        '''
        raise NotImplementedError
        hull = []
        
        v0 = self.vertices[-1]
        v1 = self.vertices[0]        
        for v2 in self.vertices[1:]:
            l10 = v0 - v1
            l12 = v2 - v1
            cp = np.cross(l10, l12)
            direction = np.dot(cp, self.plane.normal)
            ccw = direction <= 0 # Allow for straight lines
            
            if ccw:
                hull.append(v0)
                v0 = v1
            v1 = v2
            
        
    def offset(self):
        '''
        move line segments perpendicularly toward inside, then recover vertices from intersections
        Assumes vertices are in CCW order relative to plane.
        '''
        
        lines = []
        
        # offset lines
        #
        v0 = self.vertices[-1]
        for v1 in self.vertices:
            line = Line.between(v0, v1)
            normal = np.cross(line.unit, self.plane.normal)
            line.offset += normal * self.setback
            lines.append(line)
            v0 = v1
            
        vertices = []
        
        # recover vertices
        #
        l0 = lines[-1]
        for l1 in lines:
            d0, d1 = l0.nearest(l1)
            separation = np.linalg.norm(l0.unmap(d0) - l1.unmap(d1))
            assert(separation < self.tolerance)
            vertices.append(l0.unmap(d0))
            l0 = l1
            
        self.vertices = np.array(vertices)
        
    def to_panel(self):
        # TODO: setback for material thickness
        self.solve()
        self.sort_ccw()
        self.offset()
        vertices = self.vertices
        triangles = np.array([], dtype=np.int32)
        perimeter = list(range(len(vertices)))
        perimeter.append(0)
        perimeter = np.array(perimeter, dtype=np.int32)
        points = np.array([self.plane.map(vertex, self.tolerance) for vertex in self.vertices])
        return Panel(self.vertices, triangles, [perimeter], 1, points)
            
    
