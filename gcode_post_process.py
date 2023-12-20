#!/usr/bin/env python3

from collections import OrderedDict
import logging
import argparse
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import datetime
import functools
from abc import ABC

class Operation():
    '''
    Basic class for parsing/serializing g-code expressions. Mostly focuesd on G0-G4.
    '''
    parse_map = OrderedDict([
        ("G", int),
        ("M", int),
        ("X", float),
        ("Y", float),
        ("Z", float),
        ("I", float),
        ("J", float),
        ("K", float),
        ("F", float)
    ])
    __slots__ = tuple(itertools.chain(parse_map.keys(), ("line", "source")))
    
    def __init__(self, line=None, source=None):
        self.line = line
        self.source = source
        
    @classmethod
    def parse(cls, line, source):
        result = cls(line, source)
        for token in line.split():
            try:
                key = token[0]
                parser = cls.parse_map[key]
                value = parser(token[1:])
                setattr(result, key, value)
            except KeyError:
                raise ValueError(f"Unrecognized gcode token: '{token}'")
        return result
        
    def distance_squared(self, other, keys=("X", "Y")):
        return sum(map(lambda key: pow(getattr(self, key) - getattr(other, key), 2), keys))

    def __str__(self):
        tokens = [ f"{attr}{getattr(self, attr)}" for attr in self.parse_map.keys() if hasattr(self, attr) ]
        return " ".join(tokens)
        
    def __repr__(self):
        return f"{self.source}: {str(self)}"
        

def optimize_step_over(lines, radius, offset=0):
    '''
    Removes step-overs to identical Z-hieght within the specified radius of travel.
    '''
    last_op = None
    pending = []
    threshold = pow(radius, 2)
    
    count = 0
    
    for index, line in enumerate(lines):
        if not line.startswith("G"):
            yield line
            continue
        
        line_number = offset + index    
        op = Operation.parse(line, line_number)
        if op.G == 0:
            pending.append(line)
        elif op.G > 0 and op.G <= 4:
            if pending:
                if last_op is None or (op.distance_squared(last_op) > threshold and last_op.Z == op.Z):
                    yield from pending
                else:
                    logging.debug("Optimized out step-over in lines %d-%d: '%s'", line_number - len(pending), line_number, pending)
                    count += 1
                    
                pending.clear()
            last_op = op
            yield line
        else:
            yield from pending
            pending.clear()
            yield line
            
    logging.info("Optimized out %d step-overs with distance less than %f.", count, radius)
            

def get_preamble_range(lines):
    '''
    Try to find the preamble for the g-code that might need to be replicated per file
    '''
    for index, line in enumerate(lines):
        if not line.startswith("G0 "):
            continue
        if not line.startswith("G0 Z"):
            raise ValueError("First position command didn't set Z!!! We don't know how to handle this safely!!!")
            
        return (0, index)
        
    raise ValueError("No preamble detected!")
    
def get_postamble_range(lines):
    '''
    Try to find the postamble for the g-code that might need to be replicated per file
    '''
    for index, line in enumerate(lines):
        if not line.startswith("M"):
            continue
            
        return (index, len(lines))
        
    raise ValueError("No postamble detected!")

class GcodeFile():

    def __init__(self, preamble, lines, postamble):
        self.preamble = preamble
        self.lines = lines
        self.postamble = postamble
        
    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            for lines in (self.preamble, self.lines, self.postamble):
                handle.writelines(lines)
    
    @staticmethod
    def get_preamble_range(lines):
        '''
        Try to find the preamble for the g-code that might need to be replicated per file
        '''
        for index, line in enumerate(lines):
            if not line.startswith("G0 "):
                continue
            if not line.startswith("G0 Z"):
                raise ValueError("First position command didn't set Z!!! We don't know how to handle this safely!!!")
                
            return (0, index)
            
        raise ValueError("No preamble detected!")
        
    @staticmethod
    def get_postamble_range(lines):
        '''
        Try to find the postamble for the g-code that might need to be replicated per file
        '''
        for index, line in enumerate(lines):
            if not line.startswith("M"):
                continue
                
            return (index, len(lines))
            
        raise ValueError("No postamble detected!")
        
    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
            
        preamble_range = cls.get_preamble_range(lines)
        preamble = lines[preamble_range[0]: preamble_range[1]]
        logging.debug("Preamble located on lines %d-%d", *preamble_range)
        
        postamble_range = cls.get_postamble_range(lines)
        postamble = lines[postamble_range[0]: postamble_range[1]]
        logging.debug("Postamble located on lines %d-%d", *postamble_range)
        
        return cls(preamble, lines[preamble_range[1]:postamble_range[0]], postamble)
            
    
def split_z_layers(gcode: GcodeFile, z_thresh=0):
    '''
    Splits a large g-code job into multiple jobs by z-height.
    '''
    layers = []
    
    safe_travel_height = gcode.lines[0] # TODO: Cross-check
    
    start = 0
    current_z = None
    
    for index, line in enumerate(gcode.lines):
        if line.strip().startswith("("):
            continue
        
        line_number = index + len(gcode.preamble)
        op = Operation.parse(line, line_number)
        if hasattr(op, "Z"):
            if op.Z >= z_thresh:
                pass
            elif current_z is None:
                current_z = op.Z
            elif op.Z < current_z:
                logging.debug("Layer created from lines %d-%d at z-height %f.", len(gcode.preamble) + start, len(gcode.preamble) + index, current_z)
                layer = []
                if len(layers):
                    move_to_start = Operation.parse(gcode.lines[start], start + len(gcode.preamble))
                    del move_to_start.Z
                    layer.append(safe_travel_height)
                    layer.append(f"{str(move_to_start)}\n")
                layer.extend(gcode.lines[start:index])
                
                layers.append(GcodeFile(gcode.preamble, layer, gcode.postamble))
                start = index
                current_z = None
            else:
                pass
                
    if start < len(gcode.lines): 
        logging.debug("Layer created from lines %d-%d at z-height %f.", len(gcode.preamble) + start, len(gcode.preamble) + len(gcode.lines), current_z)
        layer = []
        if len(layers):
            move_to_start = Operation.parse(gcode.lines[start], start + len(gcode.preamble))
            del move_to_start.Z
            layer.append(safe_travel_height)
            layer.append(f"{str(move_to_start)}\n")
        layer.extend(gcode.lines[start:])
        
        layers.append(GcodeFile(gcode.preamble, layer, gcode.postamble))
    
    logging.info("Identifed %d layers", len(layers))
    return layers
            
        
def walk_arc(begin, end, center, step, reverse=False):
    '''
    Walks an arc from beginning to end with at most step-length intervals.
    Always walks CCW.
    '''
    p0 = begin - center
    p1 = end - center
    
    r = np.linalg.norm(p0[:2])
    r_check = np.linalg.norm(p1[:2])
    
    if np.absolute(r - r_check) > step:
        logging.warn("Bad circle center %s for %s %s leads to different radii %f %f", center, begin, end, r, r_check)
    
    a0 = np.arctan2(p0[1], p0[0])
    a1 = np.arctan2(p1[1], p1[0])
    
    angular_distance = a1 - a0
        
    if angular_distance < 0:
        angular_distance += 2 * np.pi
        
    samples = int(np.ceil(r * angular_distance / step))
    
    if reverse:
        angle_start = a1
        angle_step = -angular_distance / samples
        z_start = end[2]
        z_step = (begin[2] - end[2]) / samples
        last = begin
        first = end
    else:
        angle_start = a0
        angle_step = angular_distance / samples
        z_start = begin[2]
        z_step = (end[2] - begin[2])
        last = end
        first = begin
    
    yield first.copy()
    
    for index in range(1, samples):
        angle = angle_start + angle_step * index
        z = z_start + z_step * index
        yield np.array((center[0] + np.cos(angle) * r, center[1] + np.sin(angle) *  r, z))
        
    yield last.copy()
    
def arc_length(begin, end, center):
    '''
    Compute the length of the arc
    '''
    p0 = begin - center
    p1 = end - center
    
    r = np.linalg.norm(p0[:2])
    
    a0 = np.arctan2(p0[1], p0[0])
    a1 = np.arctan2(p1[1], p1[0])
    
    angular_distance = a1 - a0
    
    if angular_distance < 0:
        angular_distance += 2 * np.pi
    
    xy = angular_distance * r
    z = end[2] - begin[2]
    
    return np.sqrt(pow(xy, 2) + pow(z, 2))
    
def arc_bounds(begin, end, center):
    '''
    Compute the XYZ bounding box of the arc
    '''
    p0 = begin - center
    p1 = end - center
    
    r = np.linalg.norm(p0[:2])
    d = np.linalg.norm(end - begin)
    
    a0 = np.arctan2(p0[1], p0[0])
    a1 = np.arctan2(p1[1], p1[0])
    
    angular_distance = a1 - a0
    
    if a0 < 0:
        a0 += 2 * np.pi
    
    if a1 < a0:
        a1 += 2 * np.pi
        
    min_bound, max_bound = line_bounds(begin, end)
    
    targets = ((0, r, 0), (np.pi / 2, 0, r), (np.pi, -r, 0), (np.pi * 3 / 2, 0, -r))
    for angle, x, y in targets:
        if angle < a0:
            angle += 2 * np.pi
    
        if angle <= a1:
            point = np.array((center[0] + x, center[1] + y, begin[2]))
            min_bound = np.minimum(point, min_bound)
            max_bound = np.maximum(point, max_bound)
    
    
    # TODO: np.arctan2 is only so precise :/
    #
    tolerance = min(r, d) * 0.0001
    margin = np.array((tolerance, tolerance, tolerance))
    
    return (min_bound - margin, max_bound + margin)
        
def walk_line(begin, end, step):
    '''
    Walks a line from beginning to end with at most step-length intervals.
    '''
    pointer = end - begin
    length = np.linalg.norm(pointer)
    
    samples = int(np.ceil(length / step)) + 1
    
    increment = pointer / samples
    
    current = begin + increment
    
    yield begin.copy()
    
    for ignored in range(1, samples):
        after = current + increment
        yield current
        current = after
        
    yield end.copy()
    
def line_length(begin, end):
    '''
    Compute the length of the arc
    '''
    return np.linalg.norm(end - begin)
    
    
def line_bounds(begin, end):
    '''
    Compute the XYZ bounding box of the line
    '''
    return (np.minimum(begin, end), np.maximum(begin, end))
        
class Path():
    '''
    Captures the path taken by an operation, making implicit state explicit.
    
    Steps progress linearly from beginning to end, so that the path may be
    shortened or split in the inituitive way.
    '''
    __slots__: ("op", "begin", "end", "step", "keep")
    
    def __init__(self, op, begin, end, step, keep=False):
        self.begin = begin.copy()
        self.end = end.copy()
        self.op = op
        self.step = step
        self.keep = keep
        
    def arc_center(self):
        '''
        Applies only for G2 and G3 codes.
        '''
        return self.begin + np.array((self.op.I, self.op.J, self.op.K))
           
    def walk(self):
        '''
        Uniform interface for generating points along the path.
        '''
        if self.op.G == 0 or self.op.G == 1:
            # Straight line, though G0 isn't suppose to modify material.
            yield from walk_line(self.begin, self.end, self.step)
        
        elif self.op.G == 2:
            # Clockwise arc, flip begin, end
            yield from walk_arc(self.end, self.begin, self.arc_center(), self.step, reverse=True)
                
        elif self.op.G == 3:
            # Counter-clockwise arc
            yield from walk_arc(self.begin, self.end, self.arc_center(), self.step)
            
        else:
            raise NotImplementedError("Path for '%s' is not implemented!", op)
    
    def length(self):
        '''
        Uniform interface for path length.
        '''
        if self.op.G == 0 or self.op.G == 1:
            # Straight line
            return line_length(self.begin, self.end)
            
        elif self.op.G == 2:
            # Clockwise arc, flip begin, end
            return arc_length(self.end, self.begin, self.arc_center())
                
        elif self.op.G == 3:
            # Counter-clockwise arc
            return arc_length(self.begin, self.end, self.arc_center())
            
        else:
            raise NotImplementedError("Path for '%s' is not implemented!", op)
            
            
    def bounds(self):
        '''
        Get the min and max XYZ extents
        '''
        if self.op.G == 0 or self.op.G == 1:
            return line_bounds(self.begin, self.end)
            
        elif self.op.G == 2:
            # Clockwise arc, flip begin, end
            return arc_bounds(self.end, self.begin, self.arc_center())
                
        elif self.op.G == 3:
            # Counter-clockwise arc
            return arc_bounds(self.begin, self.end, self.arc_center())
            
        else:
            raise NotImplementedError("Path for '%s' is not implemented!", op)
            
    def duration(self):
        '''
        Time to traverse the path
        '''
        return datetime.timedelta(minutes=self.length()/self.op.F)
    
    def is_plunge(self):
        '''
        Check if the path is Z-only
        '''
        return np.linalg.norm(self.begin[:2] - self.end[:2]) < self.step
        
    def sync_op(self):
        '''
        Update operation to reflect path changes.
        '''
        self.op.X = self.end[0]
        self.op.Y = self.end[1]
        self.op.Z = self.end[2]
        return self.op
        
    def self_check(self):
        '''
        Pair-wise validate that the path is no more than step apart and
        visits begin and end exactly.
        '''
        bounds = self.bounds()
        length = 0
        last = None
        
        for bound in bounds:
            for dim in bound:
                assert(dim < 5000)
        
        for index, point in enumerate(self.walk()):
            assert((point >= bounds[0]).all())
            assert((point <= bounds[1]).all())
        
            if last is None:
                assert((point == self.begin).all())
            else:
                segment = np.linalg.norm(point-last)
                assert(segment <= self.step)
                length += segment
                
            last = point
            
        assert((last == self.end).all())
        assert(np.absolute(length - self.length()) < self.step)
        
    @classmethod  
    def between(cls, a, b, feed, g=0):
        '''
        Create a linear path bridging two paths.
        '''
        op = Operation()
        op.G = g
        op.F = feed
        return cls(op, a.end, b.begin, min(a.step, b.step))
        
    @classmethod  
    def z_exit(cls, path, z, feed, g=0):
        '''
        Create a path lifting out of the specified path.
        '''
        op = Operation()
        op.G = g
        op.F = feed
        result = cls(op, path.end, path.end, path.step)
        result.end[2] = z
        return result
        
    @classmethod  
    def z_enter(cls, path, z, feed, g=1):
        '''
        Create a path loweriung into the specified path.
        '''
        op = Operation()
        op.G = g
        op.F = feed
        result = cls(op, path.begin, path.begin, path.step)
        result.begin[2] = z
        return result
        
        
class Tool(ABC):
    def __init__(self):
        pass
    
class Endmill(Tool):

    def __init__(self, endmill_size, voxel_size):
        self.voxel_size = voxel_size
        self.endmill_size = endmill_size
        
        voxel_diameter = self.endmill_size / voxel_size
        voxel_radius = voxel_diameter / 2
        voxel_r2 = pow(voxel_radius, 2)
        
        mask_width = int(np.ceil(voxel_diameter))
        mask_center = mask_width / 2
        
        real_width = mask_width * voxel_size
        self.offset = np.array((real_width / 2, real_width / 2, 0.0))
        
        shape = (mask_width, mask_width)
        
        self.mask = np.full(shape, np.inf)
        
        mid = mask_width // 2
        if mask_width & 1:
            mid += 1
            
        for a in range(0, mid):
            for b in range (a, mid):
                check = pow(a - mask_center , 2) + pow(b - mask_center, 2)
                if check > voxel_r2:
                    continue
                    
                self.mask[a, b ] = 0
                self.mask[mask_width - a - 1, b ] = 0
                self.mask[a, mask_width - b - 1 ] = 0
                self.mask[mask_width - a - 1, mask_width - b - 1 ] = 0
                
                self.mask[b, a ] = 0
                self.mask[mask_width - b - 1, a ] = 0
                self.mask[b, mask_width - a - 1 ] = 0
                self.mask[mask_width - b - 1, mask_width - a - 1 ] = 0
                
                    
        logging.debug("Endmill mask %f size, %f voxel:\n%s", self.endmill_size, self.voxel_size, self.mask)
        
class VoxelModel():
    #TODO: Split out endmill into a 'Tool' interface.
    
    def __init__(self, bounds, voxel_size):
        self.bounds = bounds
        self.voxel_size = voxel_size
        self.voxel_scale = 1.0/voxel_size
        self.reset()
        
    def reset(self):
        '''
        Create a prestine voxel map at the surface height
        '''
        volume = self.bounds[1] - self.bounds[0]
        voxel_x = 1 + int(np.ceil(volume[0] * self.voxel_scale))
        voxel_y = 1 + int(np.ceil(volume[1] * self.voxel_scale))
        logging.debug("Voxel map size: %dx%d", voxel_x, voxel_y)
        self.voxels = np.full((voxel_x, voxel_y), self.bounds[1][2])
        
    def coordinates(self, tool, center):
        '''
        translate from real world coordinates to voxel coordinates.
        '''
        assert(tool.voxel_size == self.voxel_size)
        
        position = center - self.bounds[0] - tool.offset
        
        i = int(np.round(position[0] * self.voxel_scale))
        j = int(np.round(position[1] * self.voxel_scale))
        shape = np.shape(tool.mask)
        
        return i, j, shape
            
    def carve(self, tool, center):
        '''
        Update the voxel map with the endmill at the specified position.
        '''
        i, j, shape = self.coordinates(tool, center)
        
        self.voxels[i : i + shape[0], j : j + shape[1]] = np.minimum(
            self.voxels[i : i + shape[0], j : j + shape[1]],
            tool.mask + center[2] )
    
    def touch(self, tool, center):
        '''
        Determine if the endmill touches the voxel map at the specified
        position.
        '''
        i, j, shape = self.coordinates(tool, center)
        max_z = np.amax(self.voxels[i : i + shape[0], j : j + shape[1]] - tool.mask)
        if max_z > center[2]:
            logging.warning("Tool below voxel height %f at %s!", z, center)
        return max_z >= center[2]
        
    def crash(self, tool, center):
        '''
        Determine if the endmill crashes into the voxel map at the specified
        position.
        '''
        i, j, shape = self.coordinates(tool, center)
        max_z = np.amax(self.voxels[i : i + shape[0], j : j + shape[1]] - tool.mask)
        return max_z > center[2]

class Simulator():
    '''
    Removes redundand z passes assuming the cutting head can handle up to X depth. Implemented
    via multi-pass voxel model
    '''
    
    def __init__(self, gcode, surface_height, tool, voxel_size, max_step, debug=True):
        self.gcode = gcode
        self.surface_height = surface_height
        self.tool = tool
        self.voxel_size = voxel_size
        self.voxel_scale = 1.0/voxel_size
        self.max_step = max_step
        self.safe_z = surface_height
        self.z_values = set() 
        self.debug = debug
        self.z_feeds = set()
        self.xy_feeds = set()
        
    def parse_paths(self):
        '''
        Process operations to extract implicit prior state for each operation.
        This makes it possible to reason about each operation independently.
        '''
        last_x = None
        last_y = None
        last_z = None
        
        ops = [Operation.parse(line, index + len(self.gcode.preamble)) for index, line in enumerate(self.gcode.lines) if not line.startswith("(")]
        
        self.paths = []
        self.duration = datetime.timedelta()
        
        for op in ops:
            x = op.X if hasattr(op, "X") else last_x
            y = op.Y if hasattr(op, "Y") else last_y
            z = op.Z if hasattr(op, "Z") else last_z
            
            if last_x is None or last_y is None or last_z is None:
                logging.warning("No path for op '%s' at line %d: Unknown prior position.", op, op.source)
                if last_z:
                    self.safe_z = max(last_z, self.safe_z)
            else:
                path = Path(op, np.array((last_x, last_y, last_z)), np.array((x, y, z)), self.voxel_size)
                self.paths.append(path)
                
                if hasattr(path.op, "F"):
                    self.duration += path.duration()
                    
                    if path.is_plunge():
                        self.z_feeds.add(op.F)
                    else:
                        self.xy_feeds.add(op.F)
                    
            if hasattr(op, "X"):
                last_x = op.X
            if hasattr(op, "Y"):
                last_y = op.Y
            if hasattr(op, "Z"):
                last_z = op.Z
                self.z_values.add(op.Z)
                
                
        for a, b in zip(self.paths[:-1], self.paths[1:]):
            if np.linalg.norm(a.end - b.begin) > self.voxel_size:
                raise RuntimeError(f"Path parsing error: end/begin mismatch in sequence {a.end} != {b.begin}!")
                
        logging.info("Parsed %d paths from %d gcode operations (%d lines), %s estimated job time.", len(self.paths), len(ops), len(self.gcode.lines), self.duration)
        logging.debug("Gcode z-values: %s", sorted(self.z_values))
        logging.debug("Gcode z-feeds: %s", sorted(self.z_feeds))
        logging.debug("Gcode xy-feeds: %s", sorted(self.xy_feeds))
        
        if self.debug:
            for index, path in enumerate(self.paths):
                logging.debug("Parsing: Validating path (%d/%d) '%s'.", index + 1, len(self.paths), path.op)
                path.self_check()
                assert((path.begin[:2] == path.end[:2]).all() or path.begin[2] == path.end[2]) 
        
    def find_bounding_box(self):
        '''
        Find the bounding box of all operations that interact with the material
        ignoring step overs/air travel.
        '''
        min_bound = np.array((np.inf, np.inf, np.inf))
        max_bound = -min_bound
        
        for index, path in enumerate(self.paths):
            if min(path.begin[2], path.end[2]) > self.surface_height:
                logging.debug("Bounding: Ignoring path (%d/%d) '%s': Endmill above material.", index + 1, len(self.paths), path.op)
                continue
                
            if path.is_plunge() and path.begin[2] < path.end[2]:
                logging.debug("Bounding: Ignoring path (%d/%d) '%s': Endmill retraction.", index + 1, len(self.paths), path.op)
                continue
                
            logging.debug("Bounding: Simulating path (%d/%d) '%s'.", index + 1, len(self.paths), path.op)
            
            bounds = path.bounds()
            if bounds[0][2] > self.surface_height:
                continue
                
            if bounds[1][2] > self.surface_height:
                bounds[1][2] = self.surface_height
            
            min_bound = np.minimum(bounds[0] - self.tool.offset, min_bound)
            max_bound = np.maximum(bounds[1] + self.tool.offset, max_bound)
                
        logging.info("Cutting area bounds: %s to %s", min_bound, max_bound)
        self.bounds = (min_bound, max_bound)
        
    def init_voxel_map(self):
        '''
        Create a prestine voxel map at the surface height
        '''
        self.model = VoxelModel(self.bounds, self.voxel_size)
        
    def carve_range(self, z_min, z_max):
        '''
        Incrementally simulate paths by z-range.
        '''
        for index, path in enumerate(self.paths):
            if min(path.begin[2], path.end[2]) > z_max:
                logging.debug("Carve: Ignoring path (%d/%d) '%s': Above current z threshold.", index + 1, len(self.paths), path.op)
                continue
                
            if max(path.begin[2], path.end[2]) < z_min:
                logging.debug("Carve: Ignoring path (%d/%d) '%s': Below current z threshold.", index + 1, len(self.paths), path.op)
                continue
                
            if path.is_plunge() and path.begin[2] < path.end[2]:
                logging.debug("Carve: Ignoring path (%d/%d) '%s': Endmill retraction.", index + 1, len(self.paths), path.op)
                continue
                
            if path.is_plunge() and len(self.paths) > (index + 1) and np.linalg.norm(path.end - self.paths[index + 1].begin) <= self.voxel_size:
                logging.debug("Carve: Ignoring path (%d/%d) '%s': Endmill extension.", index + 1, len(self.paths), path.op)
                continue
                
            logging.debug("Carve: Simulating path (%d/%d) '%s'.", index + 1, len(self.paths), path.op)
            
            for point in path.walk():
                if point[2] < z_min or point[2] > z_max:
                    continue
                self.model.carve(self.tool, point)
                
    def optimize_range(self, z_min, z_max):
        '''
        Incrementally optimize paths by z-range.
        '''
        kept = 0
        for index, path in enumerate(self.paths):
            if path.keep:
                logging.debug("Optimize: Ignoring path (%d/%d) '%s': Already optimized.", index + 1, len(self.paths), path.op)
                continue
                
            if min(path.begin[2], path.end[2]) > z_max:
                logging.debug("Optimize: Ignoring path (%d/%d) '%s': Above current z threshold.", index + 1, len(self.paths), path.op)
                continue
                
            if max(path.begin[2], path.end[2]) < z_min:
                logging.debug("Optimize: Ignoring path (%d/%d) '%s': Below current z threshold.", index + 1, len(self.paths), path.op)
                continue
                
            if path.is_plunge() and path.begin[2] < path.end[2]:
                logging.debug("Optimize: Ignoring path (%d/%d) '%s': Endmill retraction.", index + 1, len(self.paths), path.op)
                continue
                
            if path.is_plunge() and len(self.paths) > (index + 1) and np.linalg.norm(path.end - self.paths[index + 1].begin) <= self.voxel_size:
                logging.debug("Optimize: Ignoring path (%d/%d) '%s': Endmill extension.", index + 1, len(self.paths), path.op)
                continue
                
            for point in path.walk():
                if point[2] < z_min or point[2] > z_max:
                    continue
                if self.model.touch(tool, point):
                    # TODO: Optimize path length
                    kept += 1
                    path.keep = True
                    break
            
            if path.keep:        
                logging.debug("Optimize: Keeping path (%d/%d) '%s'.", index + 1, len(self.paths), path.op)
            else:
                logging.debug("Optimize: Discarding path (%d/%d) '%s'.", index + 1, len(self.paths), path.op)
                    
            if path.keep and path.op.G == 0:
                logging.warn("Optimize: G0 path marked as keep!")
                    
        logging.info("Optimize: Keeping %d of %d ops in z-range %f to %f", kept, len(self.paths), z_min, z_max)
        return kept
        
    def adaptive_process(self):
        '''
        Carve and optimize based on max_step and z_values
        '''
        self.parse_paths()
        self.find_bounding_box()
        self.init_voxel_map()
        
        total = 0
        
        z_min = np.inf
        for z_max in sorted(filter(lambda z: z <= self.surface_height, self.z_values), reverse=True):
            if z_max > z_min:
                logging.debug("Adaptive: Ignoring already processed z-value %f.", z_max)
                continue
                
            z_min = z_max - self.max_step
            
            logging.info("Adaptive: Processing z-range %f to %f.", z_min, z_max)
            self.carve_range(z_min, z_max)
            kept = self.optimize_range(z_min, z_max)
            total += kept
            logging.info("Adaptive: Gained %d paths to %d total.", kept, total)
            
            if self.debug:
                fig = plt.figure()
                shape = np.shape(self.model.voxels)
                X, Y = np.mgrid[0:shape[0], 0:shape[1]]
                pcm = plt.pcolormesh(X, Y, self.model.voxels)
                fig.colorbar(pcm, extend='max')
                plt.show()    
            
    def to_gcode(self):
        check_model = VoxelModel(self.bounds, self.voxel_size)
        paths = []
        
        retract_feed = min(self.z_feeds)
        extend_feed = min(self.z_feeds)
        travel_feed = max(self.xy_feeds)
        
        for index, path in enumerate(self.paths):
            if not path.keep:
                logging.debug("GCode: Processing path (%d/%d) '%s': Optimized out.", index + 1, len(self.paths), path.op)
                continue
            
            # Try to step over or direct travel before step-over
            if len(paths):
                xy_distance = np.linalg.norm(paths[-1].end[:2] - path.begin[:2])
                if xy_distance < self.voxel_size:
                    if paths[-1].end[2] == path.begin[2]:
                        # Step-over optimization
                        logging.debug("GCode: Processing path (%d/%d) '%s': Direct continuation, same z.", index + 1, len(self.paths), path.op)
                    else:
                        logging.debug("GCode: Processing path (%d/%d) '%s': Direct continuation, new z.", index + 1, len(self.paths), path.op)
                        paths.append(Path.between(paths[-1], path, feed=extend_feed))
                else:
                    travel_height = max(paths[-1].end[2], path.begin[2])
                    
                    # try direct travel
                    travel = path.between(paths[-1], path, feed=travel_feed, g=0)
                    travel.begin[2] = travel_height
                    travel.end[2] = travel_height
                    
                    crash = any(map(functools.partial(self.model.crash, self.tool), travel.walk()))
                    
                    if crash:
                        travel.begin[2] = self.safe_z
                        travel.end[2] = self.safe_z
                        
                    logging.debug("GCode: Processing path (%d/%d) '%s': Travel at height %f.", index + 1, len(self.paths), path.op, travel.begin[2])
                        
                    if travel.begin[2] != paths[-1].end[2]:
                        paths.append(Path.between(paths[-1], travel, feed=retract_feed, g=0))
                        
                    paths.append(travel)
                    
                    if travel.end[2] != path.begin[2]:
                        paths.append(Path.between(travel, path, feed=extend_feed, g=1))
                    

            else:
                logging.debug("GCode: Processing path (%d/%d) '%s': Initial path.", index + 1, len(self.paths), path.op)
                paths.append(Path.z_enter(path, self.safe_z, feed=extend_feed, g=1))
                
            paths.append(path)
        
        # incrementally carve for travel checks.
        #
        duration = datetime.timedelta()
        bounds_min = self.bounds[0]
        bounds_max = self.bounds[1]
        bounds_max[2] = max(self.z_values)
        for index, path in enumerate(paths):
            duration += path.duration()
            path.sync_op()
            logging.debug("GCode: Verifying path (%d/%d) '%s'.", index + 1, len(paths), path.op)
            for point in path.walk():
                if self.debug and ((point < bounds_min).any() or (point > bounds_max).any()):
                    raise RuntimeError(f"Point {point} outside of bounding box {bounds_min} to {bounds_max}!")
                check_model.carve(self.tool, point)
                
        max_delta = np.amax(np.abs(self.model.voxels - check_model.voxels))
        if max_delta:
            logging.critical("WARNING: Optimized gcode departs from model by up to %f!!!", max_delta)
            
        if self.debug:
            shape = np.shape(self.model.voxels)
            fig, axes = plt.subplots(3, 1)
            X, Y = np.mgrid[0:shape[0], 0:shape[1]]
            
            for ax, data in zip(axes, (self.model.voxels, check_model.voxels, self.model.voxels - check_model.voxels)):
                pcm = ax.pcolormesh(X, Y, data)
                fig.colorbar(pcm, ax=ax, extend='max')
                
            plt.show()
                
        #if not (original_voxels == self.voxels).all():
        #    #raise RuntimeError("Voxel map does not match! Optimization failed!")
        #    pass
        
        # Setup preamble to lift and travel to first path
        ops = []
        op = Operation()
        op.G = 0
        op.Z = self.safe_z
        ops.append(op)
        
        op = Operation()
        op.G = 0
        op.Z = self.safe_z
        op.X = paths[0].begin[0]
        op.Y = paths[0].begin[1]
        ops.append(op)
        
        # Synchronize ops and add path ops
        ops.extend(map(Path.sync_op, paths))
        
        logging.info("GCode: Generated %d ops from %d paths, estimated run time %s (down from %s)", len(ops), len(self.paths), duration, self.duration)
        
        lines = [ f"{op}\n" for op in ops ]
        
        return GcodeFile(self.gcode.preamble, lines, self.gcode.postamble)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GCode post-processing")
    parser.add_argument("input", help="Input path")
    parser.add_argument("output", help="Output path")
    parser.add_argument("-l", "--log-level", default="warn", choices=("warn", "info", "error", "critical", "debug"), help="Log level")
    parser.add_argument("-s", "--step-over-radius", type=float, help="Step-over radius")
    parser.add_argument("-z", "--z-layers", type=float, help="Split file into z-layers below the specified z-height. Output path will be used as a template")
    parser.add_argument("-v", "--voxel-size", type=float, default=0.5, help="Size of voxels")
    parser.add_argument("-e", "--endmill-size", type=float, default=6.35, help="Diameter of endmill")
    parser.add_argument("--surface-height", type=float, default=0.0, help="Height of surface")
    parser.add_argument("-r", "--roughing-step", type=float, help="Max roughing step for combining z layers")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging mode")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level)
    
    gcode = GcodeFile.load(args.input)
        
    if args.step_over_radius:
        gcode.lines = [ line for line in optimize_step_over(gcode.lines, args.step_over_radius, len(gcode.preamble)) ]
        
    if args.roughing_step:
        tool = Endmill(args.endmill_size, args.voxel_size)
        sim = Simulator(gcode, args.surface_height, tool, args.voxel_size, args.roughing_step, debug=args.debug)
        sim.adaptive_process()
        gcode = sim.to_gcode()
    
    if args.z_layers is not None:
        if '{index}' not in args.output:
            logging.error("Output path must be a template with '{index}'!")
            sys.exit(255)
        for index, layer in enumerate(split_z_layers(gcode, args.z_layers)):
            path = args.output.format(index=index)
            logging.debug("Writing '%s'...", path)
            gcode.save(path)
    else:
        gcode.save(args.output)
        
