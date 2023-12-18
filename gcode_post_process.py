#!/usr/bin/env python3

from collections import OrderedDict
import logging
import argparse
import sys
import itertools
import numpy as np

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
    r2 = np.dot(p0[:2], p0[:2])
    r2_check = np.dot(p1[:2], p1[:2])
    
    r = np.sqrt(r2)
    r_check = np.sqrt(r2_check)
    
    if np.absolute(r - r_check) > step:
        logging.warn("Bad circle center %s for %s %s leads to different radii %f %f", center, begin, end, r, r_check)
    
    a0 = np.arctan2(p0[1], p0[0])
    a1 = np.arctan2(p1[1], p1[0])
    
    angular_distance = a1 - a0
        
    if angular_distance < 0:
        angular_distance += 2 * np.pi
        
    samples = int(np.ceil(r * angular_distance / step))
    
    if reverse:
        steps = zip(np.linspace(angular_distance, 0, samples), np.linspace(p1[2], p0[2], samples))
    else:
        steps = zip(np.linspace(0, angular_distance, samples), np.linspace(p0[2], p1[2], samples))
    
    for offset, z in steps:
        angle = a0 + offset
        yield np.array((center[0] + np.cos(angle) * r, center[1] + np.sin(angle) *  r, z))
        
        
def walk_line(begin, end, step):
    '''
    Walks a line from beginning to end with at most step-length intervals.
    '''
    pointer = end - begin
    length = np.linalg.norm(pointer)
    
    samples = int(np.ceil(length / step))
    
    increment = pointer / samples
    
    current = begin.copy()
    
    for ignored in range(0, samples + 1):
        yield current
        current += increment
        
class Path():
    '''
    Captures the path taken by an operation, making implicit state explicit.
    
    Steps progress linearly from beginning to end, so that the path may be
    shortened or split in the inituitive way.
    '''
    __slots__: ("op", "begin", "end", "step", "keep")
    
    def __init__(self, op, begin, end, step, keep=False):
        self.begin = begin
        self.end = end
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
            
    def is_plunge(self):
        return np.linalg.norm(self.begin[:2] - self.end[:2]) < self.step
     

class RoughingFilter():
    '''
    Removes redundand z passes assuming the cutting head can handle up to X depth. Implemented
    via multi-pass voxel model
    '''
    
    def __init__(self, gcode, surface_height, endmill_size, voxel_size, max_step):
        self.gcode = gcode
        self.surface_height = surface_height
        self.endmill_size = endmill_size
        self.voxel_size = voxel_size
        self.voxel_scale = 1.0/voxel_size
        self.max_step = max_step
        self.safe_z = surface_height
        self.z_values = set() 
        
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
                    
            if hasattr(op, "X"):
                last_x = op.X
            if hasattr(op, "Y"):
                last_y = op.Y
            if hasattr(op, "Z"):
                last_z = op.Z
                self.z_values.add(op.Z)
                
        logging.info("Parsed %d paths from %d gcode operations (%d lines)", len(self.paths), len(ops), len(self.gcode.lines))
        logging.debug("Gcode z-values: %s", self.z_values)
        
    def find_bounding_box(self):
        '''
        Find the bounding box of all operations that interact with the material
        ignoring step overs/air travel.
        '''
        min_bound = np.array((np.inf, np.inf, np.inf))
        max_bound = -min_bound
        
        offsets = [
            np.array((self.endmill_size, self.endmill_size, 0.0)),
            np.array((self.endmill_size, -self.endmill_size, 0.0)),
            np.array((-self.endmill_size, self.endmill_size, 0.0)),
            np.array((-self.endmill_size, -self.endmill_size, 0.0))
        ]
        
        for index, path in enumerate(self.paths):
            if min(path.begin[2], path.end[2]) > self.surface_height:
                logging.debug("Bounding: Ignoring path (%d/%d) '%s': Endmill above material.", index + 1, len(self.paths), path.op)
                continue
                
            if path.is_plunge() and path.begin[2] < path.end[2]:
                logging.debug("Bounding: Ignoring path (%d/%d) '%s': Endmill retraction.", index + 1, len(self.paths), path.op)
                continue
                
            logging.debug("Bounding: Simulating path (%d/%d) '%s'.", index + 1, len(self.paths), path.op)
                
            if path.is_plunge():
                # Not sure this optimization is worth the code.
                step = path.begin if path.begin[2] < path.end[2] else path.end
                for offset in offsets:
                    point = step + offset
                    min_bound = np.minimum(min_bound, point)
                    max_bound = np.maximum(max_bound, point)
                continue
                
            for step in path.walk():
                if step[2] > self.surface_height:
                    continue
                for offset in offsets:
                    point = step + offset
                    min_bound = np.minimum(min_bound, point)
                    max_bound = np.maximum(max_bound, point)
                
        logging.info("Cutting area bounds: %s to %s", min_bound, max_bound)
        self.bounds = (min_bound, max_bound)
        
    def make_endmill_mask(self):
        '''
        Create a matrix mask representing the endmill in voxel space.
        Much faster than piecewise interaction with the voxel map.
        '''
        width = int(np.ceil(self.endmill_size * self.voxel_scale))
        if not(width & 1):
            width += 1
        mask = np.zeros((width, width))
        r2 = pow(self.endmill_size / 2, 2)
        mid = width // 2
        for a in range(0, mid):
            for b in range (a, mid):
                check = (pow(a, 2) + pow(b, 2)) * pow(self.voxel_size, 2)
                if check > r2:
                    continue
                mask[mid + a, mid  + b ] = 1
                mask[mid + a, mid  - b ] = 1
                mask[mid - a, mid  + b ] = 1
                mask[mid - a, mid  - b ] = 1
                mask[mid + b, mid  + a ] = 1
                mask[mid + b, mid  - a ] = 1
                mask[mid - b, mid  + a ] = 1
                mask[mid - b, mid  - a ] = 1
        logging.debug("Endmill mask:\n%s", mask) 
        self.endmill_mask = mask
        
    def init_voxel_map(self):
        '''
        Create a prestine voxel map at the surface height
        '''
        volume = self.bounds[1] - self.bounds[0]
        voxel_x = 1 + int(np.ceil(volume[0] * self.voxel_scale))
        voxel_y = 1 + int(np.ceil(volume[1] * self.voxel_scale))
        logging.info("Voxel map size: %dx%d", voxel_x, voxel_y)
        self.voxels = np.full((voxel_x, voxel_y), self.surface_height)
            
    def carve_endmill(self, center):
        '''
        Update the voxel map with the endmill at the specified position.
        '''
        i = int(np.round((center[0] - self.bounds[0][0]) * self.voxel_scale))
        j = int(np.round((center[1] - self.bounds[0][1]) * self.voxel_scale))
        shape = np.shape(self.endmill_mask)
        width = shape[0]
        mid = width // 2
        self.voxels[ i - mid : i + mid + 1, j - mid : j + mid + 1 ] = np.minimum(
            self.voxels[ i - mid : i + mid + 1, j - mid : j + mid + 1 ],
            self.endmill_mask * center[2] )
    
    def touch_endmill(self, center):
        '''
        Determine if the endmill touches the voxel map at the specified
        position.
        '''
        i = int(np.round((center[0] - self.bounds[0][0]) * self.voxel_scale))
        j = int(np.round((center[1] - self.bounds[0][1]) * self.voxel_scale))
        shape = np.shape(self.endmill_mask)
        width = shape[0]
        mid = width // 2
        z = np.amin(self.voxels[ i - mid : i + mid + 1, j - mid : j + mid + 1 ] * self.endmill_mask)
        if z > center[2]:
            logging.warn("Endmill below voxel height %f at %s!", z, center)
        return z >= center[2]
        
    def carve_range(self, z_min, z_max):
        '''
        Process all paths within the specified z-range.
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
                
            logging.debug("Carve: Simulating path (%d/%d) '%s'.", index + 1, len(self.paths), path.op)
            
            for point in path.walk():
                if point[2] < z_min or point[2] > z_max:
                    continue
                self.carve_endmill(point)
                
    def optimize_range(self, z_min, z_max):
        kept = 0
        for index, path in enumerate(self.paths):
            if min(path.begin[2], path.end[2]) > z_max:
                logging.debug("Optimize: Ignoring path (%d/%d) '%s': Above current z threshold.", index + 1, len(self.paths), path.op)
                continue
                
            if max(path.begin[2], path.end[2]) < z_min:
                logging.debug("Optimize: Ignoring path (%d/%d) '%s': Below current z threshold.", index + 1, len(self.paths), path.op)
                continue
                
            if path.is_plunge() and path.begin[2] < path.end[2]:
                logging.debug("Optimize: Ignoring path (%d/%d) '%s': Endmill retraction.", index + 1, len(self.paths), path.op)
                continue
                
            logging.debug("Optimize: Evaluating path (%d/%d) '%s'.", index + 1, len(self.paths), path.op)
            for point in path.walk():
                if point[2] < z_min or point[2] > z_max:
                    continue
                if self.touch_endmill(point):
                    # TODO: Optimize path length
                    path.keep = True
                    kept += 1
                    break
                    
            if path.keep and path.op.G == 0:
                logging.warn("Optimize: G0 path marked as keep!")
                    
        logging.info("Optimize: Keeping %d of %d ops in z-range %f to %f", kept, len(self.paths), z_min, z_max)
        
                

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
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level)
    
    gcode = GcodeFile.load(args.input)
        
    if args.step_over_radius:
        gcode.lines = [ line for line in optimize_step_over(gcode.lines, args.step_over_radius, len(gcode.preamble)) ]
        
    if args.roughing_step:
        roughing = RoughingFilter(gcode, args.surface_height, args.endmill_size, args.voxel_size, args.roughing_step)
        roughing.parse_paths()
        roughing.find_bounding_box()
        roughing.init_voxel_map()
        roughing.make_endmill_mask()
        roughing.carve_range(-20.0, 0.0)
        roughing.optimize_range(-20.0, 0.0)
    
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
        
