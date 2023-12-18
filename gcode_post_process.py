#!/usr/bin/env python3

from collections import OrderedDict
import logging
import argparse
import sys
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
    __slots__ = tuple(parse_map.keys())
    
    def __init__(self):
        pass
        
    @classmethod
    def parse(cls, line):
        result = cls()
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
        

def optimize_step_over(lines, radius):
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
            
        op = Operation.parse(line)
        if op.G == 0:
            pending.append(line)
        elif op.G > 0 and op.G <= 4:
            if pending:
                if last_op is None or (op.distance_squared(last_op) > threshold and last_op.Z == op.Z):
                    yield from pending
                else:
                    logging.debug("Optimized out step-over in lines %d-%d: '%s'", index-len(pending), index, pending)
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
        
        op = Operation.parse(line)
        if hasattr(op, "Z"):
            if op.Z >= z_thresh:
                pass
            elif current_z is None:
                current_z = op.Z
            elif op.Z < current_z:
                logging.debug("Layer created from lines %d-%d at z-height %f.", len(gcode.preamble) + start, len(gcode.preamble) + index, current_z)
                layer = []
                if len(layers):
                    move_to_start = Operation.parse(gcode.lines[start])
                    del move_to_start.Z
                    layer.append(safe_travel_height)
                    layer.append(f"{move_to_start}\n")
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
            move_to_start = Operation.parse(gcode.lines[start])
            del move_to_start.Z
            layer.append(safe_travel_height)
            layer.append(f"{move_to_start}\n")
        layer.extend(gcode.lines[start:])
        
        layers.append(GcodeFile(gcode.preamble, layer, gcode.postamble))
    
    logging.info("Identifed %d layers", len(layers))
    return layers
            
def align_floor(value, unit):
    count = int(value / unit)
    result = count * unit
    if result == value:
        return result
    if result > 0:
        return result
    else:
        return result - unit

def align_ceil(value, unit):
    count = int(value / unit)
    result = count * unit
    if result == value:
        return result
    if result < 0:
        return result
    else:
        return result + unit
        
def walk_arc(begin, end, center, step):
    p0 = begin - center
    p1 = end - center
    r2 = np.dot(p0, p0)
    r2_check = np.dot(p1, p1)
    
    r = np.sqrt(r2)
    r_check = np.sqrt(r2_check)
    
    if np.absolute(r - r_check) > step:
        logging.warn("Bad circle center %s for %s %s leads to different radii %f %f", center, begin, end, r, r_check)
    
    a0 = np.arctan2(p0[1], p0[0])
    a1 = np.arctan2(p1[1], p1[0])
    
    distance = a1 - a0
        
    if distance < 0:
        distance += 2 * np.pi
        
    samples = int(np.ceil(r * distance / step))
    
    for offset in np.linspace(0, distance, samples):
        angle = a0 + offset
        yield (angle, r)
        
        
def walk_line(begin, end, step):
    pointer = end - begin
    length = np.linalg.norm(pointer)
    
    samples = int(np.ceil(length / step))
    
    increment = pointer / samples
    
    for ignored in range(0, samples + 1):
        yield begin
        begin += increment
     

class RoughingFilter():
    '''
    Removes redundand z passes assuming the cutting head can handle up to X depth. Implemented
    via multi-pass voxel model
    '''
    
    def __init__(self, gcode, surface_height, endmill_size, voxel_size, max_step):
        self.gcode = gcode
        self.ops = [Operation.parse(line) for line in gcode.lines if not line.startswith("(")]
        self.surface_height = surface_height
        self.endmill_size = endmill_size
        self.voxel_size = voxel_size
        self.voxel_scale = 1.0/voxel_size
        self.max_step = max_step
        
    def find_bounding_box(self):
        min_bound = np.array((np.inf, np.inf, np.inf))
        max_bound = -min_bound
        
        last_x = None
        last_y = None
        last_z = None
        
        offsets = (-self.endmill_size, self.endmill_size)
        
        for op in self.ops:
            x = op.X if hasattr(op, "X") else last_x
            y = op.Y if hasattr(op, "Y") else last_y
            z = op.Z if hasattr(op, "Z") else last_z
            
            if x is None or y is None or z is None:
                logging.debug("Ignoring op '%s': Unknown starting position.", op)
                
            elif z > self.surface_height:
                logging.debug("Ignoring op '%s': Endmill above material.", op)
        
            elif op.G == 0 or op.G == 1:
                for x_offset in offsets:
                    for y_offset in offsets:
                        point = np.array((x + x_offset, y + y_offset, z))
                        length = np.linalg.norm(point)
                        min_bound = np.minimum(min_bound, point)
                        max_bound = np.maximum(max_bound, point)
            
            elif op.G == 2:
                begin = np.array((last_x, last_y))
                end = np.array((x, y))
                center = begin + np.array((op.I, op.J))
                for angle, radius in walk_arc(end, begin, center, self.voxel_size):
                    radius += self.endmill_size
                    point = center + np.array((np.cos(angle) * radius, np.sin(angle) * radius))
                    point = np.append(point, z)
                    length = np.linalg.norm(point)
                    if length > 25.4*12*4:
                        raise ValueError(f"too long {length}")
                    min_bound = np.minimum(min_bound, point)
                    max_bound = np.maximum(max_bound, point)
                    
            elif op.G == 3:
                begin = np.array((last_x, last_y))
                end = np.array((x, y))
                center = begin + np.array((op.I, op.J))
                for angle, radius in walk_arc(begin, end, center, self.voxel_size):
                    radius += self.endmill_size
                    point = center + np.array((np.cos(angle) * radius, np.sin(angle) * radius))
                    point = np.append(point, z)
                    length = np.linalg.norm(point)
                    if length > 25.4*12*4:
                        raise ValueError(f"too long {length}")
                    min_bound = np.minimum(min_bound, point)
                    max_bound = np.maximum(max_bound, point)
                
            else:
                logging.warning("Ignoring op '%s': Not implemented!", op)
                    
            if hasattr(op, "X"):
                last_x = op.X
            if hasattr(op, "Y"):
                last_y = op.Y
            if hasattr(op, "Z"):
                last_z = op.Z
                
        logging.info("Cutting area bounds: %s to %s", min_bound, max_bound)
        self.bounds = (min_bound, max_bound)
        
    def make_endmill_mask(self):
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
        volume = self.bounds[1] - self.bounds[0]
        voxel_x = 1 + int(np.ceil(volume[0] * self.voxel_scale))
        voxel_y = 1 + int(np.ceil(volume[1] * self.voxel_scale))
        logging.info("Voxel map size: %dx%d", voxel_x, voxel_y)
        self.voxels = np.full((voxel_x, voxel_y), self.surface_height)
    
    def carve_to_depth(self, x, y, z):
        i = int(np.round((x - self.bounds[0][0]) * self.voxel_scale))
        j = int(np.round((y - self.bounds[0][1]) * self.voxel_scale))
        if self.voxels[i, j] < z:
            self.voxels[i, j] = z
            
    def carve_endmill(self, center):
        i = int(np.round((center[0] - self.bounds[0][0]) * self.voxel_scale))
        j = int(np.round((center[1] - self.bounds[0][1]) * self.voxel_scale))
        shape = np.shape(self.endmill_mask)
        width = shape[0]
        mid = width // 2
        self.voxels[ i - mid : i + mid + 1, j - mid : j + mid + 1 ] = np.minimum(
            self.voxels[ i - mid : i + mid + 1, j - mid : j + mid + 1 ],
            self.endmill_mask * center[2] )
    
    def touch_endmill(self, center):
        i = int(np.round((center[0] - self.bounds[0][0]) * self.voxel_scale))
        j = int(np.round((center[1] - self.bounds[0][1]) * self.voxel_scale))
        shape = np.shape(self.endmill_mask)
        width = shape[0]
        mid = width // 2
        z = np.amin(self.voxels[ i - mid : i + mid + 1, j - mid : j + mid + 1 ] * self.endmill_mask)
        if z > center[2]:
            logging.warn("Endmill below voxel height %f at %s!", z, center)
        return z >= center[2]
        
    def carve_above(self, z_threshold):
        last_x = None
        last_y = None
        last_z = None
        
        for index, op in enumerate(self.ops):
            x = op.X if hasattr(op, "X") else last_x
            y = op.Y if hasattr(op, "Y") else last_y
            z = op.Z if hasattr(op, "Z") else last_z
            
            logging.debug("Simulating op (%d/%d): '%s'...", index + 1, len(self.ops), op)
            
            if last_x is None or last_y is None or last_z is None:
                logging.debug("Ignoring op '%s': Unknown starting position.", op)
                
            elif min(z, last_z) > self.surface_height:
                logging.debug("Ignoring op '%s': Endmill above material.", op)
                
            elif max(z, last_z) < z_threshold:
                logging.debug("Ignoring op '%s': Below current z threshold.", op)
        
            elif op.G == 0 or op.G == 1:
                begin = np.array((last_x, last_y, last_z))
                end = np.array((x, y, z))
                for point in walk_line(begin, end, self.voxel_size):
                    self.carve_endmill(point)
            
            elif op.G == 2:
                begin = np.array((last_x, last_y))
                end = np.array((x, y))
                center = begin + np.array((op.I, op.J))
                for angle, radius in walk_arc(end, begin, center, self.voxel_size):
                    point = center + np.array((np.cos(angle) * radius, np.sin(angle) * radius))
                    point = np.append(point, z)
                    self.carve_endmill(point)
                    
            elif op.G == 3:
                begin = np.array((last_x, last_y))
                end = np.array((x, y))
                center = begin + np.array((op.I, op.J))
                for angle, radius in walk_arc(begin, end, center, self.voxel_size):
                    point = center + np.array((np.cos(angle) * radius, np.sin(angle) * radius))
                    point = np.append(point, z)
                    self.carve_endmill(point)
                
            else:
                logging.warning("Ignoring op '%s': Not implemented!", op)
                    
            if hasattr(op, "X"):
                last_x = op.X
            if hasattr(op, "Y"):
                last_y = op.Y
            if hasattr(op, "Z"):
                last_z = op.Z
                
                
    def optimize_range(self, z_min, z_max):
                
        last_x = None
        last_y = None
        last_z = None
        last_x_op = None
        last_y_op = None
        last_z_op = None
        
        keepers = set()
        
        for index, op in enumerate(self.ops):
            x = op.X if hasattr(op, "X") else last_x
            y = op.Y if hasattr(op, "Y") else last_y
            z = op.Z if hasattr(op, "Z") else last_z
            
            logging.debug("Optimizing op (%d/%d): '%s'...", index + 1, len(self.ops), op)
            
            touches = False
            
            if last_x is None or last_y is None or last_z is None:
                logging.debug("Ignoring op '%s': Unknown starting position.", op)
                
            elif min(z, last_z) > z_max:
                logging.debug("Ignoring op '%s': Above current z threshold.", op)
                
            elif max(z, last_z) < z_min:
                logging.debug("Ignoring op '%s': Below current z threshold.", op)
        
            elif op.G == 0 or op.G == 1:
                begin = np.array((last_x, last_y, last_z))
                end = np.array((x, y, z))
                for point in walk_line(begin, end, self.voxel_size):
                    if self.touch_endmill(point):
                        touches = True
                        break
            
            elif op.G == 2:
                begin = np.array((last_x, last_y))
                end = np.array((x, y))
                center = begin + np.array((op.I, op.J))
                for angle, radius in walk_arc(end, begin, center, self.voxel_size):
                    point = center + np.array((np.cos(angle) * radius, np.sin(angle) * radius))
                    point = np.append(point, z)
                    if self.touch_endmill(point):
                        touches = True
                        break
                    
            elif op.G == 3:
                begin = np.array((last_x, last_y))
                end = np.array((x, y))
                center = begin + np.array((op.I, op.J))
                for angle, radius in walk_arc(begin, end, center, self.voxel_size):
                    point = center + np.array((np.cos(angle) * radius, np.sin(angle) * radius))
                    point = np.append(point, z)
                    if self.touch_endmill(point):
                        touches = True
                        break
                
            else:
                logging.warning("Ignoring op '%s': Not implemented!", op)
                
            if touches:
                keepers.add(index)
                keepers.add(last_x_op)
                keepers.add(last_y_op)
                keepers.add(last_z_op)
                    
            if hasattr(op, "X"):
                last_x = op.X
                last_x_op = index
            if hasattr(op, "Y"):
                last_y = op.Y
                last_y_op = index
            if hasattr(op, "Z"):
                last_z = op.Z
                last_z_op = index
        
        logging.info("Keeping %d of %d ops in z-range %f to %f", len(keepers), len(self.ops), z_min, z_max)
        
                

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
        gcode.lines = [ line for line in optimize_step_over(gcode.lines, args.step_over_radius) ]
        
    if args.roughing_step:
        roughing = RoughingFilter(gcode, args.surface_height, args.endmill_size, args.voxel_size, args.roughing_step)
        roughing.find_bounding_box()
        roughing.init_voxel_map()
        roughing.make_endmill_mask()
        roughing.carve_above(-20.0)
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
        
