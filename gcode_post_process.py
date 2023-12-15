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

def GcodeFile():

    def __init__(self, preamble, lines, postamble):
        self.preamble = preamble
        self.lines = lines
        self.postamble = postamble
        
    def save(self, path):
        with open(path, "w", encoding="utf-8"):
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
            lines = handle.readlines
            
        preamble_range = get_preamble_range(lines)
        preamble = lines[preamble_range[0]: preamble_range[1]]
        logging.debug("Preamble located on lines %d-%d", *preamble_range)
        
        postamble_range = get_postamble_range(lines)
        postamble = lines[postamble_range[0]: postamble_range[1]]
        logging.debug("Postamble located on lines %d-%d", *postamble_range)
        
        return cls(preamble, lines[preamble_range[1]:postamble_range[0]], postamble)
            
    
def split_z_layers(lines, z_thresh=0):
    '''
    Splits a large g-code job into multiple jobs by z-height.
    '''
    preamble_range = get_preamble_range(lines)
    preamble = lines[preamble_range[0]: preamble_range[1]]
    logging.debug("Preamble located on lines %d-%d", *preamble_range)
    
    postamble_range = get_postamble_range(lines)
    postamble = lines[postamble_range[0]: postamble_range[1]]
    logging.debug("Postamble located on lines %d-%d", *postamble_range)
    layers = []
    
    safe_travel_height = lines[preamble_range[1]]
    
    start = preamble_range[1]
    current_z = None
    
    for index, line in enumerate(lines[:postamble_range[0]]):
        if index < start:
            continue
            
        if line.strip().startswith("("):
            continue
        
        op = Operation.parse(line)
        if hasattr(op, "Z"):
            if op.Z >= z_thresh:
                pass
            elif current_z is None:
                current_z = op.Z
            elif op.Z < current_z:
                logging.debug("Layer created from lines %d-%d at z-height %f.", start, index, current_z)
                layer = []
                layer.extend(preamble)
                if len(layers):
                    move_to_start = Operation.parse(lines[start])
                    del move_to_start.Z
                    layer.append(safe_travel_height)
                    layer.append(f"{move_to_start}\n")
                layer.extend(lines[start:index])
                layer.extend(postamble)
                
                layers.append(layer)
                start = index
                current_z = None
            else:
                pass
                
    if start < postamble_range[0]: 
        logging.debug("Layer created from lines %d-%d at z-height %f.", start, postamble_range[0], current_z)
        layer = []
        layer.extend(preamble)
        if len(layers):
            move_to_start = Operation.parse(lines[start])
            del move_to_start.Z
            layer.append(safe_travel_height)
            layer.append(f"{move_to_start}\n")
        layer.extend(lines[start:postamble_range[0]])
        layer.extend(postamble)
        
        layers.append(layer)
    
    logging.info("Identifed %d layers", len(layers))
    return layers


class RoughingFilter():
    '''
    Removes redundand z passes assuming the cutting head can handle up to X depth. Implemented
    via multi-pass voxel model
    '''
    
    def __init__(self, lines, surface_height, endmill_size, voxel_size):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GCode post-processing")
    parser.add_argument("input", help="Input path")
    parser.add_argument("output", help="Output path")
    parser.add_argument("-v", "--log-level", default="warn", choices=("warn", "info", "error", "critical", "debug"), help="Log level")
    parser.add_argument("-s", "--step-over-radius", type=float, help="Step-over radius")
    parser.add_argument("-z", "--z-layers", type=float, help="Split file into z-layers below the specified z-height. Output path will be used as a template")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level)
    
    with open(args.input, "r", encoding="utf-8") as input_handle:
        lines = [ line for line in input_handle ]
        
    if args.step_over_radius:
        lines = [ line for line in optimize_step_over(lines, args.step_over_radius) ]
    
    if args.z_layers is not None:
        if '{index}' not in args.output:
            logger.error("Output path must be a template with '{index}'!")
            sys.exit(255)
        for index, layer in enumerate(split_z_layers(lines, args.z_layers)):
            path = args.output.format(index=index)
            logging.debug("Writing '%s'...", path)
            with open(path, "w", encoding="utf-8") as output_handle:
                output_handle.writelines(layer)
    else:
        with open(args.output, "w", encoding="utf-8") as output_handle:
            output_handle.writelines(lines)
        
        
