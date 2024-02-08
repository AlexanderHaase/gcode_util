#!/usr/bin/env python3

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

from geometry import *
from feature import *
    
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
        indices.append(len(vertices) - 1)
            
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
        
        setback = 3 / (25.4*12)
        
        self.bulkheads = {
            Features.BACK_BULKHEAD: Bulkhead(back_bulkhead_plane, setback, tolerance),
            Features.FRONT_BULKHEAD: Bulkhead(front_bulkhead_plane, setback, tolerance),
            Features.DAY_BULKHEAD: Bulkhead(day_bulkhead_plane, setback, tolerance),
            Features.COCKPIT_BACK: Bulkhead(cockpit_back_plane, setback, tolerance),
            Features.COCKPIT_FRONT: Bulkhead(cockpit_front_plane, setback, tolerance)
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
    parser.add_argument("-t", "--tolerance", type=float, default=0.001, help="Max error tolerance")
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
        area += panel.area() * panel.quantity
        
    logging.info("Area: %f", area)
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()
    
