from more_itertools import windowed
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as et
import re
import numpy as np
import sys


def interpolateBezier(points, steps=10, t=None):
    points = tuple(points)
    if len(points) == 3:
        def mapper(t, p): return (1-t)**2 * p[0] + 2*(1-t)*t*p[1] + t**2*p[2]
    elif len(points) == 4:
        def mapper(t, p): return (np.power((1-t), 3)*p[0] +
                                  3 * np.power((1-t), 2) * t * p[1] +
                                  3*(1-t)*np.power(t, 2)*p[2] +
                                  np.power(t, 3)*p[3])
    else:
        raise Exception(
            'Can only interpolate cubic and quadratic splines (3 or 4 parameters, got: %s' % str(points))

    if t is not None:
        return mapper(t, [q[0] for q in points]), mapper(t, [q[1] for q in points])
    xGen = (mapper(t, [q[0] for q in points])
            for t in np.linspace(0, 1, steps))
    yGen = (mapper(t, [q[1] for q in points])
            for t in np.linspace(0, 1, steps))

    return zip(xGen, yGen)


def parse_coord(inp):
    return np.array([float(next(inp)),  float(next(inp))])


def _parse_coord(x, y):
    return np.array([float(x),  float(y)])


def path_to_cordinate_chomper(shape_path, PRECISION=5, verbose=False):
    prev = None

    for path in shape_path:
        dxdy = _parse_coord(path['x'], path['y'])
        # print(dxdy)

        if prev is None:
            prev = [np.nan, np.nan]

        for x, y in interpolateBezier(  # Resample the bezier curve
            [
                prev,
                dxdy,
                dxdy,
                dxdy,
            ], steps=PRECISION

        ):
            yield np.array([x, y])
        prev = dxdy


def svg_to_coordinate_chomper(inp, yield_control=False, PRECISION=5, verbose=False):
    prev = None
    command_re = re.compile(r'([a-zA-Z])|([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)')

    try:
        while True:
            chunk = next(inp)
            matches = command_re.findall(chunk)
            for match in matches:
                command = match[0] if match[0] else match[1]
                if command in 'MmLlHhVvCcSsQqTtAaZz':
                    if command == 'M':
                        start = parse_coord(inp)
                        prev = start
                        yield [np.nan, np.nan], 'M'
                        yield list(start), 'M'
                        continue
                    elif command == 'm':
                        if prev is None:
                            start = parse_coord(inp)
                        else:
                            start = parse_coord(inp) + prev
                        prev = start
                        yield [np.nan, np.nan], 'm'
                        yield list(start), 'm'
                        continue
                    elif command in 'zZ':
                        prev = start
                        yield start, 'z'
                        continue
                    elif command == 'l':
                        yield prev, 'l'
                        cur = parse_coord(inp) + prev
                        yield cur, 'l'
                        prev = cur
                        continue
                    elif command == 'L':
                        yield prev, 'L'
                        cur = parse_coord(inp)
                        yield cur, 'L'
                        prev = cur
                        continue
                    elif command == 'H':
                        yield list(prev), 'H'
                        chunk2 = next(inp)
                        x = float(chunk2)
                        prev[0] = x
                        yield list(prev), 'H'
                    elif command == 'h':
                        yield list(prev), 'h'
                        chunk2 = next(inp)
                        x = float(chunk2)
                        prev[0] += x
                        yield list(prev), 'h'
                    elif command == 'v':
                        yield list(prev), 'v'
                        chunk2 = next(inp)
                        y = float(chunk2)
                        prev[1] += y
                        yield list(prev), 'v'
                    elif command == 'V':
                        yield list(prev), 'V'
                        chunk2 = next(inp)
                        y = float(chunk2)
                        prev[1] = y
                        yield list(prev), 'V'
                    else:
                        print(f'Unknown command: {command}')
                        raise ValueError(f'Unknown command {command}')
                else:
                    if command not in 'MmLlHhVvCcSsQqTtAaZz':
                        print(f'Unexpected argument: {command}')
                        raise ValueError(f'Unexpected argument {command}')
                    yield prev, command
    except StopIteration:
        pass
    
# def svg_to_coordinate_chomper(inp, yield_control=False, PRECISION=5, verbose=False):
#     prev = None
#     print(inp)
#     try:
#         while True:
#             chunk = next(inp)

#             if chunk == 'M':
#                 print('Got new start coordinate')
#                 start = parse_coord(inp)
#                 prev = start
#                 yield [np.nan, np.nan], 'M'
#                 yield list(start), 'M'

#                 if verbose:
#                     print(f'M {start}')
#                     # plt.scatter([start[0]],[start[1]])
#                 continue

#             if chunk == 'm':
#                 # print('Got new start coordinate')

#                 # m = next(inp)
#                 if prev is None:
#                     start = parse_coord(inp)
#                 else:
#                     start = parse_coord(inp) + prev
#                 prev = start
#                 yield [np.nan, np.nan], 'm'
#                 yield list(start), 'm'
#                 # yield [np.nan,np.nan]
#                 # print(f'm {start}')
#                 continue

#             # print(chunk)
#             if chunk in 'zZ':
#                 # Go to start:
#                 # print("Returning to start coordinate")
#                 prev = start
#                 yield start, 'z'

#                 # print("Done")
#                 continue

#             if chunk.strip() == 'l':
#                 # Line to command:

#                 yield prev, 'l'
#                 cur = parse_coord(inp)+prev
#                 yield cur, 'l'
#                 prev = cur
#                 continue

#             if chunk.strip() == 'L':
#                 # Line to command:
#                 yield prev, 'L'
#                 cur = parse_coord(inp)
#                 yield cur, 'L'
#                 prev = cur
#                 if verbose:
#                     print(f'L {prev} > {cur}')
#                 continue

#             if chunk[0] == 'c':  # bezier mode
#                 # c dx1,dy1 dx2,dy2 dx,dy

#                 dxdy1 = parse_coord(inp) + prev
#                 dxdy2 = parse_coord(inp) + prev
#                 dxdy = parse_coord(inp) + prev

#                 # print('C Bezier',prev,dxdy1,dxdy2,dxdy)

#                 if yield_control:
#                     yield dxdy1
#                     yield dxdy2
#                     yield dxdy

#                 else:
#                     # Resample the bezier curve
#                     for x, y in interpolateBezier(
#                         [
#                             prev,
#                             dxdy1,
#                             dxdy2,
#                             dxdy
#                         ], steps=PRECISION

#                     ):
#                         yield np.array([x, y]), 'c'

#                 prev = dxdy
#                 # yield prev
#                 continue

#             if chunk[0] == 'C':  # bezier mode
#                 # c dx1,dy1 dx2,dy2 dx,dy

#                 dxdy1 = parse_coord(inp)
#                 dxdy2 = parse_coord(inp)
#                 dxdy = parse_coord(inp)

#                 # print('C Bezier',prev,dxdy1,dxdy2,dxdy)

#                 if yield_control:
#                     yield dxdy1
#                     yield dxdy2
#                     yield dxdy

#                 else:
#                     # Resample the bezier curve
#                     for x, y in interpolateBezier(
#                         [
#                             prev,
#                             dxdy1,
#                             dxdy2,
#                             dxdy
#                         ], steps=PRECISION

#                     ):
#                         yield np.array([x, y]), 'C'

#                 prev = dxdy
#                 # yield prev
#                 continue

#             if chunk[0] == 'A':  # ARC mode:
#                 # print('Got arc')
#                 rx, ry = parse_coord(inp)

#                 x_ax_rot = float(next(inp))
#                 large_arc = int(next(inp))
#                 sweep = int(next(inp))
#                 cx, cy = parse_coord(inp)

#                 raise NotImplementedError()
#                 # yield from arc_sampler( rx,ry,x_ax_rot,large_arc,sweep, cx,cy,n_segs = 30 )
#                 continue

#             if chunk[0] == 'q':  # quadratic bezier mode
#                 # c dx1,dy1 dx2,dy2 dx,dy

#                 dxdy1 = parse_coord(inp) + prev
#                 dxdy = parse_coord(inp) + prev

#                 # print('Q Bezier',prev,dxdy1, dxdy)

#                 if yield_control:
#                     yield dxdy1
#                     yield dxdy

#                 else:
#                     # Resample the bezier curve
#                     for x, y in interpolateBezier(
#                         [
#                             prev,
#                             dxdy1,
#                             dxdy
#                         ], steps=PRECISION

#                     ):
#                         yield np.array([x, y]), 'q'

#                 prev = dxdy
#                 # yield prev
#                 continue

#             if chunk[0] == 'Q':  # quadratic bezier , absolute
#                 # c dx1,dy1 dx2,dy2 dx,dy
#                 # C x1 y1, x2 y2, x y
#                 dxdy1 = parse_coord(inp)
#                 dxdy = parse_coord(inp)
#                 # print('Q Bezier',dxdy1, dxdy)

#                 if yield_control:
#                     yield dxdy1
#                     yield dxdy
#                 else:
#                     # Resample the bezier curve
#                     for x, y in interpolateBezier(
#                         [
#                             prev,
#                             dxdy1,
#                             dxdy
#                         ], steps=PRECISION

#                     ):
#                         yield np.array([x, y]), 'Q'

#                 prev = dxdy
#                 # yield prev
#                 continue

#             if chunk not in 'HhVv':
#                 raise ValueError(f'Unknown command {chunk}')
#                 # print("MISS:",chunk)
#                 prev += parse_coord(chunk)
#                 # print(prev)
#                 yield list(prev)  # prev #+start
#             elif chunk == 'h':
#                 # Parse the next chunk: (single horizontal coordinate).

#                 yield list(prev), 'h'

#                 chunk2 = next(inp)
#                 x = float(chunk2)
#                 prev[0] += x

#                 yield list(prev), 'h'  # +start
#             elif chunk == 'H':
#                 # Parse the next chunk: (single horizontal coordinate)

#                 yield list(prev), 'H'

#                 chunk2 = next(inp)
#                 x = float(chunk2)
#                 prev[0] = x

#                 if verbose:
#                     print(f'H {x} ({prev})')

#                 yield list(prev), 'H'  # +start

#             elif chunk == 'v':
#                 # Parse the next chunk: (single horizontal coordinate)
#                 yield list(prev), 'v'

#                 chunk2 = next(inp)
#                 y = float(chunk2)
#                 prev[1] += y

#                 yield list(prev), 'v'  # +start

#             elif chunk == 'V':
#                 # Parse the next chunk: (single horizontal coordinate)
#                 yield list(prev), 'V'

#                 chunk2 = next(inp)
#                 y = float(chunk2)
#                 prev[1] = y
#                 if verbose:
#                     print(f'H {y} ({prev})')
#                 yield list(prev), 'V'  # +start
#     except StopIteration:
#         pass


def repart(inp):
    for p in inp:
        if len(p) > 1 and p[0].upper() in 'HZMCQ':
            yield p[0]
            yield p[1:]
        else:
            yield p


def path_to_segment_blocks(shape_paths, precision=5):
    for i, path in enumerate(shape_paths):
        coordinates = []
        for (x, y) in path_to_cordinate_chomper(shape_path=path['paths'], PRECISION=precision):
            coordinates.append([x, y])
        # print(coordinates)
        if len(coordinates) > 0:
            yield np.array(list(coordinates_to_segments(coordinates)))


def svg_to_segment_blocks(svg_path, precision=5):
    tree = et.parse(svg_path)
    ns = {'sn': 'http://www.w3.org/2000/svg'}
    root = tree.getroot()
    for i, path in enumerate(root.findall('.//sn:path', ns)):
        # Parse the path in d:
        d = path.attrib['d'].replace(',', ' ')
        parts = d.split()
        coordinates = []

        for (x, y), c in svg_to_coordinate_chomper(inp=repart(parts), PRECISION=precision):
            print(x)

            coordinates.append([x, y])

        if len(coordinates) > 0:
            yield np.array(list(coordinates_to_segments(coordinates)))


def coordinates_to_segments(coordinates):
    for (x, y) in coordinates:
        if np.isnan(x):
            current = x, y
            is_down = False
            continue
        else:
            if not is_down:
                # Move the head to the target location, while still being up
                is_down = True
                prev = None

            if (x, y) != prev:
                if not np.isnan(current[0]):

                    yield [[current[0], current[1]], [x, y]]

                current = x, y
            else:
                # Dont write duplicate coordinates. Waste of space
                pass
            prev = current
