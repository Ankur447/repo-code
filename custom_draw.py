import logging
from lib import SVG_GCode
from lib.pydexarm import Dexarm
from lib.logger import formatLogger
from lxml import etree
import time

import serial
import time

#import xmltodict

serial_port = "COM5"
arm = None
svg_gcode = SVG_GCode(
    precision=10,
    z_feedrate=3000,
    x_offset=-115,
    y_offset=230,
    z_surface_touch=0,
    z_up_offset=10,
    bed_size_x=220,
    bed_size_y=140,
    longest_edge=220,
    canvas_width=733,
    canvas_height=668,
    verbose=True,
    fit_canvas=False)
logger = formatLogger(__name__)

def draw(response):
    #response = request.get_json()
    #response is the gcode
    #logger.info(f"response: {response}")
    if len(response) > 1:
        plot_gcode(response)
        #logger.info(f'{response}')
    return sendResponse(type='success', message='Path draw successfully.')

def plot_gcode(data):
    global arm
    connect_arm()
    if arm is not None:
        try:
            #data = svg_gcode.path_to_gcode(paths=response, plot_image=False, plot_file=False, gcode_path='output.gcode')
            time.sleep(0.2)
            for line in data:
                logger.info(f'{line}\r')
                arm._send_cmd(f'{line}\r')
                time.sleep(0.1)
            disconnect_arm()
        except:
            logging.error(f'Something went wrong while processing.')
    else:
        logging.error("No OnePlus Arm connected.")


def sendResponse(type='info', message='Message'):
    return {'type': type, 'message': message}

##########################################################################################
# Helpers
##########################################################################################

def connect_arm():
    logger.info("connecting arm")
    global arm
    try:
        arm = Dexarm(port=serial_port)
        x, y, z, e, a, b, c = arm.get_current_position()
        message = "OnePlus Arm connected x: {}, y: {}, z: {}, e: {}\na: {}, b: {}, c: {}".format(
            x, y, z, e, a, b, c)
        logger.info(message)
        return message
    except:
        return False

def disconnect_arm():
    logger.info("disconnecting arm")
    global arm
    if arm is not None:
        # if arm.ser.is_open:
        #     arm.go_home()
        arm.close()
        arm = None
        return "Disconnected"
    else:
        return "No OnePlus Arm connected."

def send_gcode_file(gcode_path):
    logger.info(f"Sending file {gcode_path}")
    # Parse the SVG file
    tree = etree.parse(gcode_path)
    # Get the root element of the XML tree
    root = tree.getroot()
    svg_namespace = {"svg": "http://www.w3.org/2000/svg"}
    polyline_elements = root.xpath("//svg:polyline", namespaces=svg_namespace)
    print(len(polyline_elements))
    for polyline in polyline_elements:
        # Access attributes or text content of polyline elements
        points = polyline.get("points")
        # Split the string into individual coordinate pairs
        coordinate_pairs = points.split(" ")
        # Parse each coordinate pair into its X and Y components
        coordinates = [pair.split(',') for pair in coordinate_pairs]
        draw(coordinates)
        #print(f"length of points: {len(points)}")
        #print(f"length of coordinate_pairs: {len(coordinate_pairs)}")
        #print(f"str coordinate_pairs: {str(coordinate_pairs)}")
        #print(f"str of coordinates: {str(coordinates)}").
        #print("=============")
        time.sleep(2)
    #with open(gcode_path, 'r') as file:
    #    lines = file.readlines()
    #    print(len(lines))
    #    svg_dict = xmltodict.parse(lines[1])
    #    print(svg_dict)
        #for line in svg_dict:
            #draw(line.strip())
            #response = send_gcode(line.strip(), ser)
            #print(f"KKKKKKK {line}: {svg_dict[line]}")


def main():

    logger.info("STARTING...")
    gcode_path = 'svg/img2img2.svg'
    #ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    #send_gcode_file(gcode_path)
    data = svg_gcode.svg_to_gcode(r"svg/abhi2.svg",False,False,True,r"output.gcode")
    plot_gcode(data)
    
    #svg_gcode.path_to_gcode(r"svg/ho_no_ns.svg",False,False,True)
    
    #logger.info(f"data: {data}")
    #ser.close()
    
if __name__ == '__main__':
    main()
