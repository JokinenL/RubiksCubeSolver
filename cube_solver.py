import cv2
import kociemba
import numpy as np
import math
from math import sin, cos, pi
import time

# TODO: tunnistusvaiheen rotaationuolet.
# 
# 
# 
# 
#  
# !

# initial_state = input("Enter the initial state of the cube: ")

# solution = cube.solve(initial_state)
# print("Solution is:")
# print(solution)


def approx_equals(a, b, marginal):
    if b - marginal <= a <= b + marginal:
        return True
    
    return False

def objects_to_ints(object_dict):
    int_dict = {"blue": [0,0], "green": [0,0], "yellow": [0,0],
                "white": [0,0], "red": [0,0], "orange": [0,0]}
    for color in int_dict.keys():
        int_dict['blue'][0] = object_dict['blue'][0]

    return int_dict

IMG_WIDTH = 640
IMG_HEIGHT = 360
IMG_CENTER = (IMG_WIDTH//2, IMG_HEIGHT//2)
instruction_org = (50, 20)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (255, 0, 0)
font_thickness = 1

COLORS = ["white", "yellow", "blue", "red", "green", "orange"]

low_b = np.array([105, 100, 100])
high_b = np.array([135, 255, 255])
low_r = np.array([160, 75, 75])
high_r = np.array([180, 255, 255])
low_g = np.array([55, 80, 80])
high_g = np.array([85, 255, 255])
low_o = np.array([5, 100, 100])
high_o = np.array([17, 255, 255])
low_y = np.array([20, 100, 100])
high_y = np.array([40, 255, 255])
low_w = np.array([0, 0, 150])
high_w = np.array([180, 60, 255])

class Cube:
    def __init__(self):
        empty_face = np.empty([3, 3], '<U6')
        
        self.state = {"white": empty_face, "orange": empty_face, "blue": empty_face,
                      "yellow": empty_face, "red": empty_face, "green": empty_face}
    
    def save_face(self, color, face):
        self.state[color] = face

    def get_face(self, color):
        return self.state[color]
    
    def print_state(self):
        print("State of the cube:")
        for color in self.state:
            print()
            print(f"{color} face:")
            print()
            count = 0
            for i in range(3):
                print("    | ", end="")
                for j in range(3):
                    print(self.state[color][i][j], end=" | ")
                    if j == 2:
                        print()

    def color_to_letter(self, color):
        if color == "blue":
            return "F"
        if color == "green":
            return "B"
        if color == "white":
            return "U"
        if color == "yellow":
            return "D"
        if color == "red":
            return "L"
        if color == "orange":
            return "R"

    def get_solution(self):

        state_str = ''
        for color in self.state:
            face = self.state[color]
            for i in range(3):
                for j in range(3):
                    state_str += self.color_to_letter(face[i][j])

        print("state_str:")
        print(state_str)
        print()
        solution = kociemba.solve(state_str)

        return solution

    
    def call_turn(self, turn):
        prime = False
        if len(turn) > 2:
            print("Unknown command")
            return
        if len(turn) == 2:
            if turn[1] == "'":
                turn = turn[0]
                prime = True
            elif turn[1] == "2":
                turn = turn[0] + turn[0]
            else:
                print("Unknown command")
                return

        for letter in turn:
            
            if letter == "R":
                self.right_turn(prime)
            elif letter == "L":
                self.left_turn(prime)
            elif letter == "U":
                self.up_turn(prime)
            elif letter == "D":
                self.down_turn(prime)
            elif letter == "F":
                self.front_turn(prime)
            elif letter == "B":
                self.back_turn(prime)
            else:
                print("Unknown command")
                return
        return
    
    def right_turn(self, prime):
        old_blue = self.state["blue"].copy()
        old_white = self.state["white"].copy()
        old_green = self.state["green"].copy()
        old_yellow = self.state["yellow"].copy()
        
        new_blue = old_blue.copy()
        new_white = old_white.copy()
        new_green = old_green.copy()
        new_yellow = old_yellow.copy()

        for i in range(3):
            i_reversed = abs(i - 2)

            if prime:
                new_yellow[i][2] = old_blue[i][2]
                new_blue[i][2] = old_white[i][2]
                new_white[i][2] = old_green[i_reversed][0]
                new_green[i][0] = old_yellow[i_reversed][2]
            else:
                new_blue[i][2] = old_yellow[i][2]
                new_white[i][2] = old_blue[i][2]
                new_green[i][0] = old_white[i_reversed][2]
                new_yellow[i][2] = old_green[i_reversed][0]
      
        old_orange = self.state["orange"].copy()
        new_orange = initialize_face()
        for i in range(3):
            i_reversed = abs(i - 2)
            for j in range(3):
                j_reversed = abs(j - 2)
                if prime:
                    new_orange[i][j] = old_orange[j][i_reversed]
                else:
                    new_orange[i][j] = old_orange[j_reversed][i]

        self.save_face("blue", new_blue)
        self.save_face("white", new_white)
        self.save_face("green", new_green)
        self.save_face("yellow", new_yellow)
        self.save_face("orange", new_orange)

    def left_turn(self, prime):
        old_blue = self.state["blue"].copy()
        old_white = self.state["white"].copy()
        old_green = self.state["green"].copy()
        old_yellow = self.state["yellow"].copy()
        
        new_blue = old_blue.copy()
        new_white = old_white.copy()
        new_green = old_green.copy()
        new_yellow = old_yellow.copy()

        for i in range(3):
            i_reversed = abs(i - 2)

            if prime:
                new_blue[i][0] = old_yellow[i][0]
                new_white[i][0] = old_blue[i][0]
                new_green[i][2] = old_white[i_reversed][0]
                new_yellow[i][0] = old_green[i_reversed][2]
            else:
                new_yellow[i][0] = old_blue[i][0]
                new_blue[i][0] = old_white[i][0]
                new_white[i][0] = old_green[i_reversed][2]
                new_green[i][2] = old_yellow[i_reversed][0]
      
        old_red = self.state["red"].copy()
        new_red = initialize_face()
        for i in range(3):
            i_reversed = abs(i - 2)
            for j in range(3):
                j_reversed = abs(j - 2)
                if prime:
                    new_red[i][j] = old_red[j][i_reversed]
                else:
                    new_red[i][j] = old_red[j_reversed][i]

        self.save_face("blue", new_blue)
        self.save_face("white", new_white)
        self.save_face("green", new_green)
        self.save_face("yellow", new_yellow)
        self.save_face("red", new_red)

    def up_turn(self, prime):
        old_blue = self.state["blue"].copy()
        old_red = self.state["red"].copy()
        old_green = self.state["green"].copy()
        old_orange = self.state["orange"].copy()
        
        new_blue = old_blue.copy()
        new_red = old_red.copy()
        new_green = old_green.copy()
        new_orange = old_orange.copy()

        for j in range(3):

            if prime:
                new_orange[0][j] = old_blue[0][j]
                new_blue[0][j] = old_red[0][j]
                new_red[0][j] = old_green[0][j]
                new_green[0][j] = old_orange[0][j]
            else:
                new_blue[0][j] = old_orange[0][j]
                new_red[0][j] = old_blue[0][j]
                new_green[0][j] = old_red[0][j]
                new_orange[0][j] = old_green[0][j]
      
        old_white = self.state["white"].copy()
        new_white = initialize_face()
        for i in range(3):
            i_reversed = abs(i - 2)
            for j in range(3):
                j_reversed = abs(j - 2)
                if prime:
                    new_white[i][j] = old_white[j][i_reversed]
                else:
                    new_white[i][j] = old_white[j_reversed][i]

        self.save_face("blue", new_blue)
        self.save_face("red", new_red)
        self.save_face("green", new_green)
        self.save_face("orange", new_orange)
        self.save_face("white", new_white)

    def down_turn(self, prime):
        old_blue = self.state["blue"].copy()
        old_red = self.state["red"].copy()
        old_green = self.state["green"].copy()
        old_orange = self.state["orange"].copy()
        
        new_blue = old_blue.copy()
        new_red = old_red.copy()
        new_green = old_green.copy()
        new_orange = old_orange.copy()

        for j in range(3):

            if prime:
                new_blue[2][j] = old_orange[2][j]
                new_red[2][j] = old_blue[2][j]
                new_green[2][j] = old_red[2][j]
                new_orange[2][j] = old_green[2][j]
            else:
                new_orange[2][j] = old_blue[2][j]
                new_blue[2][j] = old_red[2][j]
                new_red[2][j] = old_green[2][j]
                new_green[2][j] = old_orange[2][j]
                
        old_yellow = self.state["yellow"].copy()
        new_yellow = initialize_face()
        for i in range(3):
            i_reversed = abs(i - 2)
            for j in range(3):
                j_reversed = abs(j - 2)
                if prime:
                    new_yellow[i][j] = old_yellow[j][i_reversed]
                else:
                    new_yellow[i][j] = old_yellow[j_reversed][i]

        self.save_face("blue", new_blue)
        self.save_face("red", new_red)
        self.save_face("green", new_green)
        self.save_face("orange", new_orange)
        self.save_face("yellow", new_yellow)

    def front_turn(self, prime):
        old_red = self.state["red"].copy()
        old_white = self.state["white"].copy()
        old_orange = self.state["orange"].copy()
        old_yellow = self.state["yellow"].copy()
        
        new_red = old_red.copy()
        new_white = old_white.copy()
        new_orange = old_orange.copy()
        new_yellow = old_yellow.copy()

        for i in range(3):
            i_reversed = abs(i - 2)

            if prime:
                new_yellow[0][i] = old_red[i][2]
                new_red[i][2] = old_white[2][i_reversed]
                new_white[2][i] = old_orange[i][0]
                new_orange[i][0] = old_yellow[0][i_reversed]
            else:
                new_red[i][2] = old_yellow[0][i]
                new_white[2][i] = old_red[i_reversed][2]
                new_orange[i][0] = old_white[2][i]
                new_yellow[0][i] = old_orange[i_reversed][0]
      
        old_blue = self.state["blue"].copy()
        new_blue = initialize_face()
        for i in range(3):
            i_reversed = abs(i - 2)
            for j in range(3):
                j_reversed = abs(j - 2)
                if prime:
                    new_blue[i][j] = old_blue[j][i_reversed]
                else:
                    new_blue[i][j] = old_blue[j_reversed][i]

        self.save_face("red", new_red)
        self.save_face("white", new_white)
        self.save_face("orange", new_orange)
        self.save_face("yellow", new_yellow)
        self.save_face("blue", new_blue)

    def back_turn(self, prime):
        old_red = self.state["red"].copy()
        old_white = self.state["white"].copy()
        old_orange = self.state["orange"].copy()
        old_yellow = self.state["yellow"].copy()
        
        new_red = old_red.copy()
        new_white = old_white.copy()
        new_orange = old_orange.copy()
        new_yellow = old_yellow.copy()

        for i in range(3):
            i_reversed = abs(i - 2)

            if prime:
                new_red[i][0] = old_yellow[2][i]
                new_white[0][i] = old_red[i_reversed][0]
                new_orange[i][2] = old_white[0][i]
                new_yellow[2][i] = old_orange[i_reversed][2]
            else:
                new_yellow[2][i] = old_red[i][0]
                new_red[i][0] = old_white[0][i_reversed]
                new_white[0][i] = old_orange[i][2]
                new_orange[i][2] = old_yellow[2][i_reversed]
      
        old_green = self.state["green"].copy()
        new_green = initialize_face()
        for i in range(3):
            i_reversed = abs(i - 2)
            for j in range(3):
                j_reversed = abs(j - 2)
                if prime:
                    new_green[i][j] = old_green[j][i_reversed]
                else:
                    new_green[i][j] = old_green[j_reversed][i]

        self.save_face("red", new_red)
        self.save_face("white", new_white)
        self.save_face("orange", new_orange)
        self.save_face("yellow", new_yellow)
        self.save_face("green", new_green)

        return
    
    def copy(self, cube_to_copy):
        for color in self.state:
            self.state[color] = cube_to_copy.get_face(color).copy()
        return


cube = Cube()



def my_print(x, str):
    print()
    print(str,end=": ")
    print(x)
    return

# def point_in_contour(point, contour):
#     x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(contour)
#     x_point = point[0]
#     y_point = point[1]
#     if (x_contour <= x_point <= x_contour + w_contour and
#         y_contour <= y_point <= y_contour + h_contour):
#         return True
#     else:
#         return False

def contour_in_contour(contour_a, contour_b):
    for point_as_array in contour_a:
        point = (float(point_as_array[0][0]), float(point_as_array[0][1]))
        if cv2.pointPolygonTest(contour_b, point, False) < 0:
            return False
    return True
    
def distance(point_a, point_b):
    x_a, y_a = point_a
    x_b, y_b = point_b
    dist = math.sqrt((x_a - x_b)**2 + (y_a - y_b)**2)
    return dist


def detection_completed(face):
    
    for i in range(3):
        for j in range(3):
            if face[i][j] == "init":
                return False
    
    return True

def initialize_face():
    face = np.empty([3, 3], '<U6')
    for i in range(3):
        for j in range(3):
            face[i][j] = "init"

    return face


    
def parents_inside_face(contour_array, hierarchy_array, index, contour_face):
    hierarchy_i = hierarchy_array[0][index]
    index_of_parent = hierarchy_i[3]
    # If there is no parents at all, the index is -1
    if index_of_parent == -1:
        return False
    
    contour_parent = contour_array[index_of_parent]
    if contour_in_contour(contour_parent, contour_face):
        return True
    
    return False

                

def get_masks(img):
    mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(mask_img, low_b, high_b)
    mask_green = cv2.inRange(mask_img, low_g, high_g)
    mask_orange = cv2.inRange(mask_img, low_o, high_o)
    mask_yellow = cv2.inRange(mask_img, low_y, high_y)
    mask_white = cv2.inRange(mask_img, low_w, high_w)

    if high_r[0] > 180:
        low_r_1 = low_r
        low_r_2 = np.array([0, low_r[1], low_r[2]])
        high_r_1 = np.array([180, high_r[1], high_r[2]])
        high_r_2 = np.array([high_r[0] - 180, high_r[1], high_r[2]])
        mask_red_1 = cv2.inRange(mask_img, low_r_1, high_r_1)
        mask_red_2 = cv2.inRange(mask_img, low_r_2, high_r_2)
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

    else:
        mask_red = cv2.inRange(mask_img, low_r, high_r)

    mask_final = cv2.bitwise_or(mask_blue, mask_red)
    mask_final = cv2.bitwise_or(mask_green, mask_final)
    mask_final = cv2.bitwise_or(mask_orange, mask_final)
    mask_final = cv2.bitwise_or(mask_yellow, mask_final)
    mask_final = cv2.bitwise_or(mask_white, mask_final)

    return mask_blue, mask_red, mask_green, mask_orange, mask_yellow, mask_white, mask_final

def get_piece_contours_info(masks):
    epsilon = 5

    contours_blue, hierarchy_blue = cv2.findContours(masks[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, hierarchy_red = cv2.findContours(masks[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, hierarchy_green = cv2.findContours(masks[2], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, hierarchy_orange = cv2.findContours(masks[3], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, hierarchy_yellow = cv2.findContours(masks[4], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_white, hierarchy_white = cv2.findContours(masks[5], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    piece_contours_info = {'blue': [contours_blue, hierarchy_blue],
                        'red': [contours_red, hierarchy_red],
                        'green': [contours_green, hierarchy_green],
                        'orange': [contours_orange, hierarchy_orange],
                        'yellow': [contours_yellow, hierarchy_yellow],
                        'white': [contours_white, hierarchy_white]}
    return piece_contours_info

def get_instruction_text(color):
    if (color == "blue" or
        color == "red" or
        color == "green" or
        color == "orange"):
        center_facing_up = "white"
    elif color == "white":
        center_facing_up = "green"
    else: # if center_color == yellow
        center_facing_up = "blue"

    instruction_text = f"Show the {color} centered face {center_facing_up} center facing up"

    return instruction_text

def detect_cube(video):

    # for color in COLORS:
    #     face = initialize_face()
    #     for i in range(3):
    #         for j in range(3):
    #             face[i][j] = color
    #     cube.save_face(color, face)
    
    # cube.right_turn(prime = False)
    # cube.up_turn(prime = False)
    # cube.left_turn(prime = True)
    # cube.front_turn(prime = True)
    # return "detected"
    for color in COLORS:

        while True:

            print("Calling detect_face() from detect_cube()")
            cmd, face = detect_face(video, color)


            if cmd == "quit":
                return "quit"
            if cmd == "failed":
                return "failed"
            if cmd == "restart":
                continue
            if cmd == "completed":
                cube.save_face(color, face)
                cube.print_state()
                break
        
    return "detected"


def initialize_contours():
    init_contour = np.array([[0, 0], [IMG_WIDTH, 0], [IMG_WIDTH, IMG_HEIGHT], [0, IMG_HEIGHT]])
    result = [[init_contour, init_contour, init_contour],
              [init_contour, init_contour, init_contour],
              [init_contour, init_contour, init_contour]]
    
    return result

def initialize_areas():
    relative_max = 1.0
    area_array = np.empty([3,3])
    for i in range(3):
        for j in range(3):
            area_array[i][j] = relative_max
    return area_array

def faces_match(face_a, face_b):
    for i in range(3):
        for j in range(3):
            if face_a[i][j] != face_b[i][j]:
                return False
    return True
def approximately_square(contour, marginal = 10):

    rectangle = cv2.minAreaRect(contour)
    w, h = rectangle[1]

    if abs(w - h) > marginal:
        return False
    
    box = cv2.boxPoints(rectangle)
    box = np.intp(box)
    
    for point in contour:
        dist = cv2.pointPolygonTest(box, point, True)
        if dist > marginal:
            return False

    return True

def reverse_dict(og_dict):
    list_of_items = list(og_dict.items())
    reversed_list_of_items = reversed(list_of_items)
    reversed_dict = dict(reversed_list_of_items)
    return reversed_dict

def deg_to_rad(a_deg):
    a_rad = a_deg / 180 * pi
    return a_rad

def find_face_and_get_centers(img, marginal = 10):
    # masks = get_masks(img)
    # mask_final = masks[6]
    mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Everything but black
    low = np.array([0, 0, 100])
    high = np.array([180, 255, 255])
    mask_final = cv2.inRange(mask_img, low, high)
    # cv2.imshow('mask_final', mask_final)

    contours_final, hierarchy = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_final) > 0:
        for cnt_of_final_mask in contours_final:
            area = cv2.contourArea(cnt_of_final_mask)

            if (area < 10000 or
                area > 40000):
                continue

            rectangle = cv2.minAreaRect(cnt_of_final_mask)
            rotation_deg = abs(rectangle[2])
            rotation_rad = deg_to_rad(rotation_deg)
            center_point = rectangle[0]
            if distance(center_point, IMG_CENTER) > 50:
                continue

            if 30 < rotation_deg < 60:
                continue
            
            box = cv2.boxPoints(rectangle)
            box = np.intp(box)

            for point_as_array in cnt_of_final_mask:
                point = (float(point_as_array[0][0]), float(point_as_array[0][1]))
                dist = cv2.pointPolygonTest(box, point, True)
                if dist > marginal:
                    continue


            # Getting the right value
            if 60 <= rotation_deg <= 90:
                h, w = rectangle[1]
                if abs(w - h) > marginal:
                    continue
                w_step = w // 6
                h_step = h // 6
                top_left = box[0]
                dx_per_w_step = sin(rotation_rad)*w_step
                dx_per_h_step = cos(rotation_rad)*h_step
                dy_per_w_step = -cos(rotation_rad)*w_step
                dy_per_h_step = sin(rotation_rad)*h_step
                w_me = distance(box[0], box[1])
                # print("From 60 to 90:")
                # print(f"w_program: {w}")
                # print(f"w_me: {w_me}")
                # h_me = distance(box[1], box[2])
                # print(f"h_program: {h}")
                # print(f"h_me: {h_me}")
                # print()
            else: # if 0 < rotation_deg <= 30:
                w, h = rectangle[1]
                if abs(w - h) > marginal:
                    continue
                w_step = w // 6
                h_step = h // 6
                top_left = box[1]
                dx_per_w_step = cos(rotation_rad)*w_step
                dx_per_h_step = -sin(rotation_rad)*h_step
                dy_per_w_step = sin(rotation_rad)*w_step
                dy_per_h_step = cos(rotation_rad)*h_step
                w_me = distance(box[1], box[2])
                # print("From 0 to 30:")
                # print(f"w_program: {w}")
                # print(f"w_me: {w_me}")
                # h_me = distance(box[0], box[1])
                # print(f"h_program: {h}")
                # print(f"h_me: {h_me}")
                # print()

            # img = cv2.circle(img, box[0],
            #                  radius=2, color=(255, 0 , 0), thickness=-1)
            # img = cv2.circle(img, box[1],
            #                  radius=2, color=(0, 255, 0), thickness=-1)
            # img = cv2.circle(img, box[2],
            #                  radius=2, color=(0, 0, 255), thickness=-1)
            # img = cv2.circle(img, box[3],
            #                  radius=2, color=(255, 255, 255), thickness=-1)
            
            
            
            x_0, y_0 = top_left
            

            piece_centers = np.empty((3,3), dtype="f,f")

            cv2.drawContours(img, [box], 0, (0, 0, 100), 3)

            for i in range(3):
                for j in range(3):
                    piece_center_x = x_0 + (2*j + 1)*dx_per_w_step + (2*i + 1)*dx_per_h_step
                    piece_center_y = y_0 + (2*j + 1)*dy_per_w_step + (2*i + 1)*dy_per_h_step
                    piece_centers[i][j] = (piece_center_x, piece_center_y)
                    piece_center_x = int(piece_center_x)
                    piece_center_y = int(piece_center_y)
                    img = cv2.circle(img, (piece_center_x, piece_center_y),
                                    radius=2, color=(0, 0, 255), thickness=-1)

            return True, box, piece_centers
    
    return False, None, None

def get_img(video):
    is_ok, img = video.read()

    if not is_ok:
        print("Failed to read the video")
        return False, img

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.flip(img, 1)

    return True, img

def detect_face(video, center_color):
    already_completed = False
    face = initialize_face()
    face_to_return = initialize_face()
    relative_areas = initialize_areas()
    instruction_text = get_instruction_text(center_color)
    while True:
        success, img = get_img(video)
        if not success:
            return "failed", None
        cv2.putText(img, instruction_text, (50,50), font, font_scale, font_color,
                    font_thickness)
        
        masks = get_masks(img)


        face_found, contour_face, piece_centers = find_face_and_get_centers(img)
        if not face_found:
            if already_completed:
                return "completed", face
        else:
            # cv2.drawContours(img, [contour_face], 0, (0, 0, 100), 3)
            area_face = cv2.contourArea(contour_face)
            min_piece_area = 0.08*area_face
            piece_contours_info = get_piece_contours_info(masks)


            for color in piece_contours_info:
                contours_i = piece_contours_info[color][0]
                hierarchy_array = piece_contours_info[color][1]

                for contour_index, contour_i in enumerate(contours_i):
                    # epsilon = 0.01 * cv2.arcLength(contour_i, True)
                    # cnt = cv2.approxPolyDP(contour_i, epsilon, True)


                    area_cnt = cv2.contourArea(contour_i)
                    # if color == "red":
                    #             cv2.drawContours(img, [contour_i], 0, (100, 0, 0), 3)
                    # if (contour_in_contour(contour_i, contour_face) and
                    #     len(contour_i) > 3):
                    if (len(contour_i) > 3 and
                        area_cnt >= min_piece_area):
                            
                            
                        # cv2.drawContours(img, [cnt], 0, (0, 0, 100), 3)

                        for i in range(3):
                            for j in range(3):
                                # The center piece is already detected at this point
                                j_reversed = abs(j - 2)
                                piece_center = piece_centers[i][j]
                                if cv2.pointPolygonTest(contour_i, piece_center, False) == 1:
                                    
                                    relative_area = area_cnt / area_face
                                    previous_used = relative_areas[i][j_reversed]
                                    if (relative_area <= previous_used and
                                        not parents_inside_face(contours_i, hierarchy_array,
                                                                contour_index, contour_face)):
                                    # if (relative_area <= previous_used):
                                        if (i == 1 and j == 1):
                                            if face[i][j] != color:
                                                face = initialize_face()
                                                relative_areas = initialize_areas()
                                                piece_contours_info = reverse_dict(piece_contours_info)
                                            if color == center_color or already_completed:
                                                face[i][j] = color
                                                relative_areas[i][j] = relative_area

                                            # img_2 = img
                                            # cv2.drawContours(img_2, [cnt], 0, (0, 0, 100), 3)
                                            # cv2.drawContours(img_2, [contour_face], 0, (100, 0, 0), 3)
                                            # cv2.putText(img_2, color, instruction_org, font, font_scale, font_color,
                                            #             font_thickness)
                                            
                                            # title = str(draw_count)
                                            # cv2.imshow(title, img_2)
                                            # draw_count += 1

                                        elif face[1][1] == center_color or already_completed:  
                                            face[i][j_reversed] = color
                                            relative_areas[i][j_reversed] = relative_area
                                        
                                    
                                    # previous_cnt_used = contours_used[i][j_reversed]
                                    # if contour_in_contour(cnt, previous_cnt_used):
                                    #     face[i][j_reversed] = color
                                    #     contours_used[i][j_reversed] = cnt


        if detection_completed(face):

            if not already_completed:
                print()
                print("face:")
                print(face)
                instruction_text = "Detection completed! (c)ontinue/(r)estart?"
                face_to_return = face.copy()
                already_completed = True
            
            else:
                if center_color == "white":
                    draw_arrows(img, "x'", piece_centers)
                    if face[1][1] == "yellow":
                        return "completed", face_to_return
                if center_color == "yellow":
                    draw_arrows(img, "x", piece_centers)
                    if face[1][1] == "blue":
                        return "completed", face_to_return
                if center_color == "blue":
                    draw_arrows(img, "y'", piece_centers)
                    if face[1][1] == "red":
                        return "completed", face_to_return
                if center_color == "red":
                    draw_arrows(img, "x'", piece_centers)
                    if face[1][1] == "green":
                        return "completed", face_to_return
                if center_color == "green":
                    draw_arrows(img, "x'", piece_centers)
                    if face[1][1] == "orange":
                        return "completed", face_to_return
                if center_color == "orange":
                    draw_arrows(img, "x", piece_centers)
                    if face[1][1] == "green":
                        return "completed", face_to_return


        cv2.imshow('mask_blue', masks[0])
        # cv2.imshow('mask_red', masks[1])
        # cv2.imshow('mask_green', masks[2])
        # cv2.imshow('mask_orange', masks[3])
        # cv2.imshow('mask_yellow', masks[4])
        # cv2.imshow('mask_white', masks[5])
        # cv2.imshow('mask_final', mask_final)
        cv2.imshow('img', img)

        pressed = cv2.waitKey(1)

        if pressed == ord('q'):
            return "quit", None
        if pressed == ord(' '):
            cube.print_state()
            print()
            print("face:")
            print(face)
            
        if (pressed == ord('c') and 
            already_completed):
            return "completed", face
        if pressed == ord('r'):
            return "restart", face
        

def point_type_converter(point, int_to_float = False, float_to_int = False):
    if ((int_to_float == True and float_to_int == True) or
        (int_to_float == False and float_to_int == False)):
        print("point_type_converter: Choose ONE conversion direction!")
        return point
    x = point[0]
    y = point[1]
    if int_to_float:
        point = (float(x), float(y))

    if float_to_int:
        point = (int(x), int(y))

    return point



def draw_arrows(img, turn, piece_centers):
    print("Hello from draw arrows")

    top_left = point_type_converter(piece_centers[0][0], float_to_int=True)
    top_mid = point_type_converter(piece_centers[0][1], float_to_int=True)
    top_right = point_type_converter(piece_centers[0][2], float_to_int=True)
    mid_left = point_type_converter(piece_centers[1][0], float_to_int=True)
    mid_right = point_type_converter(piece_centers[1][2], float_to_int=True)
    bottom_left = point_type_converter(piece_centers[2][0], float_to_int=True)
    bottom_mid = point_type_converter(piece_centers[2][1], float_to_int=True)
    bottom_right = point_type_converter(piece_centers[2][2], float_to_int=True)



    arrow_thickness = 3
    arrow_color = (50, 0, 0)
    

    if turn == "U":
        cv2.arrowedLine(img, top_left, top_right, arrow_color, arrow_thickness)
    if turn == "U'":
        cv2.arrowedLine(img, top_right, top_left, arrow_color, arrow_thickness)
    if turn == "D":
        cv2.arrowedLine(img, bottom_right, bottom_left, arrow_color, arrow_thickness)
    if turn == "D'":
        cv2.arrowedLine(img, bottom_left, bottom_right, arrow_color, arrow_thickness)
    if turn == "R" or turn == "B":
        cv2.arrowedLine(img, top_right, bottom_right, arrow_color, arrow_thickness)
    if turn == "R'" or turn == "B'":
        cv2.arrowedLine(img, bottom_right, top_right, arrow_color, arrow_thickness)
    if turn == "L" or turn == "F":
        cv2.arrowedLine(img, bottom_left, top_left, arrow_color, arrow_thickness)
    if turn == "L'" or turn == "F'":
        cv2.arrowedLine(img, top_left, bottom_left, arrow_color, arrow_thickness)
    if turn == "y":
        cv2.arrowedLine(img, top_left, top_right, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, mid_left, mid_right, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, bottom_left, bottom_right, arrow_color, arrow_thickness)
    if turn == "y'":
        cv2.arrowedLine(img, top_right, top_left, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, mid_right, mid_left, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, bottom_right, bottom_left, arrow_color, arrow_thickness)
    if turn == "x":
        cv2.arrowedLine(img, top_right, bottom_right, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, top_mid, bottom_mid, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, top_left, bottom_left, arrow_color, arrow_thickness)
    if turn == "x'":
        cv2.arrowedLine(img, bottom_right, top_right, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, bottom_mid, top_mid, arrow_color, arrow_thickness)
        cv2.arrowedLine(img, bottom_left, top_left, arrow_color, arrow_thickness)

    return

def make_turn(video, turn, previous_center_color, last_turn):
    solved = False
    letter = turn[0]
    if letter == 'U' or letter == 'D':
        valid_face_centers = ["green", "red"]
    elif letter == 'R' or letter == 'L':
        valid_face_centers = ["green"]
    else: # if letter == 'F' or 'B':
        valid_face_centers = ["red"]
    
    if valid_face_centers.count(previous_center_color) > 0:
        center_color = previous_center_color
    else:
        center_color = valid_face_centers[0]

    red_pre_turn = cube.get_face("red")
    green_pre_turn = cube.get_face("green")

    if center_color == "red":
        face_pre_turn = red_pre_turn
    else: # if center_color == "green"
        face_pre_turn = green_pre_turn

    cube.call_turn(turn)
    face_post_turn = cube.get_face(center_color)

    completed = False
    face_showing = initialize_face()
    relative_areas = initialize_areas()
    instruction_text = get_instruction_text(center_color)

    while True:
        success, img = get_img(video)
        if not success:
            return "failed", None
        cv2.putText(img, instruction_text, (50,50), font, font_scale, font_color,
                    font_thickness)
        
        if not solved:
            masks = get_masks(img)


            face_found, contour_face, piece_centers = find_face_and_get_centers(img)
            if not face_found:
                face_showing = initialize_face()
                relative_areas = initialize_areas()
            else:
                # cv2.drawContours(img, [contour_face], 0, (0, 0, 100), 3)
                area_face = cv2.contourArea(contour_face)
                min_piece_area = 0.08*area_face
                piece_contours_info = get_piece_contours_info(masks)


                for color in piece_contours_info:
                    contours_i = piece_contours_info[color][0]
                    hierarchy_array = piece_contours_info[color][1]

                    for contour_index, contour_i in enumerate(contours_i):
                        # epsilon = 0.01 * cv2.arcLength(contour_i, True)
                        # cnt = cv2.approxPolyDP(contour_i, epsilon, True)


                        area_cnt = cv2.contourArea(contour_i)
                        # if color == "red":
                        #             cv2.drawContours(img, [contour_i], 0, (100, 0, 0), 3)
                        # if (contour_in_contour(contour_i, contour_face) and
                        #     len(contour_i) > 3):
                        if (len(contour_i) > 3 and
                            area_cnt >= min_piece_area):
                                
                                
                            # cv2.drawContours(img, [cnt], 0, (0, 0, 100), 3)

                            for i in range(3):
                                for j in range(3):
                                    # The center piece is already detected at this point
                                    j_reversed = abs(j - 2)
                                    piece_center = piece_centers[i][j]
                                    if cv2.pointPolygonTest(contour_i, piece_center, False) == 1:
                                        
                                        relative_area = area_cnt / area_face
                                        previous_used = relative_areas[i][j_reversed]
                                        if (relative_area <= previous_used and
                                            not parents_inside_face(contours_i, hierarchy_array,
                                                                    contour_index, contour_face)):
                                        # if (relative_area <= previous_used):
                                            if (i == 1 and j == 1):
                                                if face_showing[i][j] != color:
                                                    face_showing = initialize_face()
                                                    relative_areas = initialize_areas()
                                                if color == "red" or color == "green":
                                                    face_showing[i][j] = color
                                                    relative_areas[i][j] = relative_area

                                            elif face_showing[1][1] == "red" or face_showing[1][1] == "green":  
                                                face_showing[i][j_reversed] = color
                                                relative_areas[i][j_reversed] = relative_area
                                            


                if detection_completed(face_showing):

                    # print()
                    # print("face:")
                    # print(face)
                    # instruction_text = "Detection completed! (c)ontinue/(r)estart?"
                    # completed = True

                    if faces_match(face_showing, face_pre_turn):
                        draw_arrows(img, turn, piece_centers)

                    elif faces_match(face_showing, face_post_turn):
                        print("Turn complited")
                        if not last_turn:
                            return "completed", center_color
                        else:
                            solved = True
                    elif (faces_match(face_showing, green_pre_turn) and
                          center_color == "red"):
                        draw_arrows(img, "y", piece_centers)
                    elif (faces_match(face_showing, red_pre_turn) and
                          center_color == "green"):
                        draw_arrows(img, "y'", piece_centers)
        if solved:
            instruction_text = "Cube solved! Press 'q' to close the window."

        # cv2.imshow('mask_blue', masks[0])
        # cv2.imshow('mask_red', masks[1])
        # cv2.imshow('mask_green', masks[2])
        # cv2.imshow('mask_orange', masks[3])
        # cv2.imshow('mask_yellow', masks[4])
        # cv2.imshow('mask_white', masks[5])
        # cv2.imshow('mask_final', mask_final)
        cv2.imshow('img', img)

        pressed = cv2.waitKey(1)

        if pressed == ord('q'):
            return "quit", None
        if pressed == ord(' '):
            # cube.print_state()
            print()
            print("face_showing:")
            print(face_showing)
            print()
            print("red_pre_turn:")
            print(red_pre_turn)
            print()
            print("green_pre_turn:")
            print(green_pre_turn)
            
        # if (pressed == ord('c') and 
        #     completed):
        #     return "completed", face_showing
        # if pressed == ord('r'):
        #     return "restart", face_showing
        
    
def reform_solution(solution_str):
    # rotated = False
    solution_list = solution_str.split()
    result = []
    for turn in solution_list:
        # if ((turn[0] == "F" or turn[0] == "B") and
        #     not rotated):
        #     result.append("y")
        #     rotated = True
        # if ((turn[0] == "R" or turn[0] == "L") and
        #     rotated):
        #     result.append("y'")
        #     rotated = False

        if len(turn) > 1 and turn[1] == "2":
            result.append(turn[0])
            result.append(turn[0])
        else:
            result.append(turn)
    return result

def solve_cube(video):

    print("Hello from solve_cube!")
    cube.print_state()
    solution_str = cube.get_solution()
    solution = reform_solution(solution_str)
    print(f"solution: {solution}")
    previous_center_color = "none"
    last_turn = False

    for i, turn in enumerate(solution):
        if i == len(solution) - 1:
            last_turn = True
        print(f"calling make_turn() from solve_cube() with turn {turn}")
        cmd, previous_center_color = make_turn(video, turn, previous_center_color, last_turn)
        if cmd == "quit":
            return "quit"
        if cmd == "failed":
            return "failed"
    return "solved"

def main():

    video = cv2.VideoCapture(0)
    cmd = detect_cube(video)
    if cmd == "detected":
        cmd = solve_cube(video)
    
    if cmd == "quit":
        print("Program finished due to keyboard command")
    if cmd == "failed":
        print("Program finished due to unexpected error")
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
