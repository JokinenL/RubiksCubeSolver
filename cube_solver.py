import cv2
import kociemba
import numpy as np
import math
from math import sin, cos, pi

IMG_WIDTH = 640
IMG_HEIGHT = 360
IMG_CENTER = (IMG_WIDTH//2, IMG_HEIGHT//2)
COLORS = ["white", "yellow", "blue", "red", "green", "orange"]

# Limits for the six colors of the cube. Note that these are in HSV-format.
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

# The class Cube represents the Rubik's Cube itself. It has six faces which are saved
# in an attribute called state which represents the current state i.e. scramble of the cube.
# Faces of the cube are usually referenced either via color of the center piece (e.g. blue face)
# via the position of the face relative to the user (e.g. front face). In this program faces are
# considered to match the relative positions as follows: white = up; blue = front; orange = right;
# green = back; red = left; yellow = down. This should explain the use of letters U, F, R, B, L 
# and D in the program. The logic of the syntax of turns goes like this: U means turning the
# up-pointing face clockwise and U' (U-prime) means turning it counter clockwise. Letters x and y
# among the turns stand for rotating the whole cube around the x-axis or y-axis. The actual
# solution is taken from the Kociemba librarys 'solve' function.

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

# Returns the boolean value of contour_a being inside contour_b.
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

# The functionn is used to check if every piece of a given face is being detected after
# initalization.
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


# The function is used to check if a given contour has parents i.e. surrounding contours
# of the same color inside the face. This is useful in a tricky situation where the center piece
# is for example blue and all the other pieces are of the face are for example red. That creates a
# situation where the program is easily tricked to conclude that the center piece is also red since
# it is inside the (inner) red contour which is approximately of a size of the center piece. This
# is a unique situation which can be regognized by checking who is the parent of an inner red
# circle. If it has a parent of a size of the face it is concluded that the center piece is
# of a different color (in this case blue).
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

                
# The funtion creates and returns the masks for all the six colors of the cube. These are
# used to regognize the piece colors.
def get_masks(img):
    mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(mask_img, low_b, high_b)
    mask_green = cv2.inRange(mask_img, low_g, high_g)
    mask_orange = cv2.inRange(mask_img, low_o, high_o)
    mask_yellow = cv2.inRange(mask_img, low_y, high_y)
    mask_white = cv2.inRange(mask_img, low_w, high_w)

    # The color limits (global variables set above) may vary with the lighting of the 
    # environment. It is possible that sometimes the best hue values go beyond the zero
    # (or beyond the 180 degrees which is the same thing). In taht case the red mask has
    # to be created of two separate masks, one from "lower" limit to 180 degrees and one from
    # zero (180) degrees to "higher" limit.
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


    return mask_blue, mask_red, mask_green, mask_orange, mask_yellow, mask_white

# The function creates a data structure which can be used to iterate through all the contours of
# given color and the hiererchy vectors of each contour (which are used when finding contours
# parents).
def get_piece_contours_info(masks):
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

# The function creates and returns the instruction text based on a center color that needs to
# be pointed to the camera. The logic here follows the rules represented in the comment text of
# the class Cube: white => up, blue => front, etc. It is crucial that the right face is pointing
# up when a given face is pointing to the camera, otherwise the interpretation of the face
# showing is not unambiguous.
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
    text = f"Show the {color} centered face {center_facing_up} center facing up"
    return text

# The function draws needed instruction texts on the image with white font on a black background.
def draw_text(img, text, origin):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    top_left_x = origin[0]
    top_left_y = origin[1]
    bg_color = (0, 0, 0)
    text_color = (255, 255, 255)
    size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    w, h = size
    text_bottom_left = point_type_converter((top_left_x, top_left_y + 1.2*h), float_to_int=True)
    rect_top_left = (top_left_x, top_left_y)
    rect_bottom_right = point_type_converter((top_left_x + w, top_left_y + 1.5*h), float_to_int=True)

    cv2.rectangle(img, rect_top_left, rect_bottom_right, bg_color, -1)
    cv2.putText(img, text, text_bottom_left, font, font_scale, text_color, font_thickness)

    return

# The function iterates through all center pieces detecting the faces around them using
# the function detect_face() which is actually the function where pretty mutch everthing
# happens considering the computer vision.
def detect_cube(video):

    for color in COLORS:

        while True:

            cmd, face = detect_face(video, color)


            if cmd == "quit":
                return "quit"
            if cmd == "failed":
                return "failed"
            if cmd == "restart":
                continue
            if cmd == "completed":
                cube.save_face(color, face)
                break
        
    return "detected"

# The function is called from detect_face() to initialize the relative areas of the contours 
# used in detection. The areas are relative to the area of the whole face. 1.0 i.e. 100 % is
# a good initalization value since during detection the smallest relative area is being looked
# for and every time a smaller value is found the old value is replaced by it.
def initialize_areas():
    relative_max = 1.0
    area_array = np.empty([3,3])
    for i in range(3):
        for j in range(3):
            area_array[i][j] = relative_max
    return area_array

# The function returns the boolean value of face_a and face_b being identical.
def faces_match(face_a, face_b):
    for i in range(3):
        for j in range(3):
            if face_a[i][j] != face_b[i][j]:
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

# The function is supposed to find a cube from the image and compute the cordinates of each
# piece center. The funtion returns three values: 
# 1) boolean of whether a face is found
# 2) contour of the face
# 3) numpy array of piece center points

def find_face_and_get_centers(img, marginal = 10):
    mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Everything but black. Black background is necessary for program to work properly.
    low = np.array([0, 0, 100])
    high = np.array([180, 255, 255])
    mask_final = cv2.inRange(mask_img, low, high)

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
            # Skipping cases where the cube is rotated too mutch. It has to be unambiguous
            # which side is facing up.
            if 30 < rotation_deg < 60:
                continue
            
            box = cv2.boxPoints(rectangle)
            box = np.intp(box)

            # Making sure that the object recognized is aprroximately square
            for point_as_array in cnt_of_final_mask:
                point = (float(point_as_array[0][0]), float(point_as_array[0][1]))
                dist = cv2.pointPolygonTest(box, point, True)
                if dist > marginal:
                    continue

            # Computing the transition in x and y axis when moving in the set of cordinates
            # of the rotated cube.
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
 
            x_0, y_0 = top_left

            piece_centers = np.empty((3,3), dtype="f,f")

            # Drawing the contour of the detected face so user can see that
            # the face is detected.
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

# The function forms the image to the desired form from the video output from webcam.
def get_img(video):
    is_ok, img = video.read()

    if not is_ok:
        print("Failed to read the video")
        return False, img

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.flip(img, 1)

    return True, img

# The function is called separately for each of the six faces (defined by center colors)
# of the cube. As inputs, it takes the video image from the webcam and the color of the
# center of the face that should be detected. The first return value contains the information
# about whether the detection was successful, or if it was interrupted due to error or users
# command 'q'. The second return value is 3 x 3 numpy array representing the detected face.
# from the keyboard.

# The funtion prints the detected face as it shows on the screen. Note that the actual face
# being saved to the Cube-object is as it shows in the real world (not mirrored).
def print_mirrored_face(og_face):
    mirrored_face = initialize_face()
    for i in range(3):
        for j in range(3):
            j_reversed = abs(j - 2)
            mirrored_face[i][j] = og_face[i][j_reversed]
    print()
    print("face (mirrored as on the screen):")
    print(mirrored_face)

def detect_face(video, center_color):
    # Detected means that the function has come up with some detection of the face. Verified
    # means that hte user has verified that the detection was right and the face can be saved to
    # the Cube-object.
    detected = False
    verified = False
    gray_face = initialize_face()

    face = initialize_face()
    face_to_return = initialize_face()
    relative_areas = initialize_areas()
    instruction_text = get_instruction_text(center_color)
    while True:
        success, img = get_img(video)
        if not success:
            return "failed", None
        draw_text(img, instruction_text, origin = (10, 10))
        
        # When the face is detected but has not yet been verified, only thing the function does
        # is to wait for user to either verify it or restart the detection with keyboard commands.
        if detected == verified:
        
            masks = get_masks(img)


            face_found, contour_face, piece_centers = find_face_and_get_centers(img)

            # This usually means that theres some kind of movement with the cube (turning or
            # rotating) which means it is time to initialize the detection process in order
            # to detect pieces of the new face showing.
            if not face_found:
                face = initialize_face()
                relative_areas = initialize_areas()
            else:
                area_face = cv2.contourArea(contour_face)
                min_piece_area = 0.08*area_face
                piece_contours_info = get_piece_contours_info(masks)


                for color in piece_contours_info:
                    contours_i = piece_contours_info[color][0]
                    hierarchy_array = piece_contours_info[color][1]

                    for contour_index, contour_i in enumerate(contours_i):
                        area_cnt = cv2.contourArea(contour_i)
                        # Any contour outlining pieces inside the face should have
                        # more than three corners. Contours smaller than computed minimum
                        # area for a single piece are also ignored as noise.
                        if (len(contour_i) > 3 and
                            area_cnt >= min_piece_area):


                            for i in range(3):
                                for j in range(3):
                                    # Since the image is mirrored but the Cube-object is formed
                                    # as it is in real world, both the original value and the
                                    # reversed value of the column index j is needed.
                                    j_reversed = abs(j - 2)
                                    piece_center = piece_centers[i][j]
                                    if cv2.pointPolygonTest(contour_i, piece_center, False) == 1:
                                        
                                        relative_area = area_cnt / area_face
                                        previous_used = relative_areas[i][j_reversed]
                                        if (relative_area <= previous_used and
                                            not parents_inside_face(contours_i, hierarchy_array,
                                                                    contour_index, contour_face)):

                                            if (i == 1 and j == 1):
                                                # If the center color changes it is likely
                                                # that the cube has been rotated so the face should
                                                # be initialized in order to erase the values refering
                                                # to the previous face. Colors are saved to the face only
                                                # if the correct face is showing (detected center color
                                                # matches the desired one). Exeption is, if the desired face
                                                # has already been detected and verified. Then any 
                                                # face showing is detected in order to draw instruction
                                                # arrows for the user so he knows where to rotate.
                                                if face[i][j] != color:
                                                    face = initialize_face()
                                                    relative_areas = initialize_areas()
                                                    piece_contours_info = reverse_dict(piece_contours_info)
                                                if color == center_color or verified:
                                                    face[i][j] = color
                                                    relative_areas[i][j] = relative_area


                                            elif face[1][1] == center_color or verified:  
                                                face[i][j_reversed] = color
                                                relative_areas[i][j_reversed] = relative_area
                                            



            if detection_completed(face) and not detected:

                print_mirrored_face(face)
                instruction_text = "Correctly detected? y/n?"
                detected = True
                    
            if verified:
                if center_color == "white":
                    instruction_text = get_instruction_text("yellow")
                    if detection_completed(face):
                        draw_arrows(img, "x'", piece_centers)
                        if face[1][1] == "yellow":
                            return "completed", face_to_return
                if center_color == "yellow":
                    instruction_text = get_instruction_text("blue")
                    if detection_completed(face):
                        draw_arrows(img, "x", piece_centers)
                        if face[1][1] == "blue":
                            return "completed", face_to_return
                if center_color == "blue":
                    instruction_text = get_instruction_text("red")
                    if detection_completed(face):
                        draw_arrows(img, "y'", piece_centers)
                        if face[1][1] == "red":
                            return "completed", face_to_return
                if center_color == "red":
                    instruction_text = get_instruction_text("green")
                    if detection_completed(face):
                        draw_arrows(img, "y'", piece_centers)
                        if face[1][1] == "green":
                            return "completed", face_to_return
                if center_color == "green":
                    instruction_text = get_instruction_text("orange")
                    if detection_completed(face):
                        draw_arrows(img, "y'", piece_centers)
                        if face[1][1] == "orange":
                            return "completed", face_to_return
                if center_color == "orange":
                    instruction_text = get_instruction_text("green")
                    if detection_completed(face):
                        draw_arrows(img, "y", piece_centers)
                        if face[1][1] == "green":
                            return "completed", face_to_return


        # Uncomment these to check if the color limits set in the beggining
        # are suitable for your environment.

        # cv2.imshow('mask_blue', masks[0])
        # cv2.imshow('mask_red', masks[1])
        # cv2.imshow('mask_green', masks[2])
        # cv2.imshow('mask_orange', masks[3])
        # cv2.imshow('mask_yellow', masks[4])
        # cv2.imshow('mask_white', masks[5])
        if not verified:
            draw_face(img, face)
        else:
            draw_face(img, gray_face)

        cv2.imshow('img', img)

        pressed = cv2.waitKey(1)

        if pressed == ord('q'):
            return "quit", None
        if pressed == ord(' '):
            cube.print_state()
            print()
            print("face:")
            print(face)
            
        if (pressed == ord('y') and 
            detected):
            face_to_return = face.copy()
            verified = True
        if pressed == ord('n'):
            return "restart", face
        
# Some funtions require values of the x and y cordinates to be floats, some functions
# require them to be ints. This funktion is used to convert them.
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


# The function draws guiding arrows to the image so user knows where to rotate or turn.
def draw_arrows(img, turn, piece_centers):

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

# The function draws the face (as it is detected during the function detect_face()) on the screen.
def draw_face(img, face):
    colors_bgr = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0),
              "orange": (0, 123, 255), "yellow": (0, 255, 255), "white": (255, 255, 255),
              "gray": (90, 90, 90), "black": (0, 0, 0)}
    
    origin_face = (500, 40)
    piece_width = 30
    piece_height = 30

    for i in range(3):
        for j in range(3):
            j_reversed = abs(j - 2)
            origin_piece_x = origin_face[0] + j*piece_width
            origin_piece_y = origin_face[1] + i*piece_height
            origin_piece = (origin_piece_x, origin_piece_y)
            end_point_piece_x = origin_piece_x + piece_width
            end_point_piece_y = origin_piece_y + piece_height
            end_point_piece = (end_point_piece_x, end_point_piece_y)

            color_name = face[i][j_reversed]
            if color_name == "init":
                color_name = "gray"
            color_value = colors_bgr[color_name]
            
            cv2.rectangle(img, origin_piece, end_point_piece, color_value, -1)

    for i in range(3):

        for j in range(3):
            j_reversed = abs(j - 2)
            origin_piece_x = origin_face[0] + j*piece_width
            origin_piece_y = origin_face[1] + i*piece_height
            origin_piece = (origin_piece_x, origin_piece_y)
            end_point_piece_x = origin_piece_x + piece_width
            end_point_piece_y = origin_piece_y + piece_height
            end_point_piece = (end_point_piece_x, end_point_piece_y)
            
            cv2.rectangle(img, origin_face, end_point_piece, colors_bgr["black"], 2)

    draw_text(img, text = "DETECTION", origin = (496, 132))
            

# The function is used to supervise and guide the user to make correct turns and rotations
# with the cube. It follows the structure and logic of the function detect face: check the
# comments from there to understand the main structure.
def make_turn(video, turn, previous_center_color, last_turn):
    solved = False
    letter = turn[0]
    # To simplify things, user is always required to show either red or green centered face
    # of the cube during the solve. All the possible moves are easy to execute with either of
    # the cube positions.
    if letter == 'U' or letter == 'D':
        valid_face_centers = ["green", "red"]
    elif letter == 'R' or letter == 'L':
        valid_face_centers = ["green"]
    else: # if letter == 'F' or 'B':
        valid_face_centers = ["red"]
    # Getting rid of unnecessary rotations
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

    face_showing = initialize_face()
    relative_areas = initialize_areas()
    instruction_text = get_instruction_text(center_color)

    while True:
        success, img = get_img(video)
        if not success:
            return "failed", None
        draw_text(img, instruction_text, origin = (10, 10))
        
        if not solved:
            masks = get_masks(img)


            face_found, contour_face, piece_centers = find_face_and_get_centers(img)
            if not face_found:
                face_showing = initialize_face()
                relative_areas = initialize_areas()
            else:
                area_face = cv2.contourArea(contour_face)
                min_piece_area = 0.08*area_face
                piece_contours_info = get_piece_contours_info(masks)


                for color in piece_contours_info:
                    contours_i = piece_contours_info[color][0]
                    hierarchy_array = piece_contours_info[color][1]

                    for contour_index, contour_i in enumerate(contours_i):
                        area_cnt = cv2.contourArea(contour_i)

                        if (len(contour_i) > 3 and
                            area_cnt >= min_piece_area):

                            for i in range(3):
                                for j in range(3):
                                    j_reversed = abs(j - 2)
                                    piece_center = piece_centers[i][j]
                                    if cv2.pointPolygonTest(contour_i, piece_center, False) == 1:
                                        
                                        relative_area = area_cnt / area_face
                                        previous_used = relative_areas[i][j_reversed]
                                        if (relative_area <= previous_used and
                                            not parents_inside_face(contours_i, hierarchy_array,
                                                                    contour_index, contour_face)):
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

                    if faces_match(face_showing, face_pre_turn):
                        draw_arrows(img, turn, piece_centers)

                    elif faces_match(face_showing, face_post_turn):
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

        cv2.imshow('img', img)

        pressed = cv2.waitKey(1)

        if pressed == ord('q'):
            return "quit", None
        if pressed == ord(' '):
            print()
            print("face_showing:")
            print(face_showing)
            print()
            print("red_pre_turn:")
            print(red_pre_turn)
            print()
            print("green_pre_turn:")
            print(green_pre_turn)
            

        
# The solution taken from Kocimba's solve() function is a string where double moves are
# marked so that "U2" stands for "U U". Here the solution is saved in a list of single letters
# where only addition after the letter can be the "'" sign which stands for going counter
# clockwise.
def reform_solution(solution_str):
    solution_list = solution_str.split()
    result = []
    for turn in solution_list:

        if len(turn) > 1 and turn[1] == "2":
            result.append(turn[0])
            result.append(turn[0])
        else:
            result.append(turn)
    return result

# The function is called after the detection is done. It gets the solution for the given scramble
# from Kociemba's solve() function and iterates through every turn in the solution calling make_turn()
# function to execute the turn.
def solve_cube(video):

    
    solution_str = cube.get_solution()
    solution = reform_solution(solution_str)

    previous_center_color = "none"
    last_turn = False

    for i, turn in enumerate(solution):
        if i == len(solution) - 1:
            last_turn = True
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
