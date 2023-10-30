import cv2
import kociemba
import numpy as np
import math
import time

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

width = 640
height = 360
center_w = width//2
center_h = height//2
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
low_g = np.array([60, 80, 80])
high_g = np.array([80, 255, 255])
low_o = np.array([5, 120, 120])
high_o = np.array([12, 255, 255])
low_y = np.array([20, 100, 100])
high_y = np.array([40, 255, 255])
low_w = np.array([120, 0, 150])
high_w = np.array([180, 50, 255])

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

def detect_cube(video):
    for color in COLORS:
        cmd = detect_face(video, color)
        if cmd == "quit":
            return "quit"
    return "detected"

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


def correct_center_showing(piece_contours_info, contour_face, face_center, correct_color, img):
    area_face = cv2.contourArea(contour_face)
    min_piece_area = 0.08*area_face
    center_color = "init"
    area_used = area_face
    

    
    for color in piece_contours_info:
            contours_i = piece_contours_info[color][0]
            hierarchy_array = piece_contours_info[color][1]

            
            for contour_index, contour_i in enumerate(contours_i):
                epsilon = 0.01 * cv2.arcLength(contour_i, True)
                # cnt = cv2.approxPolyDP(contour_i, epsilon, True)
                cnt = contour_i

                area_cnt = cv2.contourArea(cnt)
                if (contour_in_contour(cnt, contour_face) and
                    len(cnt) > 3 and
                    area_cnt >= min_piece_area and
                    cv2.pointPolygonTest(cnt, face_center, False) == 1):
                        
                        if  area_cnt < area_used:
                            center_color = color
                            area_used = area_cnt
                            cnt_used = cnt

                        elif (area_cnt == area_used and
                            not parents_inside_face(contours_i, hierarchy_array, contour_index, contour_face)):
                            center_color = color
                            area_used = area_cnt
                            cnt_used = cnt

    if center_color == correct_color:
        print(f"Color used: {center_color}")
        # cv2.drawContours(img, cnt_used, -1, (0, 0, 100), 3)
        # cv2.imshow('img_2', img)

        return True
    else:
        return False
                    

def get_masks(img):
    mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(mask_img, low_b, high_b)
    mask_red = cv2.inRange(mask_img, low_r, high_r)
    mask_green = cv2.inRange(mask_img, low_g, high_g)
    mask_orange = cv2.inRange(mask_img, low_o, high_o)
    mask_yellow = cv2.inRange(mask_img, low_y, high_y)
    mask_white = cv2.inRange(mask_img, low_w, high_w)

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



def detect_face(video, center_color):
    correct_count = 0
    center_verified = False
    face = initialize_face()

    if (center_color == "blue" or
        center_color == "red" or
        center_color == "green" or
        center_color == "orange"):
        center_facing_up = "white"
    elif center_color == "white":
        center_facing_up = "green"
    else: # if center_color == yellow
        center_facing_up = "blue"

    instruction_text = f"Show the {center_color} centered face {center_facing_up} center facing up"
    while True:
        is_ok, img = video.read()

        if not is_ok:
            break

        img = cv2.resize(img, (width, height))
        img = cv2.flip(img, 1)
        cv2.putText(img, instruction_text, instruction_org, font, font_scale, font_color,
                    font_thickness)

        masks = get_masks(img)
        mask_final = masks[6]

        contours_face, hierarchy = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_face) > 0:
            for contour_face in contours_face:
                area_face = cv2.contourArea(contour_face)
                # Area of a single piece should be 1/9 of the face area, but
                # the shapes are not detected with perfect accuracy so a little
                # error marginal is needed here. Hence the minimum acceptable
                # area for the piece is set lower than it should theoretically be.
                min_piece_area = 0.08*area_face
                x, y, w, h = cv2.boundingRect(contour_face)

                center_x = x + w//2
                center_y = y + h//2
                center_marginal = 50
                square_shape_marginal = 10
                if (10000 <= area_face <= 40000 and
                   approx_equals(center_x, center_w, center_marginal) and
                   approx_equals(center_y, center_h, center_marginal) and
                   approx_equals(w, h, square_shape_marginal)):

                    piece_w = w // 3
                    piece_h = h // 3
                    piece_contours_info = get_piece_contours_info(masks)

                    piece_centers = np.zeros((3,3,2))
                    piece_centers = np.empty((3,3), dtype="f,f")
                    for i in range(3):
                        for j in range(3):
                            piece_center_x = x + (j + 0.5)*piece_w
                            piece_center_y = y + (i + 0.5)*piece_h
                            piece_centers[i][j] = (piece_center_x, piece_center_y)

                    if not center_verified:
                        if correct_center_showing(piece_contours_info,
                            contour_face, piece_centers[1][1],
                            center_color, img):
                            correct_count += 1
                        else:
                            correct_count = 0
                        
                        if correct_count >= 10:
                            center_verified = True
                            print(f"center verified: {center_color}")
                            face[1][1] = center_color

                    if center_verified:

                        for color in piece_contours_info:
                            contours_i = piece_contours_info[color][0]
                            hierarchy_array = piece_contours_info[color][1]

                            
                            for contour_index, contour_i in enumerate(contours_i):
                                epsilon = 0.01 * cv2.arcLength(contour_i, True)
                                cnt = cv2.approxPolyDP(contour_i, epsilon, True)

                                area_cnt = cv2.contourArea(cnt)
                                if (contour_in_contour(cnt, contour_face) and
                                    len(cnt) > 3):
                                    if area_cnt >= min_piece_area:
                                        
                                        cv2.drawContours(img, [cnt], 0, (0, 0, 100), 3)

                                        for i in range(3):
                                            for j in range(3):
                                                # The center piece is already detected at this point
                                                if (i == 1 and j == 1):
                                                    continue
                                                j_reversed = abs(j - 2)
                                                piece_center = piece_centers[i][j]
                                                if cv2.pointPolygonTest(cnt, piece_center, False) == 1:
                                                    face[i][j_reversed] = color

                    if detection_completed(face):
                        cube.save_face(center_color, face)
                        return "completed"
                                  

        cv2.imshow('mask_blue', masks[0])
        cv2.imshow('mask_red', masks[1])
        cv2.imshow('mask_green', masks[2])
        cv2.imshow('mask_orange', masks[3])
        cv2.imshow('mask_yellow', masks[4])
        cv2.imshow('mask_white', masks[5])
        cv2.imshow('mask_final', mask_final)
        cv2.imshow('img', img)

        pressed = cv2.waitKey(1)

        if pressed == ord('q'):
            return "quit"
        if pressed == ord(' '):
            cube.print_state()
            print()
            print("face:")
            print(face)

def solve_cube(video):
    print("Hello from solve_cube!")
    cube.print_state()
    print(cube.get_solution())
    return

def main():

    video = cv2.VideoCapture(0)
    cmd = detect_cube(video)
    if cmd == "detected":
        cmd = solve_cube(video)
    
    if cmd == "quit":
        print("Program finished due to keyboard command")
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
