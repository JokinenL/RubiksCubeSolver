import cv2
import kociemba
import numpy as np
import math
from math import sin, cos, pi
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
low_o = np.array([3, 100, 100])
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


def correct_center_showing(piece_contours_info, contour_face, face_center, correct_color, img):
    area_face = cv2.contourArea(contour_face)
    min_piece_area = 0.07*area_face
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


def detect_cube(video):

    for color in COLORS:
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
        while True:

            print("Calling detect_face() from detect_cube()")
            cmd, face = detect_face(video, instruction_text, color)


            if cmd == "quit":
                return "quit"
            if (cmd == "completed" and face[1][1] == color):
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

def deg_to_rad(a_deg):
    a_rad = a_deg / 180 * pi
    return a_rad

def find_face_and_get_centers(img, marginal = 10):
    # masks = get_masks(img)
    # mask_final = masks[6]
    mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Everything but black
    low = np.array([0, 0, 50])
    high = np.array([180, 255, 255])
    mask_final = cv2.inRange(mask_img, low, high)
    cv2.imshow('mask_final', mask_final)

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

def detect_face(video, instruction_text, center_color):
    completed = False
    face = initialize_face()
    relative_areas = initialize_areas()
    while True:
        is_ok, img = video.read()

        if not is_ok:
            print("Breaking the while-loop in detect_face()")
            break

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.flip(img, 1)
        cv2.putText(img, instruction_text, (50,50), font, font_scale, font_color,
                    font_thickness)
        
        if not completed:
            masks = get_masks(img)
            mask_final = masks[6]


            face_found, contour_face, piece_centers = find_face_and_get_centers(img)
            if face_found:
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
                        if (contour_in_contour(contour_i, contour_face) and
                            len(contour_i) > 3):
                            if area_cnt >= min_piece_area:
                                
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
                                                if (i == 1 and j == 1):
                                                    if face[i][j] != color:
                                                        face = initialize_face()
                                                        relative_areas = initialize_areas()
                                                    if color == center_color:
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

                                                elif face[1][1] == center_color:  
                                                    face[i][j_reversed] = color
                                                    relative_areas[i][j_reversed] = relative_area
                                                
                                            
                                            # previous_cnt_used = contours_used[i][j_reversed]
                                            # if contour_in_contour(cnt, previous_cnt_used):
                                            #     face[i][j_reversed] = color
                                            #     contours_used[i][j_reversed] = cnt

                            else: # if area_cnt < min_piece_area
                                
                                do_something = 0
                                # cv2.drawContours(img, [cnt], 0, (100, 0, 0), 3)

            if detection_completed(face):
                # if verifying:
                #     return "completed", face
                # else: # if not verifying
                #     cmd, face_verified = detect_face(video, instruction_text, True)
                #     if faces_match(face, face_verified):
                #         return "completed", face
                print()
                print("face:")
                print(face)
                instruction_text = "Detection completed! (c)ontinue/(r)estart?"
                completed = True

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
            cube.print_state()
            print()
            print("face:")
            print(face)
            
        if (pressed == ord('c') and 
            completed):
            return "completed", face
        if pressed == ord('r'):
            return "restart", face
        
    print("Reached end of the function detect_face()")

            
def show_instructions(video, turn):

    return

def solve_cube(video):

    print("Hello from solve_cube!")
    cube.print_state()
    solution = cube.get_solution()
    print(solution)
    for turn in solution:
        show_instructions(video, turn)
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
