import cv2
import kociemba
import numpy as np
import math

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
low_b = np.array([105, 100, 100])
high_b = np.array([135, 255, 255])
low_r = np.array([160, 100, 100])
high_r = np.array([180, 255, 255])
low_g = np.array([60, 100, 100])
high_g = np.array([80, 255, 255])
low_o = np.array([2, 100, 100])
high_o = np.array([15, 255, 255])
low_y = np.array([20, 100, 100])
high_y = np.array([40, 255, 255])
low_w = np.array([120, 0, 150])
high_w = np.array([180, 50, 255])

def initialize_cube():
    cube = np.empty([6, 3, 3], '<U6')
    for i in range(6):
        for j in range(3):
            for k in range(3):
                cube[i][j][k] = "init"
    return cube

cube = initialize_cube()



def my_print(x, str):
    print()
    print(str,end=": ")
    print(x)
    return

def point_in_contour(point, contour):
    x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(contour)
    x_point = point[0]
    y_point = point[1]
    if (x_contour <= x_point <= x_contour + w_contour and
        y_contour <= y_point <= y_contour + h_contour):
        return True
    else:
        return False

def contour_in_contour(contour_a, contour_b):
    x_a1, y_a1, w_a, h_a = cv2.boundingRect(contour_a)
    x_b1, y_b1, w_b, h_b = cv2.boundingRect(contour_b)
    x_a2 = x_a1 + w_a
    y_a2 = y_a1 + h_a
    x_b2 = x_b1 + w_b
    y_b2 = y_b1 + h_b

    if (x_b1 <= x_a1 <= x_b2 and
        x_b1 <= x_a2 <= x_b2 and
        y_b1 <= y_a1 <= y_b2 and
        y_b1 <= y_a2 <= y_b2):
        return True
    else:
        return False
    
def distance(point_a, point_b):
    x_a, y_a = point_a
    x_b, y_b = point_b
    dist = math.sqrt((x_a - x_b)**2 + (y_a - y_b)**2)
    return dist

def main():

    video = cv2.VideoCapture(0)
    # instruction_text = "Welcome to CubeSolver!"

    while True:


        is_ok, img = video.read()

        if not is_ok:
            break

        img = cv2.resize(img, (width, height))
        img = cv2.flip(img, 1)
        
        
        # Masks
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

        contours_face, hierarchy = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_face) > 0:
            for contour_face in contours_face:
                area_face = cv2.contourArea(contour_face)
                # Area of a single piece should be 1/9 of the face area, but
                # the shapes are not detected with perfect accuracy so a little
                # error marginal is needed here. Hence the minimum acceptable
                # area for the piece is set lower than it should theoretically be.
                min_piece_area = area_face//10
                x, y, w, h = cv2.boundingRect(contour_face)

                center_x = x + w//2
                center_y = y + h//2
                center_marginal = 50
                square_shape_marginal = 10
                if (10000 <= area_face <= 40000 and
                   approx_equals(center_x, center_w, center_marginal) and
                   approx_equals(center_y, center_h, center_marginal) and
                   approx_equals(w, h, square_shape_marginal)):

                    epsilon = 0.1*cv2.arcLength(contour_face, True)

                    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
                    piece_w = w // 3
                    piece_h = h // 3
                    contours_blue, hierarchy_blue = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_red, hierarchy_red = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_green, hierarchy_green = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_orange, hierarchy_orange = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_yellow, hierarchy_yellow = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_white, hierarchy_white = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    piece_contour_info = {'blue': [contours_blue, hierarchy_blue],
                                      'red': [contours_red, hierarchy_red],
                                      'green': [contours_green, hierarchy_green],
                                      'orange': [contours_orange, hierarchy_orange],
                                      'yellow': [contours_yellow, hierarchy_yellow],
                                      'white': [contours_white, hierarchy_white]}
                    
                    

                    piece_centers = np.zeros((3,3,2))
                    piece_centers = np.empty((3,3), dtype="f,f")
                    for i in range(3):
                        for j in range(3):
                            piece_center_x = x + (j + 0.5)*piece_w
                            piece_center_y = y + (i + 0.5)*piece_h
                            piece_centers[i][j] = (piece_center_x, piece_center_y)

                    for color in piece_contour_info:
                        contours_i = piece_contour_info[color][0]

                        
                        for conour_i in contours_i:
                            epsilon = 0.01 * cv2.arcLength(conour_i, True)
                            cnt = cv2.approxPolyDP(conour_i, epsilon, True)

                            area_cnt = cv2.contourArea(cnt)
                            if (contour_in_contour(cnt, contour_face) and
                                len(cnt) > 3):
                                if area_cnt >= min_piece_area:
                                    cv2.drawContours(img, [cnt], 0, (0, 0, 100), 3)

                                    for i in range(3):
                                        for j in range(3):
                                            piece_center = piece_centers[i][j]
                                            if cv2.pointPolygonTest(cnt, piece_center, False) == 1:
                                                cube[0][i][j] = color
                                # The white center piece with blue GAN-logo (brand of the cube)
                                # has to be detected differently
                                # elif (color == 'white' and
                                #       area_cnt > 0):
                                #     M = cv2.moments(cnt)
                                #     cnt_center_x = float(M["m10"] / M["m00"])
                                #     cnt_center_y = float(M["m01"] / M["m00"])
                                #     cnt_center = (cnt_center_x, cnt_center_y)
                                #     face_center = piece_centers[1][1]
                                #     if distance(cnt_center, face_center) <= 5:
                                #         cube[0][1][1] = 'white'


                                  

        # cv2.imshow('mask_blue', mask_blue)
        # cv2.imshow('mask_red', mask_red)
        # cv2.imshow('mask_green', mask_green)
        # cv2.imshow('mask_orange', mask_orange)
        # cv2.imshow('mask_yellow', mask_yellow)
        # cv2.imshow('mask_white', mask_white)
        # cv2.imshow('mask_final', mask_final)
        cv2.imshow('img', img)

        pressed = cv2.waitKey(1)

        if pressed == ord('q'):
            break
        if pressed == ord(' '):
            print()
            print("Face:")
            print(cube[0])
            # print()
            # print("hierarchy_green:")
            # print(hierarchy_green)
            # print()
            # print("hierarchy_blue:")
            # print(hierarchy_blue)
            # print()
            # print("bounding_line (of the face):")
            # print(bounding_line)
            # print()
            # print(contour_face)


    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
