import cv2 as cv2
import kociemba as cube
import numpy as np

# initial_state = input("Enter the initial state of the cube: ")

# solution = cube.solve(initial_state)
# print("Solution is:")
# print(solution)
width = 640
height = 360
center_w = width//2
center_h = height//2

def my_print(x, str):
    print(str,end=": ")
    print(x)
    return

def print_color_info(color_name, img):
    BGR_color = np.uint8([[img[center_h][center_w]]])
    HSV_color = cv2.cvtColor((BGR_color), cv2.COLOR_BGR2HSV)
    # print(color_name + " (BGR): ",end="")
    # print(BGR_color)
    print(color_name + " (HSV): ",end="")
    print(HSV_color)
    print()

def get_sigle_color_limits(img):

    BGR_color = np.uint8([[img[center_h][center_w]]])
    HSV_color = cv2.cvtColor((BGR_color), cv2.COLOR_BGR2HSV)
    hue = HSV_color[0][0][0]

    hue_low = hue - 2
    hue_high = hue + 2

    if hue_low < 0:
        hue_low = 0
        hue_high = 4
    
    if hue_high > 180:
        hue_high = 180
        hue_low = 176

    low = [hue_low, 100, 100]
    high = [hue_high, 255, 255]
    limits = [low, high]
    return limits

def calibration_completed(hsv_limits):
    for color in list(hsv_limits.keys()):
        if hsv_limits[color][1][2] == 0:
            return False

    return True

def main():

    video = cv2.VideoCapture(0)
    color = (255,0,0)

    low = [0,0,0]
    high = [0,0,0]

    hsv_limits = {"blue": [low,high], "green": [low,high], "yellow": [low,high],
                  "white": [low,high], "red": [low,high], "orange": [low,high]}
    while True:


        is_ok, img = video.read()

        if not is_ok:
            break

        img = cv2.resize(img, (width, height))
        img = cv2.flip(img, 1)
        
        

        cv2.rectangle(img, (width//2-100, height//2-100), (width//2+100, height//2+100), (255, 0, 0), 5)
        cv2.rectangle(img, (center_w-5, center_h-5), (center_w+5, center_h+5), (255, 0, 0), 3)
        cv2.rectangle(img, (10,10), (30,30), color, 20)
        cv2.imshow('img', img)
        # cv2.imshow('hsv', HSV_img)

        pressed = cv2.waitKey(1)

        if pressed == ord('q'):
            break

        if pressed == ord('p'):
            print(hsv_limits)

        if pressed == ord('d'):
            BGR_color = img[center_h][center_w]
            B = BGR_color[0]
            G = BGR_color[1]
            R = BGR_color[2]

            color = tuple([int(B),int(G),int(R)])
            print("From the if statement: ", end="")
            print(color)

        if pressed == ord('b'):
            print_color_info("Blue", img)
            hsv_limits["blue"] = get_sigle_color_limits(img)
            
        if pressed == ord('g'):
            print_color_info("Green", img)
            hsv_limits["green"] = get_sigle_color_limits(img)

        if pressed == ord('y'):
            print_color_info("Yellow", img)
            hsv_limits["yellow"] = get_sigle_color_limits(img)


        if pressed == ord('w'):
            print_color_info("White", img)
            hsv_limits["white"] = get_sigle_color_limits(img)


        if pressed == ord('r'):
            print_color_info("Red", img)
            hsv_limits["red"] = get_sigle_color_limits(img)

        if pressed == ord('o'):
            print_color_info("Orange", img)
            hsv_limits["orange"] = get_sigle_color_limits(img)

        if pressed == ord('a'):
            if calibration_completed(hsv_limits):
                print(hsv_limits)
            else:
                print("Please finnish the calibration")

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()