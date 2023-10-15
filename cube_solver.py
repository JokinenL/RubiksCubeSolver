import cv2
import kociemba as cube
import numpy as np

# initial_state = input("Enter the initial state of the cube: ")

# solution = cube.solve(initial_state)
# print("Solution is:")
# print(solution)

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
hsv_limits = np.load('/home/lauri/Desktop/Kandi/RubiksCubeSolver/hsv_limits.npy',
                     allow_pickle = True)
print(hsv_limits.item())



def my_print(x, str):
    print()
    print(str,end=": ")
    print(x)
    return


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
        # low_b = np.array(hsv_limits.item()["blue"][0])
        # high_b = np.array(hsv_limits.item()["blue"][1])
        # low_r = np.array(hsv_limits.item()["red"][0])
        # high_r = np.array(hsv_limits.item()["red"][1])
        # low_g = np.array(hsv_limits.item()["green"][0])
        # high_g = np.array(hsv_limits.item()["green"][1])
        # low_o = np.array(hsv_limits.item()["orange"][0])
        # high_o = np.array(hsv_limits.item()["orange"][1])
        # low_y = np.array(hsv_limits.item()["yellow"][0])
        # high_y = np.array(hsv_limits.item()["yellow"][1])
        # low_w = np.array(hsv_limits.item()["white"][0])
        # high_w = np.array(hsv_limits.item()["white"][1])

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
        low_w = np.array([105, 0, 150])
        high_w = np.array([145, 50, 255])

        
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

        contours, hierarchy = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w//2
                center_y = y + h//2
                center_marginal = 100
                if (10000 <= area <= 40000 and
                   center_w - center_marginal <= center_x <= center_w + center_marginal and
                   center_h - center_marginal <= center_y <= center_h + center_marginal):

                    
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
        
        ws = 150
        hs = ws
        areas = ws*hs
        x1s = center_w - ws//2
        x2s = center_w + ws//2
        y1s = center_h - hs//2
        y2s = center_h + hs//2

        wb = 220
        hb = wb
        areab = wb*hb
        x1b = center_w - wb//2
        x2b = center_w + wb//2
        y1b = center_h - hb//2
        y2b = center_h + hb//2

        instruction_text = f"areas: {areas}, areab: {areab}"

        # cv2.putText(img, instruction_text, instruction_org, font, font_scale, font_color,
        #             font_thickness)
        # cv2.rectangle(img, (x1s, y1s), (x2s, y2s), (0, 0, 255))
        # cv2.rectangle(img, (x1b, y1b), (x2b, y2b), (0, 0, 255))

        cv2.imshow('mask_blue', mask_blue)
        cv2.imshow('mask_red', mask_red)
        cv2.imshow('mask_green', mask_green)
        cv2.imshow('mask_orange', mask_orange)
        cv2.imshow('mask_yellow', mask_yellow)
        cv2.imshow('mask_white', mask_white)
        cv2.imshow('mask_final', mask_final)
        cv2.imshow('img', img)



        pressed = cv2.waitKey(1)

        if pressed == ord('q'):
            break
        if pressed == ord(' '):
            my_print(hsv_limits.dtype, 'hsv_limits.dtype')
            my_print(hsv_limits.item()['blue'], "hsv_limits.item()['blue']")
            a = np.array([[1,2,3],[1,2,3]])
            my_print(a.dtype, 'a.dtype')

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
