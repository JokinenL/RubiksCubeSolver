import cv2 as cv2
import kociemba as cube
import numpy as np

# initial_state = input("Enter the initial state of the cube: ")

# solution = cube.solve(initial_state)
# print("Solution is:")
# print(solution)

video = cv2.VideoCapture(0)
while True:

    is_ok, input_image = video.read()

    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(input_image, input_image, mask=mask)

    cv2.imshow('input_image', input_image )
    cv2.imshow('result', result)
    cv2.imshow('mask', mask)



    

    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord('p'):
        # TODO: Tulosta input_imagen tiedot:
        # Kuinka suuri taulukko kyseessä?
        # Yksittäisten piksleien väri?
        # Tätä kautta saat kalibbroitua värientunnistusrajat optimaalisiksi
        # juuri kyseiseen ympäristöön nähden.
        break

video.release()
cv2.destroyAllWindows()
