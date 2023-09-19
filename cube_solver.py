import cv2 as cv2
import kociemba as cube

# initial_state = input("Enter the initial state of the cube: ")

# solution = cube.solve(initial_state)
# print("Solution is:")
# print(solution)

video = cv2.VideoCapture(0)
while True:

    is_ok, image_input = video.read()
    cv2.imshow("output", image_input)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
