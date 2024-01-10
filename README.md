# RubiksCubeSolver

This is a program designed to solve the Rubik's Cube. The program is coded and tested on Huawei laptop with Ubuntu 22.04.3 operating system, but it should work using any computer with a webcam. Python libraries cv2, kociemba, numpy, and math are required to run this program.

To solve the Rubik's Cube with this program, download it to your computer and start it from the terminal or IDE. After the program starts running, a window displaying video image from your webcam with instructions on the top of the window should appear on the screen. From this point on, using the program should be pretty straight forward by just following the instructions on the screen. The cube is first detected by showing it to the camera side by side following the instructions at the top of the window and the arrows guiding you on where to rotate. Then the program guides you to make the correct turns to solve the cube by by drawing similar arrows that are used during the detection.

Note that it is expected to use a cube with standard coloring (white on the top, blue on the front, orange on the right, green on the back, red on the left, and yellow on the bottom). You might also need to change the limit values of the colors that are determined on the lines 13-24 of the code depending on your cube and the lighting conditions.

Also note that when the program tells you to show, for example, the green face to the camera, it refers to the face with green center piece since the color of the center piece determines the final color of the face. For people with no experience on Rubik's Cubes this might not be obvious.

Here will be the link to the video on solving a cube with this program in the future.
