import time

from pynput import mouse

# https://pythonhosted.org/pynput/mouse.html

controller = mouse.Controller()

# read pointer position
print(f'The current pointer position is {controller.position}')
time.sleep(5)

# set pointer position
controller.position = (10, 20)
print(f'The current pointer position is {controller.position}')
time.sleep(2)

# move pointer relative to current position
controller.move(5, -5)
print(f'The current pointer position is {controller.position}')
time.sleep(2)

# press and release
print('Pressing and releasing left button')
controller.press(mouse.Button.left)
controller.release(mouse.Button.left)
time.sleep(2)

controller.position = (818, 360)
# press and release
print('Pressing and releasing right button')
controller.press(mouse.Button.right)
controller.release(mouse.Button.right)
time.sleep(2)

controller.move(-10, 0)

# double click
print('Double clicking left button')
controller.click(mouse.Button.left, 2)
time.sleep(2)

# scroll two steps down
print('Scrolling')
controller.scroll(0, 2)
