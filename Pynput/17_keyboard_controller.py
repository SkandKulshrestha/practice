import time

from pynput import keyboard

controller = keyboard.Controller()
time.sleep(5)

# press and release space
print('Press and release space key')
controller.press(keyboard.Key.space)
controller.release(keyboard.Key.space)
time.sleep(2)

# type a lower case a
print('Press and release "a"')
controller.press('a')
controller.release('a')
time.sleep(2)

# type a lower case b
print('Tap "b"')
controller.tap('b')
time.sleep(2)

# type an upper case A
print('Press and release "A"')
controller.press('A')
controller.release('A')
time.sleep(2)

# type an upper case B
print('Press and release "B" using shift key pressed')
print(f'Is shift key pressed {controller.shift_pressed}')
with controller.pressed(keyboard.Key.shift):
    print(f'Is shift key pressed {controller.shift_pressed}')
    controller.press('b')
    controller.release('b')
time.sleep(2)

# type 'Hello World' using the shortcut type method
print('Typing "Hello World"')
controller.type('Hello World')
time.sleep(2)

print('Press and hold multiple keys')
with controller.pressed(keyboard.Key.ctrl, keyboard.Key.shift):
    print(f'Is shift key pressed {controller.shift_pressed}')
    controller.press('a')
    controller.release('a')
