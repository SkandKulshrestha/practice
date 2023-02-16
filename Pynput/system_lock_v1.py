#! python3

import os
from datetime import datetime

from pynput import mouse, keyboard

system_open = True


def lock_system():
    global system_open
    if system_open:
        print(f'System locked at {datetime.now()}')
        system_open = False
        os.system('rundll32.exe user32.dll,LockWorkStation')
    raise mouse.Listener.StopException


def on_move(x: int, y: int) -> bool | None:
    print(f'Mouse moved to ({x}, {y})')
    lock_system()
    return False


def on_click(x: int, y: int, button: mouse.Button, pressed: bool) -> bool | None:
    print(f'Mouse clicked at ({x}, {y}) with {button} with {pressed}')
    lock_system()
    return False


def on_scroll(x: int, y: int, dx: int, dy: int) -> bool | None:
    print(f'Mouse scrolled at ({x}, {y})({dx}, {dy})')
    lock_system()
    return False


def on_press(key: keyboard.Key | keyboard.KeyCode | None):
    try:
        print(f'alphanumeric key {key.char} pressed')
    except AttributeError:
        print(f'special key {key} pressed')
    lock_system()


def on_release(key: keyboard.Key | keyboard.KeyCode | None):
    print(f'{key} released')
    lock_system()


if __name__ == '__main__':
    print(f'Script started at {datetime.now()}')

    # mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
    # keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    # mouse_listener.start()
    # keyboard_listener.start()
    # mouse_listener.join()
    # keyboard_listener.join()

    with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as mouse_listener, \
            keyboard.Listener(on_press=on_press, on_release=on_release) as keyboard_listener:
        mouse_listener.join()
        keyboard_listener.join()
