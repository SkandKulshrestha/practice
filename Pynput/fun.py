from time import sleep
from pynput import mouse
from pynput import keyboard

import os

# import win32api
# from time import sleep

# savedpos = win32api.GetCursorPos()
# print('Points:', savedpos)
# while(True):
# curpos = win32api.GetCursorPos()
# if savedpos != curpos:
# print('Points:', curpos)
# print("Mouse moved")
# savedpos = curpos
# break
# sleep(0.05)

# print('lock')
# os.system('rundll32.exe user32.dll,LockWorkStation')

# sleep(1)


# GetSystemMetrics
# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getsystemmetrics

# https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes

import win32api
import win32con


def key_pressed(key_value):
    key_status = win32api.GetAsyncKeyState(key_value)
    return key_status & 0x8000


def desired_key_pressed(keys):
    for key in keys:
        if win32api.GetAsyncKeyState(key) & 0x8000:
            print(f'{key} is pressed')
            return True
    return False


def main():
    keys = (
        win32con.VK_LCONTROL, win32con.VK_RCONTROL,
        win32con.VK_LWIN,
        win32con.VK_RETURN,
        win32con.VK_LMENU, win32con.VK_RMENU,
        win32con.VK_LSHIFT, win32con.VK_RSHIFT,
        win32con.VK_TAB
    )
    while not desired_key_pressed(keys):
        # print('waiting...')

        sleep(0.02)


if __name__ == '__main__':
    main()
    print("bye")
