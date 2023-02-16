import os
import win32api
from time import sleep, time
from datetime import datetime


def check_for_sleep(sleep_time):
    print('Check for sleep')
    initial = win32api.GetCursorPos()
    initial_time = datetime.now()
    print(initial_time, initial)

    sleep(sleep_time)
    current = win32api.GetCursorPos()
    current_time = datetime.now()
    print(current_time, current)

    if initial == current:
        return True
    return False


def mouse_moved():
    print('Mouse movement detecting...')
    initial = win32api.GetCursorPos()
    initial_time = datetime.now()
    print(initial_time, initial)

    while True:
        current = win32api.GetCursorPos()
        current_time = datetime.now()
        if initial != current:
            print(current_time, current)
            return
        sleep(0.05)


def check_for_pattern(pattern, desired_time):
    print('Identifying the pattern')
    initial_time = time()
    for point_x, point_y in pattern:
        while True:
            current_x, current_y = win32api.GetCursorPos()
            current_time = time()
            # print(current_time, current_x, current_y)
            if (current_time - initial_time) > desired_time:
                print('Pattern cannot be created in desired time')
                return False
            if (point_x - 100) < current_x < (point_x + 100) and (point_y - 100) < current_y < (point_y + 100):
                print('Point in range. Look for next point')
                break
            else:
                # print('Point not in range yet')
                pass
            sleep(0.05)
    return True


def system_lock():
    os.system('rundll32.exe user32.dll,LockWorkStation')


def main():
    my_pattern = [
        (280, 120),
        (1130, 120),
        (1130, 590),
    ]
    if check_for_sleep(1):
        # mouse do not move from its position
        # wait for first mouse movement
        mouse_moved()

        # mouse has been moved
        # check for pattern within desired time
        result = check_for_pattern(my_pattern, 2)
        if not result:
            # system_lock()
            print('System going to lock')
        else:
            print('Pattern Identified')


if __name__ == '__main__':
    main()
