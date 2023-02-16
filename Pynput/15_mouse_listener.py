from pynput import mouse


def on_move(x: int, y: int) -> bool | None:
    print(f'Mouse moved to ({x}, {y})')
    if x == 10:
        print('Stopping listener by returning "False" statement')
        return False
    elif y == 10:
        print('Stopping listener by raising "StopException"')
        raise mouse.Listener.StopException


def on_click(x: int, y: int, button: mouse.Button, pressed: bool) -> bool | None:
    # listener can be stopped by raising StopException or returning False
    if pressed:
        print(f'Mouse clicked at ({x}, {y}) with {button}')
    return


def on_scroll(x: int, y: int, dx: int, dy: int) -> bool | None:
    # listener can be stopped by raising StopException or returning False
    print(f'Mouse scrolled at ({x}, {y})({dx}, {dy})')
    return


with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
    listener.join()
