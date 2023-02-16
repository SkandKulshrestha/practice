from pynput import mouse


class MyException(Exception):
    pass


def on_click(x, y, button, pressed):
    if button == mouse.Button.left:
        raise MyException(button)


# Collect events until released
with mouse.Listener(
        on_click=on_click) as listener:
    try:
        listener.join()
    except MyException as e:
        print('{0} was clicked'.format(e.args[0]))
