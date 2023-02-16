from pynput import keyboard


def on_press(key: keyboard.Key | keyboard.KeyCode | None):
    try:
        print(f'alphanumeric key {key.char} pressed')
    except AttributeError:
        print(f'special key {key} pressed')


def on_release(key: keyboard.Key | keyboard.KeyCode | None):
    print(f'{key} released')
    if key == keyboard.Key.esc:
        # Stop listener
        raise keyboard.Listener.StopException


# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
