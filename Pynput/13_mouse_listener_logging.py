import os.path
import logging

from pynput.mouse import Listener

logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'mouse.txt'),
    level=logging.DEBUG,
    format='%(asctime)s: %(message)s'
)


def on_move(x, y):
    logging.info(f'Mouse moved to ({x}, {y})')


def on_click(x, y, button, pressed):
    if pressed:
        logging.info(f'Mouse clicked at ({x}, {y}) with {button}')


def on_scroll(x, y, dx, dy):
    logging.info(f'Mouse scrolled at ({x}, {y})({dx}, {dy})')


with Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
    listener.join()
