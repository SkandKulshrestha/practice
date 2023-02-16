from pynput import keyboard

# The event listener will be running in this block
with keyboard.Events() as events:
    for event in events:
        print(f'Received event {event}')
        if isinstance(event, keyboard.Events.Release):
            if event.key == keyboard.Key.esc:
                break
