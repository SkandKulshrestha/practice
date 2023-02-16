from pynput import mouse

with mouse.Events() as events:
    for event in events:
        print(f'Received event {event}')
        if isinstance(event, mouse.Events.Move):
            if event.x == 10:
                break
