def combine(*providers):
    while True:
        frames = [next(provider, None) for provider in providers]
        if any([frame is None for frame in frames]): break
        yield frames
