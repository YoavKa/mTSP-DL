from time import localtime, strftime


def time_str():
    return strftime('%H:%M:%S', localtime())


def pretty_print(*args):
    print(time_str() + ':', *args)


def log_tensorboard(writer, step, scalars=None, images=None, histograms=None, texts=None):
    if scalars is not None:
        for tag, value in scalars.items():
            writer.add_scalar(tag, value, step)
    if images is not None:
        for tag, image in images.items():
            writer.add_image(tag, image, step)
    if histograms is not None:
        for tag, value in histograms.items():
            writer.add_histogram(tag, value, step)
    if texts is not None:
        for tag, value in texts.items():
            writer.add_text(tag, value, step)
