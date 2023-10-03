import logging
import logging.handlers


def log_receiver(queue):
    root = logging.getLogger()
    fh = logging.FileHandler('runner.log', 'w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    f = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s')
    fh.setFormatter(f)
    ch.setFormatter(f)
    root.addHandler(fh)
    root.addHandler(ch)

    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            import sys, traceback
            print('Logging exception:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def configure_log_producer(logger, queue):
    h = logging.handlers.QueueHandler(queue)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False
