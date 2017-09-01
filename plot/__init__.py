import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = False

import pyglet
pyglet.options['debug_gl'] = False


# XXX: on windows, this strategy works with 1 window at a time
import time
import traceback
def inputhook(context):
    try:
        while not context.input_is_ready() and pyglet.app.windows:
            pyglet.clock.tick()
            for w in pyglet.app.windows:
                w.switch_to()
                w.dispatch_events()
                w.dispatch_event('on_draw')
                try:
                    w.flip()
                except AttributeError:
                    pass
                time.sleep(0.001) # performance knob: lower values yield better
                                  # responsiveness, but also increase CPU loading
                                  # NOTE: the default value (0.001) is rather
                                  # aggressive, so make sure to close your figures
                                  # when you don't need them
    except AttributeError as e:
        # skip bug on linux when closing a window
        if e.args[0] == ("'NoneType' object has no "
                         "attribute '_fbconfig'"):
            pass
        else:
            raise
    except Exception:
        # we should never get here; try catching the exception earlier
        print('plot.__init__.inputhook :: '
              'uncaught exception, closing all windows')
        traceback.print_exc()
        for w in pyglet.app.windows:
            w.dispatch_event('on_close')
    return 0

import IPython
IPython.terminal.pt_inputhooks.register('pyglet2', inputhook)
