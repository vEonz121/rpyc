from regex import R
import rpyc as r
from rpyc.utils.server import ThreadedServer

class MyService(r.Service):
    ALIASES = ["floop", "bloop"]
    def on_connect(self, conn):
        print(f"Someone connected!")
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        print(f"Someone Disconnected!")
        pass

    def exposed_get_answer(self): # this is an exposed method
        return 42

    exposed_the_real_answer_though = 43     # an exposed attribute

    def exposed_get_question(self):  # while this method is not exposed
        return "what is the airspeed velocity of an unladen swallow?"
    

# t = ThreadedServer(MyService, port=18861)
# t.start()


mysvc = r.OneShotServer(service=MyService, port=18861, auto_register=True)
mysvc.start()


