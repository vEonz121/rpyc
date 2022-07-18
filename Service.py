
import rpyc as r
from rpyc.utils.server import ThreadedServer


# run this first on the terminal
# python rpyc_registry.py -l true -t 500'



class MyService(r.Service):
    ALIASES= ['MY','FOO', 'BAR']
    def on_connect(self, conn):
        print(f"Someone connected!")
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        print(f"Someone Disconnected!")
        pass

    def exposed_get_answer(self, x): # this is an exposed method
        return f"{x} was the input"

    exposed_the_real_answer_though = 43     # an exposed attribute

    def exposed_get_question(self):  # while this method is not exposed
        return "what is the airspeed velocity of an unladen swallow?"
    




# Go to command prompt and get your ipv4 to start this server..
t = ThreadedServer(service=MyService,hostname='192.168.103.200', port=1856, auto_register=True, listener_timeout=14)
t.start()

