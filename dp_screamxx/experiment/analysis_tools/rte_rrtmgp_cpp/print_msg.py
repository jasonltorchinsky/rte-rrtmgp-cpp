# Standard Library Imports
from datetime import datetime

# Third-Party Library Imports

# Local imports

"""
Prints a message with a time-stampe.
"""
def print_msg(msg: str):
    current_time: str = datetime.now().strftime("%H:%M:%S")
    print("[{}]: {}".format(current_time, msg), flush = True)