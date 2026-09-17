# Standard Library Imports
from datetime import datetime

# Third-Party Library Imports

# Local Library Imports
from consts.dtypes import NP_INT

def print_msg(msg: str, l_rank: NP_INT):
    current_time: str = datetime.now().strftime("%H:%M:%S")
    out_msg: str = "[{}]: [{}]: {}".format(current_time, l_rank, msg)
    print(out_msg, flush = True)