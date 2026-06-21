# Standard Library Imports
from datetime import datetime

# Third-Party Library Imports

# Local Library Imports
from consts.dtypes import NP_INT
from consts.numeric import MPI_ROOT

def print_msg(msg: str, l_rank: NP_INT):
    if l_rank == MPI_ROOT:
        current_time: str = datetime.now().strftime("%H:%M:%S")
        out_msg: str = "[{}]: {}".format(current_time, msg)
        print(out_msg, flush = True)