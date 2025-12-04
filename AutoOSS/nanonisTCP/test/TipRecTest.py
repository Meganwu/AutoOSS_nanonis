# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 08:32:02 2025

@author: jced0001
"""

from nanonisTCP import nanonisTCP
from nanonisTCP.TipRec import TipRec
import traceback

"""
Set up the TCP connection and interface
"""
def run_test(TCP_IP='127.0.0.1', TCP_PORT=6501, debug=True, version=13520):
    # Listening Port: see Nanonis File>Settings>TCP Programming Interface
    NTCP = nanonisTCP(TCP_IP, TCP_PORT, version=version)                                # Nanonis TCP interface
    try:
        tipRec = TipRec(NTCP)                                                           # Nanonis Tip Recorder Module

        """
        Data Get
        """
        channel_indexes, data = tipRec.DataGet()
        if(debug):
            print("Channel indexes")
            print(channel_indexes)
            print("----------------------------------------------------------------------")

        """
        Data save
        """
        tipRec.DataSave(base_name="tip_move_test", clear_buffer=False)
        
        """
        Buffer Clear
        """
        tipRec.BufferClear()

        """
        Buffer size Set/Get
        """
        buffer_size = tipRec.BufferSizeGet()
        tipRec.BufferSizeSet(buffer_size+4)                                             # Get the buffer size
        if(debug):
            print("Buffer size")
            print(buffer_size + 4)
            print("----------------------------------------------------------------------")
        

    except:
        NTCP.close_connection()
        print(traceback.format_exc())
        return(traceback.format_exc())
    
    NTCP.close_connection()
    return "success"

run_test()