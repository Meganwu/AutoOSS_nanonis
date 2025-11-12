# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:40:21 2022

@author: jced0001
"""

class TipRec:
    """
    Nanonis Follow Me Module
    """
    def __init__(self, nanonisTCP):
        self.nanonisTCP = nanonisTCP

    def BufferSizeSet(self, size):
        """
        Sets the size of the tip position buffer.

        Parameters
        ----------
        size : int
            Size of the buffer in number of positions.
        """
        hex_rep = self.nanonisTCP.make_header('TipRec.BufferSizeSet', body_size=4)
        hex_rep += self.nanonisTCP.to_hex(size, 4)
        
        self.nanonisTCP.send_command(hex_rep)
        self.nanonisTCP.receive_response(0)

    def BufferSizeGet(self):
        """
        Returns the buffer size of the Tip Move Recorder.

        Auguments: None

        Return arguments (if Send response back flag is set to True when sending requiest message):

            -buffer size (int) is the number of data elements in the Tip Move Recorder
            -Error described in the Response message>Body section:           
        """
        hex_rep = self.nanonisTCP.make_header('TipRec.BufferSizeGet', body_size=0)
        
        self.nanonisTCP.send_command(hex_rep)
        
        # Receive Response
        response = self.nanonisTCP.receive_response(4)
        
        # Extract buffer size from response
        buffer_size = self.nanonisTCP.hex_to_int(response[0:4])
        
        return buffer_size
        
    def BufferClear(self):
        """
        Clears the tip position buffer.

        Auguments: None

        Return arguments (if Send response back flag is set to True when sending requiest message):

            -Error described in the Response message>Body section:           
        """
        hex_rep = self.nanonisTCP.make_header('TipRec.BufferClear', body_size=0)
        
        self.nanonisTCP.send_command(hex_rep)
        self.nanonisTCP.receive_response(0)

    def DataGet(self):
        """
        Returns the indexes and values of the channels acquired while the tip is moving in Follow Me mode (displaed in the Tip Move Recorder).

        Auguments: None

        Return arguments (if Send response back flag is set to True when sending requiest message):

            -Number of channels (int) is the number of recorded channels. It defines the size of the Channel indexes array
            -Channel indexes (int array) is an array of channel indexes that were recorded in the Tip Move Recorder. The index is comprised between 0 and 127,
            and it corresponds to the full list of signals avaialble in the system.
            To get the signal name and its corresponding index in the list of the 128 available signals in the Nanonis Controller, use the Signal.NamesGet function, or check the RT Idx value in the Signal Magager module.
            -Data rows (int) defines the number of rows of the Data array
            -Data columns (int) defines the number of columns of the Data array
            -Data (float 32) is a 2D array of recorded data while moving the tip
            -Error described in the Response message>Body section:
        """
        hex_rep = self.nanonisTCP.make_header('TipRec.DataGet', body_size=0)
        
        self.nanonisTCP.send_command(hex_rep)
        
        # Receive Response
        response = self.nanonisTCP.receive_response(0, True)
        
        # Extract data from response
        num_channels = self.nanonisTCP.hex_to_int(response[0:4])
        channel_indexes = [self.nanonisTCP.hex_to_int(response[i:i+4]) for i in range(4, 4 + num_channels * 4, 4)]
        data_rows = self.nanonisTCP.hex_to_int(response[4 + num_channels * 4:8 + num_channels * 4])
        data_columns = self.nanonisTCP.hex_to_int(response[8 + num_channels * 4:12 + num_channels * 4])
        
        data_start_index = 12 + num_channels * 4
        data_size = data_rows * data_columns * 4
        data = [self.nanonisTCP.hex_to_float32(response[i:i+4]) for i in range(data_start_index, data_start_index + data_size, 4)]
        
        return (num_channels, channel_indexes, data_rows, data_columns, data)
    
    def DataSave(self, clear_buffer, size, basename):
        """
        Saves the tip position buffer to a file.

        ParametersAuguments:
        ----------
        clear_buffer : bool
            If True, clears the buffer after saving.
        size : int
            Size of the data to save.
        basename : str
            Base name for the saved file.

        Returns
            -Error described in the Response message>Body section:
        """
        hex_rep = self.nanonisTCP.make_header('TipRec.DataSave', body_size=8 + len(basename))
        hex_rep += self.nanonisTCP.to_hex(int(clear_buffer), 4)
        hex_rep += self.nanonisTCP.to_hex(size, 4)
        hex_rep += self.nanonisTCP.string_to_hex(basename)
        
        self.nanonisTCP.send_command(hex_rep)
        self.nanonisTCP.receive_response(0)
        