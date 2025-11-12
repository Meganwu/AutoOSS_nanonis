#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 2024

@author: Nian Wu 
"""

from .nanonisTCP import nanonisTCP
from .Bias import Bias
from .Scan import Scan
from .Current import Current
from .FolMe import FolMe
from .SafeTip import SafeTip
from .ZController import ZController
from .AutoApproach import AutoApproach
from .Motor import Motor
from .LockIn import LockIn
from .Marks import Marks
from .Pattern import Pattern
from .BiasSpectr import BiasSpectr
from .Osci2T import Osci2T
from .TipShaper import TipShaper
from .UserOut import UserOut
from .Signals import Signals
from .TipRec import TipRec


class nanonisController(nanonisTCP):
    """
    Nanonis controller for all modules
    
    """    
    def __init__(self):
        self.nanonisTCP = nanonisTCP(version=13520) 
        self.Bias = Bias(self.nanonisTCP)
        self.Current = Current(self.nanonisTCP)
        self.FolMe = FolMe(self.nanonisTCP)
        self.SafeTip = SafeTip(self.nanonisTCP)
        self.ZController = ZController(self.nanonisTCP)
        self.AutoApproach = AutoApproach(self.nanonisTCP)
        self.Motor = Motor(self.nanonisTCP)
        self.LockIn = LockIn(self.nanonisTCP)
        self.Marks = Marks(self.nanonisTCP)
        self.Pattern = Pattern(self.nanonisTCP)
        self.BiasSpectr = BiasSpectr(self.nanonisTCP)
        self.Osci2T = Osci2T(self.nanonisTCP)
        self.TipShaper = TipShaper(self.nanonisTCP)
        self.UserOut = UserOut(self.nanonisTCP)
        self.Scan = Scan(self.nanonisTCP)
        self.Signals = Signals(self.nanonisTCP)
        self.TipRec = TipRec(self.nanonisTCP)

    


    



