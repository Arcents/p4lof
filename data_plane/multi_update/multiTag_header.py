
from scapy.all import *
import sys, os

TYPE_TAG_FORWORD = 0x1212
TYPE_IPV4 = 0x0800

class MyTag(Packet):
    name = "MyTag"
    fields_desc = [
        BitField("bos", 0, 1),
        BitField("tag", 0, 7)
    ]
    def mysummary(self):
        return self.sprintf("bos=%bos%, tag=%tag%")


bind_layers(Ether, MyTag, type=TYPE_TAG_FORWORD)
bind_layers(MyTag, MyTag, bos=0)
bind_layers(MyTag, IP, bos=1)

