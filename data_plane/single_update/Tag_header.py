
from scapy.all import *
import sys, os

TYPE_TAG_FORWORD = 0x1212
TYPE_IPV4 = 0x0800

class MyTag(Packet):
    name = "MyTag"
    fields_desc = [
        ShortField("protoID", 0),
        ShortField("tag", 0)
    ]
    def mysummary(self):
        return self.sprintf("protoID=%protoID%, tag=%tag%")


bind_layers(Ether, MyTag, type=TYPE_TAG_FORWORD)
bind_layers(MyTag, IP, protoID=TYPE_IPV4)

