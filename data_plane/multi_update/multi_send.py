#!/usr/bin/env python
import argparse
import sys
import socket
import random
import struct
import argparse

from scapy.all import sendp, send, get_if_list, get_if_hwaddr, hexdump
from scapy.all import Packet
from scapy.all import Ether, IP, UDP, TCP
from myTunnel_header import MyTag

tag_switches=[1,2]

def get_if():
    ifs=get_if_list()
    iface=None # "h1-eth0"
    for i in get_if_list():
        if "ens33" in i:
            iface=i
            break;
    if not iface:
        print "Cannot find eth0 interface"
        exit(1)
    return iface

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip_addr', type=str, help="The destination IP address to use")
    parser.add_argument('message', type=str, help="The message to include in packet")
    # parser.add_argument('--tag', type=int, default=None, help='The myTunnel tag to use, if unspecified then myTunnel header will not be included in packet')
    args = parser.parse_args()

    addr = socket.gethostbyname(args.ip_addr)
    # tag = args.tag
    iface = get_if()


    i = 0
    pkt =  Ether(src=get_if_hwaddr(iface), dst='ff:ff:ff:ff:ff:ff')
    for t in tag_switches:
        try:
            pkt = pkt / MyTag(bos=0, tag=int(t))
            i = i+1
        except ValueError:
            pass    
    if pkt.haslayer(MyTag):
        pkt.getlayer(MyTag, i).bos = 1

    pkt = pkt / IP(dst=addr) / args.message

    # if (tag is not None):
    #     # print "sending on interface {} to tag {}".format(iface, str(tag))
    #     pkt =  Ether(src=get_if_hwaddr(iface), dst='ff:ff:ff:ff:ff:ff')
    #     pkt = pkt / MyTag(tag=tag) / IP(dst=addr) / args.message
    # else:
    #     print "sending on interface {} to IP addr {}".format(iface, str(addr))
    #     pkt =  Ether(src=get_if_hwaddr(iface), dst='ff:ff:ff:ff:ff:ff')
    #     pkt = pkt / IP(dst=addr) / TCP(dport=1234, sport=random.randint(49152,65535)) / args.message

    pkt.show2()
#    hexdump(pkt)
#    print "len(pkt) = ", len(pkt)
    sendp(pkt, iface=iface, verbose=False)


if __name__ == '__main__':
    main()
