import struct
import socket
import errno

class CommunicationBase(object):
    def __init__(self, ip, port):