import socket
import time
import numpy as np
local_hosts = '10.7.34.150'
port = 20000
img_h = 720
img_w = 960
num_port = (img_h//30)*(img_w//40)
socket_port_all = {}
for i in range(num_port):
    socket_port_all[str(i)] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket_port_all[str(i)].bind((local_hosts, port+i+1))
    socket_port_all[str(i)].settimeout(0.001)
    socket_port_all[str(i)].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print(num_port)
while 1:
    for i in range(num_port):
        try:
            data = socket_port_all[str(i)].recv(4 * 65536)
        except:
            continue
        print('port is %d'%i)
        data = np.frombuffer(data, dtype=np.uint8)
        print(data.shape)
    print('aa')
    time.sleep(0.1)
