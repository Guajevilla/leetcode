# -*- coding: UTF-8 -*-
# 文件名：server.py

import socket  # 导入 socket 模块
import cv2
from PIL import Image
import numpy as np
import threading
import time
def get_massage(socket_model):
    try:
        data = socket_model.recv(65536)
    except:
        print('time out')
        return False
    return data

def receive_from_port(socket_model,img,index,img_w):
    data = get_massage(socket_model)
    hang = index//(img_w//160)
    lie = index%(img_w//160)
    #print('hang = %d lie = %d'%(hang,lie))
    #print(img.shape)
    if data == False:
        return
    else:
        data = np.frombuffer(data, dtype=np.uint8)
        data.resize(120, 160, 3)
        img[hang*120:hang*120+120,lie*160:lie*160+160] = data

def receive_from_port_check(socket_model):
    data = get_massage(socket_model)
    if data == False or data != bytes('start', encoding='utf-8'):
        s.sendto(bytes('start', encoding='utf-8'), ((target_hosts, 23000)))
        return
    if data == bytes('end frame', encoding='utf-8'):
        exit(0)

class myThread(threading.Thread):
    def __init__(self, threadID, name,img_w, *function_input):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.function_input = function_input
        self.img_w = img_w
    def run(self):
        receive_from_port(self.function_input[0],self.function_input[1],self.function_input[2],self.img_w)

class Thread_check(threading.Thread):
    def __init__(self, threadID, name,*function_input):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.function_input = function_input
    def run(self):
        receive_from_port_check(self.function_input[0])

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建 socket 对象
local_hosts = '10.7.34.150'
target_hosts = '10.7.34.0'
port = 23000  # 设置端口
s.bind((local_hosts, port))  # 绑定端口
s.settimeout(0.01)
start_flag = 0
while 1:
    # init
    data = get_massage(s)
    if start_flag == 0:
        if data == bytes('start frame',encoding='utf-8'):
            start_flag = 1
        else:
            continue
    if start_flag == 1:
        s.sendto(bytes('img_h', encoding='utf-8'), ((target_hosts, 23000)))
        data = get_massage(s)
        if data == False:
            continue
        data = str(data,encoding="utf-8")
        initial = data.split('_')[0]+'_'+data.split('_')[1]
        if initial != 'img_h':
            continue
        img_h = int(data.split('_')[-1])
        start_flag = 2
    if start_flag == 2:
        s.sendto(bytes('img_w', encoding='utf-8'), ((target_hosts, 23000)))
        data = get_massage(s)
        if data == False:
            continue
        data = str(data, encoding="utf-8")
        initial = data.split('_')[0]+'_'+data.split('_')[1]
        if initial != 'img_w':
            continue
        img_w = int(data.split('_')[-1])
        start_flag = 3
    if start_flag == 3:
        break

print('finish init')
img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

num_port = (img_h//120*img_w//160)
socket_port_all = {}
for i in range(num_port):
    socket_port_all[str(i)] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket_port_all[str(i)].bind((local_hosts, port+i+1))
    socket_port_all[str(i)].settimeout(1)

flag_finish = False
while flag_finish == False:
    hang = 0
    lie = 0
    thread_check = Thread_check(0, 'check' + str(0),s)
    thread_check.start()
    # print('show')
    index = 0
    for hang in range(0, img_h, 120):
        for lie in range(0, img_w, 160):
            thread = myThread(index, 'receive' + str(index),img_w,
                              socket_port_all[str(index)],img,index)
            thread.start()
            index+=1
    #print('wait')
    while 1:
        run_num = threading.activeCount()
        print(run_num)
        if run_num <= 3:
            break
        else:
            time.sleep(0.01)
    thread_check.join()
    print('show')
    cv2.imshow('a',img)
    cv2.waitKey(1)
    #img = img*0

s.close()

# import cv2
#
# cv2.imshow('a',cv2.imread("/media/kun/_Program/leetcode/misc/show3d.png"))
# cv2.waitKey(0)
