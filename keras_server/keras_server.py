import socket
import errno
import logging
from logging import handlers
import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras import applications
from keras.models import Model, load_model
from keras.layers import Convolution2D,Conv2D, Dense, Input, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate,Activation,GlobalAveragePooling2D
import skimage.io # pyinstaller can't compile excute file, so it's require to convert by pil
import skimage.transform

def setLog(appname):
    logger = logging.getLogger(appname)
    log_level  = logging.INFO
    fileMaxByte = 1024 * 1024 * 100 #100MB
    filename = logger.name + ".log"
    fileHandler = logging.handlers.RotatingFileHandler(filename
                                                       , maxBytes=fileMaxByte
                                                       , backupCount=10)
    streamHandler = logging.StreamHandler()
    fomatter = logging.Formatter('[%(asctime)s]%(message)s')
    fileHandler.setFormatter(fomatter)
    streamHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    logger.setLevel(log_level)
    return logger

def get_test_set(path, tta=1, crop_size = 224, mean = None):
    x_batch = [] 
    try:
        img = skimage.io.imread(path)
    except:
        return None

    if mean is not None:
        if img.shape != mean.shape:
            img = skimage.transform.resize(img,mean.shape, mode = 'constant', preserve_range=True).astype(np.uint8)
            logger.warn("mean shape is " + str(mean.shape) + ", img shape is " + str(img.shape) + "so, reisize img to mean size")
        img = np.array(img, dtype=float)
        img =  img - mean


    w =  img.shape[1]
    h =  img.shape[0]
    x = (w-crop_size)//2
    y = (h-crop_size)//2
    centor = img[y:y+crop_size,x:x+crop_size]

    lt = img[0:crop_size, 0:crop_size]
    rt = img[0:crop_size, w-crop_size:w]
    rb = img[h-crop_size:h, w-crop_size:w]
    lb = img[h-crop_size:h, 0:crop_size]
    if tta ==1:
        x_batch.append(centor)
    elif tta == 4:
        r90 = np.rot90(centor)
        r180 = np.rot90(r90)
        r270 = np.rot90(r180)
        x_batch.append(centor)
        x_batch.append(r90)
        x_batch.append(r180)
        x_batch.append(r270)
    elif tta == 5:
        x_batch.append(centor)
        x_batch.append(lt)
        x_batch.append(rt)
        x_batch.append(rb)
        x_batch.append(lb)
        
    x_batch = np.array(x_batch)
    return x_batch

def Main():
    host = "localhost"
    port = 12345
    app_name = "keras_server"
    model_path = ""
    mean_file_path = ""
    tta_num = 5

    logger = setLog(app_name)
    while True: #connect wait 
        mySocket = socket.socket()
        mySocket.bind((host,port))
        mySocket.listen(1)
        logger.info("Connection wait : " + host + ':' + str(port))
        conn, addr = mySocket.accept()
        logger.info("Connection from: " + str(addr))

        while True: #message wait
            try:
                data = conn.recv(1024).decode()
            except ConnectionResetError as e:
                if e.errno == errno.ECONNRESET:
                    logger.error("disconnected client")
                else:
                    logger.error("unkown error")
                break
            
            if not data:
                 break
            logger.info("receive message : " + str(data))
            receive_string = str(data)
            input_para_list = receive_string.split(',')
            # want message format : image_path, model_path, crop_size, mean_file_path
            # sample : "Z:/image/TRUE_615_5397_0t_002.jpg
            # ,Z:/model/xception.h5,512,Z:/model/mean.npy"
            if model_path != input_para_list[1]:
                model_path  = input_para_list[1]
                model = load_model(model_path)
            if mean_file_path != input_para_list[3]:
                mean_file_path = input_para_list[3]
                mean_arr = np.load(mean_file_path)

            crop_size = int(input_para_list[2])
            img_path = input_para_list[0]
            in_test_set = get_test_set(img_path, crop_size=crop_size, tta = tta_num,mean=mean_arr)
            if in_test_set is None:
                continue
            probs = model.predict( in_test_set,batch_size=tta_num)
            probs = np.average(probs, axis = 0)
            y_pred = np.argmax(probs)
            y_prob = probs[y_pred]
            
           # message format : cls_num, prob, image_path
            send_message = str(y_pred) + "," + str(y_prob) + "," +img_path
            logger.info("sending message: " + send_message)
            try:
                conn.send(send_message.encode())
            except ConnectionResetError as e:
                if e.errno == errno.ECONNRESET:
                    logger.error("disconnected client")
                else:
                    logger.error("unkown error")
                break

        conn.close()
     
if __name__ == '__main__':
    Main()