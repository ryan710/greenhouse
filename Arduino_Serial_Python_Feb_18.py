#sep8a: RH 65
#sep8b: rh65
#sep15: temp22
import serial
from datetime import datetime
import datetime
import time
import csv
import sys



start = time.time()
runtime = 310
serial_port = 'com3'
baud_rate = 9600

print(datetime.datetime.now())

ser = serial.Serial(serial_port, baud_rate)
with open(r"C:\\Users\\r\\Desktop\\opD\\feb_2021\\2.18\\feb18.csv", "a+") as f:     #### r"C:\\Users\\r\\Desktop\\logFiles\feb18.csv", "a+") as f:
    while True:
        if time.time() > start + runtime : sys.exit("end")
        
        init = (datetime.datetime.now().time())
        a = (init.hour * 60) + (init.minute) + (init.second / 60)
        
        line = ser.readline();
        line = line.decode("utf-8")
        f.writelines(["1000,"])
        
        end = (datetime.datetime.now().time())
        
        a = ((init.hour * 60) + (init.minute) + ((init.second) / (60)))
        
        b = ((end.hour * 60) + (end.minute) + (end.second / 60))

        c = b - a
        
        f.writelines([line.strip(), ",%s"%(a), ",%s"%(b), ",%s\n"%(c)])
        print(line);
        
        

    

