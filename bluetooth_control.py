import serial
import time
print("Starting...")
port = "/dev/tty.HC-05-DevB"
bluetooth = serial.Serial(port, 9600)  
print("Connected!")  
bluetooth.flushInput()  
for i in range(5):  
    print("ping")  
    bluetooth.Write(b"Boop"+str.encode(str(i)))  
    input_data = bluetooth.readline()  
    print(input_data.decode())  
    time.sleep(0.1)  
bluetooth.close()  
print("Done")  