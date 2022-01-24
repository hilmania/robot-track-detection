# Arduino Code

# int x;

# void setup() {
#   Serial.begin(115200);
#   Serial.setTimeout(1);
# }

# void loop() {
#   while (!Serial.available());
#   x = Serial.readString().toInt();
#   Serial.print(x + 1);
# }

import serial
import time

arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)


def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data


while True:
    num = input("Enter a number: ")
    value = write_read(num)
    print(value)
