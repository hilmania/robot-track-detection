import bluetooth
import time
# Detect all Bluetooth devices and Create an array with all the MAC addresses
print("Searching for devices...")
# nearby_
# Run through all the devices found and list their name
# num = 0
# print("Select your device by entering its coresponding number...")
# for i in nearby_devices:
#     num += 1
# print(str(num) + ": " + bluetooth.lookup_name(i))

# Allow the user to select their Arduino
# selection = int(input("> ")) - 1
bd_addr = "00:18:E4:34:C4:57"
port = 1
passkey = "1234"
# Show user selection
# print("You have selected " + bluetooth.lookup_name(nearby_devices[selection]))

# Connect to bluetooth address and port
try:
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    sock.connect((bd_addr, port))
    data = "L"
    sock.send(data)
    time.sleep(3)
    data = "Z"
    sock.send(data)
    data = sock.recv(1024)
    print(data)

except bluetooth.btcommon.BluetoothError as err:
    print(err)
    pass

# Print out appearsto be those of Serial.println and not bluetooth.println

sock.getsockname()
sock.getpeername()

sock.close()