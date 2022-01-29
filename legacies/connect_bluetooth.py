import bluetooth, subprocess

def scan():
    print("Scanning for bluetooth devices ...")
    passkey = "1234"
    subprocess.call("kill -9 `pidof bluetooth-agent`", shell=True)

    devices = bluetooth.discover_devices(lookup_names = True, lookup_class = True)
    number_of_devices = len(devices)
    print(number_of_devices, " devices found")
    for addr, name, device_class in devices:
        print("\n")
        print("Device :")
        print("Device Name : %s " % (name))
        print("Device MAC Address : %s " % (addr))
        print("Device Class: %s " % (device_class))
        print("\n")


    # bd_addr = "00:18:E4:34:C4:57"
    # port = 1

    # status = subprocess.call("bluetooth-agent " + passkey + " &", shell=True)

    # try:
    #     sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    #     sock.connect((bd_addr, port))
    # except bluetooth.btcommon.BluetoothError as err:
    #     # Error handler
    #     pass

    # # sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    # # sock.connect((bd_addr, port))

    # print("Connected")
    # sock.settimeout(1.0)
    # sock.send("x")
    # print("Sent data")

    # data = sock.recv(1)
    # print("received [%s]" % data)

    # sock.close()
    # # sock.send("hello!!")
    # #
    # # sock.close()
    # return
scan()
