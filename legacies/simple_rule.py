def fuzzy_robot(d, a):
    if d > 5:
        if -30 <= a < 30:
            VR = 150
            VL = 150
        elif -180 <= a < -30:
            VR = 30
            VL = 150
        elif 30 <= a <= 180:
            VR = 150
            VL = 30
        else:
            VR = 0
            VL = 0
    else:
        VR = 0
        VL = 0

    vel = []
    vel.append(VR)
    vel.append(VL)
    return vel


hasil_VR = fuzzy_robot(60, -30)
print(hasil_VR)