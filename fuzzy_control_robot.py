import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

distance = ctrl.Antecedent(np.arange(0, 150, 1), 'distance')
uni_angle = np.array([-180, -90, -30, 0, 30, 90, 180])
angle = ctrl.Antecedent(uni_angle, 'angle')

uni_velocity = np.array([0, 55, 105, 155, 205, 255])
velocity = ctrl.Consequent(uni_velocity, 'velocity')

distance['sangat_dekat'] = fuzz.trapmf(distance.universe, [0, 0, 25, 50])
distance['dekat'] = fuzz.trimf(distance.universe, [25, 50, 75])
distance['cukup'] = fuzz.trimf(distance.universe, [50, 75, 100])
distance['jauh'] = fuzz.trimf(distance.universe, [75, 100, 125])
distance['sangat_jauh'] = fuzz.trapmf(distance.universe, [100, 125, 150, 150])

distance.view()

angle['sangat_kiri'] = fuzz.trapmf(angle.universe, [-180, -180, -90, -30])
angle['kiri'] = fuzz.trimf(angle.universe, [-90, -30, 0])
angle['lurus'] = fuzz.trimf(angle.universe, [-30, 0, 30])
angle['kanan'] = fuzz.trimf(angle.universe, [0, 30, 90])
angle['sangat_kanan'] = fuzz.trapmf(angle.universe, [30, 90, 180, 180])

angle.view()

velocity['stop'] = fuzz.trapmf(velocity.universe, [0, 0, 55, 105])
velocity['very_slow'] = fuzz.trimf(velocity.universe, [55, 105, 155])
velocity['slow'] = fuzz.trimf(velocity.universe, [105, 155, 205])
velocity['normal'] = fuzz.trimf(velocity.universe, [155, 205, 255]) 
velocity['fast'] = fuzz.trimf(velocity.universe, [205, 255, 255])

velocity.view()

plt.show()
