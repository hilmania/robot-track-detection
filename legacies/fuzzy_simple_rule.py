import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

distance = ctrl.Antecedent(np.arange(0, 150, 1), 'distance')
uni_angle = np.array([-180, -30, 0, 30, 180])
angle = ctrl.Antecedent(uni_angle, 'angle')

uni_velocity = np.array([0, 55, 100, 175, 255])
velocity = ctrl.Consequent(uni_velocity, 'velocity')

distance['dekat'] = fuzz.trapmf(distance.universe, [0, 0, 5, 15])
distance['cukup'] = fuzz.trimf(distance.universe, [10, 20, 30])
distance['jauh'] = fuzz.trapmf(distance.universe, [20, 30, 200, 200])

distance.view()
plt.title("Distance Membership Function")
#
angle['kiri'] = fuzz.trapmf(angle.universe, [-180, -180, -30, 0])
angle['lurus'] = fuzz.trimf(angle.universe, [-30, 0, 30])
angle['kanan'] = fuzz.trapmf(angle.universe, [0, 30, 180, 180])

angle.view()
plt.title("Angle Membership Function")
#
velocity['stop'] = fuzz.trapmf(velocity.universe, [0, 0, 55, 100])
velocity['slow'] = fuzz.trimf(velocity.universe, [55, 100, 175])
velocity['fast'] = fuzz.trimf(velocity.universe, [100, 175, 255])

velocity.view()
plt.title("Velocity Membership Function")

# rule_roda_kanan

rule1_roda_kanan = ctrl.Rule(distance['dekat'] | angle['kiri'], velocity['stop'])
rule2_roda_kanan = ctrl.Rule(distance['dekat'] | angle['lurus'], velocity['stop'])
rule3_roda_kanan = ctrl.Rule(distance['dekat'] | angle['kanan'], velocity['slow'])

rule4_roda_kanan = ctrl.Rule(distance['cukup'] | angle['kiri'], velocity['stop'])
rule5_roda_kanan = ctrl.Rule(distance['cukup'] | angle['lurus'], velocity['slow'])
rule6_roda_kanan = ctrl.Rule(distance['cukup'] | angle['kanan'], velocity['slow'])

rule7_roda_kanan = ctrl.Rule(distance['jauh'] | angle['kiri'], velocity['slow'])
rule8_roda_kanan = ctrl.Rule(distance['jauh'] | angle['lurus'], velocity['fast'])
rule9_roda_kanan = ctrl.Rule(distance['jauh'] | angle['kanan'], velocity['fast'])

# # rule roda kiri
rule1_roda_kiri = ctrl.Rule(distance['dekat'] | angle['kiri'], velocity['slow'])
rule2_roda_kiri = ctrl.Rule(distance['dekat'] | angle['lurus'], velocity['stop'])
rule3_roda_kiri = ctrl.Rule(distance['dekat'] | angle['kanan'], velocity['stop'])

rule4_roda_kiri = ctrl.Rule(distance['cukup'] | angle['kiri'], velocity['slow'])
rule5_roda_kiri = ctrl.Rule(distance['cukup'] | angle['lurus'], velocity['slow'])
rule6_roda_kiri = ctrl.Rule(distance['cukup'] | angle['kanan'], velocity['stop'])

rule7_roda_kiri = ctrl.Rule(distance['jauh'] | angle['kiri'], velocity['fast'])
rule8_roda_kiri = ctrl.Rule(distance['jauh'] | angle['lurus'], velocity['fast'])
rule9_roda_kiri = ctrl.Rule(distance['jauh'] | angle['kanan'], velocity['slow'])

robot_ctrl_right = ctrl.ControlSystem(
    [rule1_roda_kanan, rule2_roda_kanan, rule3_roda_kanan, rule4_roda_kanan, rule5_roda_kanan,
    rule6_roda_kanan, rule7_roda_kanan, rule8_roda_kanan, rule9_roda_kanan]
)

robot_ctrl_left = ctrl.ControlSystem(
    [rule1_roda_kiri, rule2_roda_kiri, rule3_roda_kiri, rule4_roda_kiri, rule5_roda_kiri,
    rule6_roda_kiri, rule7_roda_kiri, rule8_roda_kiri, rule9_roda_kiri]
)

wheel_right = ctrl.ControlSystemSimulation(robot_ctrl_right)
wheel_left = ctrl.ControlSystemSimulation(robot_ctrl_left)

wheel_right.input['distance'] = 60
wheel_right.input['angle'] = 25

wheel_left.input['distance'] = 60
wheel_left.input['angle'] = 25

wheel_right.compute()
wheel_left.compute()

print("velocity_right_wheel : ", wheel_right.output['velocity'])
print("velocity_left_wheel : ", wheel_left.output['velocity'])
velocity.view(sim=wheel_right)
plt.title("Velocity Right Wheel")
velocity.view(sim=wheel_left)
plt.title("Velocity Left Wheel")

plt.show()
