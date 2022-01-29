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
plt.title("Distance Membership Function")

angle['sangat_kiri'] = fuzz.trapmf(angle.universe, [-180, -180, -90, -30])
angle['kiri'] = fuzz.trimf(angle.universe, [-90, -30, 0])
angle['lurus'] = fuzz.trimf(angle.universe, [-30, 0, 30])
angle['kanan'] = fuzz.trimf(angle.universe, [0, 30, 90])
angle['sangat_kanan'] = fuzz.trapmf(angle.universe, [30, 90, 180, 180])

angle.view()
plt.title("Angle Membership Function")

velocity['stop'] = fuzz.trapmf(velocity.universe, [0, 0, 55, 105])
velocity['very_slow'] = fuzz.trimf(velocity.universe, [55, 105, 155])
velocity['slow'] = fuzz.trimf(velocity.universe, [105, 155, 205])
velocity['normal'] = fuzz.trimf(velocity.universe, [155, 205, 255]) 
velocity['fast'] = fuzz.trimf(velocity.universe, [205, 255, 255])

velocity.view()
plt.title("Velocity Membership Function")

# rule_roda_kanan
rule1_roda_kanan = ctrl.Rule(distance['sangat_dekat'] | angle['sangat_kiri'], velocity['stop'])
rule2_roda_kanan = ctrl.Rule(distance['sangat_dekat'] | angle['kiri'], velocity['very_slow'])
rule3_roda_kanan = ctrl.Rule(distance['sangat_dekat'] | angle['lurus'], velocity['very_slow'])
rule4_roda_kanan = ctrl.Rule(distance['sangat_dekat'] | angle['kanan'], velocity['slow'])
rule5_roda_kanan = ctrl.Rule(distance['sangat_dekat'] | angle['sangat_kanan'], velocity['very_slow'])

rule6_roda_kanan = ctrl.Rule(distance['dekat'] | angle['sangat_kiri'], velocity['very_slow'])
rule7_roda_kanan = ctrl.Rule(distance['dekat'] | angle['kiri'], velocity['very_slow'])
rule8_roda_kanan = ctrl.Rule(distance['dekat'] | angle['lurus'], velocity['slow'])
rule9_roda_kanan = ctrl.Rule(distance['dekat'] | angle['kanan'], velocity['slow'])
rule10_roda_kanan = ctrl.Rule(distance['dekat'] | angle['sangat_kanan'], velocity['slow'])

rule11_roda_kanan = ctrl.Rule(distance['cukup'] | angle['sangat_kiri'], velocity['slow'])
rule12_roda_kanan = ctrl.Rule(distance['cukup'] | angle['kiri'], velocity['slow'])
rule13_roda_kanan = ctrl.Rule(distance['cukup'] | angle['lurus'], velocity['normal'])
rule14_roda_kanan = ctrl.Rule(distance['cukup'] | angle['kanan'], velocity['normal'])
rule15_roda_kanan = ctrl.Rule(distance['cukup'] | angle['sangat_kanan'], velocity['normal'])

rule16_roda_kanan = ctrl.Rule(distance['jauh'] | angle['sangat_kiri'], velocity['slow'])
rule17_roda_kanan = ctrl.Rule(distance['jauh'] | angle['kiri'], velocity['normal'])
rule18_roda_kanan = ctrl.Rule(distance['jauh'] | angle['lurus'], velocity['normal'])
rule19_roda_kanan = ctrl.Rule(distance['jauh'] | angle['kanan'], velocity['normal'])
rule20_roda_kanan = ctrl.Rule(distance['jauh'] | angle['sangat_kanan'], velocity['fast'])

rule21_roda_kanan = ctrl.Rule(distance['sangat_jauh'] | angle['sangat_kiri'], velocity['normal'])
rule22_roda_kanan = ctrl.Rule(distance['sangat_jauh'] | angle['kiri'], velocity['normal'])
rule23_roda_kanan = ctrl.Rule(distance['sangat_jauh'] | angle['lurus'], velocity['normal'])
rule24_roda_kanan = ctrl.Rule(distance['sangat_jauh'] | angle['kanan'], velocity['fast'])
rule25_roda_kanan = ctrl.Rule(distance['sangat_jauh'] | angle['sangat_kanan'], velocity['fast'])

# rule roda kiri
rule1_roda_kiri = ctrl.Rule(distance['sangat_dekat'] | angle['sangat_kiri'], velocity['very_slow'])
rule2_roda_kiri = ctrl.Rule(distance['sangat_dekat'] | angle['kiri'], velocity['slow'])
rule3_roda_kiri = ctrl.Rule(distance['sangat_dekat'] | angle['lurus'], velocity['slow'])
rule4_roda_kiri = ctrl.Rule(distance['sangat_dekat'] | angle['kanan'], velocity['very_slow'])
rule5_roda_kiri = ctrl.Rule(distance['sangat_dekat'] | angle['sangat_kanan'], velocity['stop'])

rule6_roda_kiri = ctrl.Rule(distance['dekat'] | angle['sangat_kiri'], velocity['slow'])
rule7_roda_kiri = ctrl.Rule(distance['dekat'] | angle['kiri'], velocity['slow'])
rule8_roda_kiri = ctrl.Rule(distance['dekat'] | angle['lurus'], velocity['normal'])
rule9_roda_kiri = ctrl.Rule(distance['dekat'] | angle['kanan'], velocity['very_slow'])
rule10_roda_kiri = ctrl.Rule(distance['dekat'] | angle['sangat_kanan'], velocity['very_slow'])

rule11_roda_kiri = ctrl.Rule(distance['cukup'] | angle['sangat_kiri'], velocity['slow'])
rule12_roda_kiri = ctrl.Rule(distance['cukup'] | angle['kiri'], velocity['normal'])
rule13_roda_kiri = ctrl.Rule(distance['cukup'] | angle['lurus'], velocity['normal'])
rule14_roda_kiri = ctrl.Rule(distance['cukup'] | angle['kanan'], velocity['slow'])
rule15_roda_kiri = ctrl.Rule(distance['cukup'] | angle['sangat_kanan'], velocity['slow'])

rule16_roda_kiri = ctrl.Rule(distance['jauh'] | angle['sangat_kiri'], velocity['slow'])
rule17_roda_kiri = ctrl.Rule(distance['jauh'] | angle['kiri'], velocity['normal'])
rule18_roda_kiri = ctrl.Rule(distance['jauh'] | angle['lurus'], velocity['slow'])
rule19_roda_kiri = ctrl.Rule(distance['jauh'] | angle['kanan'], velocity['slow'])
rule20_roda_kiri = ctrl.Rule(distance['jauh'] | angle['sangat_kanan'], velocity['normal'])

rule21_roda_kiri = ctrl.Rule(distance['sangat_jauh'] | angle['sangat_kiri'], velocity['normal'])
rule22_roda_kiri = ctrl.Rule(distance['sangat_jauh'] | angle['kiri'], velocity['normal'])
rule23_roda_kiri = ctrl.Rule(distance['sangat_jauh'] | angle['lurus'], velocity['slow'])
rule24_roda_kiri = ctrl.Rule(distance['sangat_jauh'] | angle['kanan'], velocity['normal'])
rule25_roda_kiri = ctrl.Rule(distance['sangat_jauh'] | angle['sangat_kanan'], velocity['fast'])


# rule_roda_kiri
robot_ctrl_right = ctrl.ControlSystem(
    [rule1_roda_kanan, rule2_roda_kanan, rule3_roda_kanan, rule4_roda_kanan, rule5_roda_kanan,
    rule6_roda_kanan, rule7_roda_kanan, rule8_roda_kanan, rule9_roda_kanan, rule10_roda_kanan,
    rule11_roda_kanan, rule12_roda_kanan, rule13_roda_kanan, rule14_roda_kanan, rule15_roda_kanan,
    rule16_roda_kanan, rule17_roda_kanan, rule18_roda_kanan, rule19_roda_kanan, rule20_roda_kanan,
    rule21_roda_kanan, rule22_roda_kanan, rule23_roda_kanan, rule24_roda_kanan, rule25_roda_kanan]
)

robot_ctrl_left = ctrl.ControlSystem(
    [rule1_roda_kiri, rule2_roda_kiri, rule3_roda_kiri, rule4_roda_kiri, rule5_roda_kiri,
    rule6_roda_kiri, rule7_roda_kiri, rule8_roda_kiri, rule9_roda_kiri, rule10_roda_kiri,
    rule11_roda_kiri, rule12_roda_kiri, rule13_roda_kiri, rule14_roda_kiri, rule15_roda_kiri,
    rule16_roda_kiri, rule17_roda_kiri, rule18_roda_kiri, rule19_roda_kiri, rule20_roda_kiri,
    rule21_roda_kiri, rule22_roda_kiri, rule23_roda_kiri, rule24_roda_kiri, rule25_roda_kiri]
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
