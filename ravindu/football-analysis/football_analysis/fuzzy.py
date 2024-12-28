import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# fuzzy input variables
confidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'confidence')
size = ctrl.Antecedent(np.arange(0, 5000, 100), 'size')
position_y = ctrl.Antecedent(np.arange(0, 1080, 0.1), 'position_y')

# define fuzzy output variable 
improved_confidence = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'improved_confidence')

# fuzzy membership functions for confidence
confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.5])
confidence['medium'] = fuzz.trimf(confidence.universe, [0.2, 0.5, 0.8])
confidence['high'] = fuzz.trimf(confidence.universe, [0.5, 1, 1])

# fuzzy membership functions for size
size['small'] = fuzz.trimf(size.universe, [0, 0, 1500])
size['medium'] = fuzz.trimf(size.universe, [1000, 2500, 4000])
size['large'] = fuzz.trimf(size.universe, [3000, 5000, 5000])

# fuzzy membership functions for position_y
position_y['low'] = fuzz.trimf(position_y.universe, [0, 0, 360])
position_y['medium'] = fuzz.trimf(position_y.universe, [300, 540, 780])
position_y['high'] = fuzz.trimf(position_y.universe, [720, 1080, 1080])

# fuzzy membership functions for improved_confidence
improved_confidence['low'] = fuzz.trimf(improved_confidence.universe, [0, 0, 0.5])
improved_confidence['medium'] = fuzz.trimf(improved_confidence.universe, [0.2, 0.5, 0.8])
improved_confidence['high'] = fuzz.trimf(improved_confidence.universe, [0.5, 1, 1])

# fuzzy rules
rule1 = ctrl.Rule(confidence['high'] & size['small'] & position_y['high'], improved_confidence['high'])
rule2 = ctrl.Rule(confidence['medium'] & size['medium'] & position_y['mid'], improved_confidence['medium'])
rule3 = ctrl.Rule(confidence['low'] & size['small'] & position_y['high'], improved_confidence['medium'])
rule4 = ctrl.Rule(confidence['low'] & size['large'], improved_confidence['low'])

rules = [rule1, rule2, rule3, rule4]

# control system
improved_confidence_ctrl = ctrl.ControlSystem(rules)
improved_confidence_sim = ctrl.ControlSystemSimulation(improved_confidence_ctrl)

# function to calculate improved confidence
def calculate_improved_confidence(confidence_val, bbox_width, bbox_height, position_y_val):
    box_area = bbox_width * bbox_height
    improved_confidence_sim.input['confidence'] = confidence_val
    improved_confidence_sim.input['size'] = box_area
    improved_confidence_sim.input['position_y'] = position_y_val
    improved_confidence_sim.compute()
    return improved_confidence_sim.output['improved_confidence']
