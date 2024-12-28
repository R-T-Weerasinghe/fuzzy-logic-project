import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def create_football_fuzzy_system():
    # Input variables
    vertical_pos = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'vertical_position')
    horizontal_pos = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'horizontal_position')
    ball_size = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'ball_size')
    person_ball_ratio = ctrl.Antecedent(np.arange(0, 15.01, 0.1), 'person_ball_ratio')
    aspect_ratio = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'aspect_ratio')
    initial_confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'initial_confidence')

    # Output variable
    confidence = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'confidence')

    # Membership functions for vertical position
    vertical_pos['bottom'] = fuzz.trimf(vertical_pos.universe, [0, 0, 0.2])
    vertical_pos['mid_bottom'] = fuzz.trimf(vertical_pos.universe, [0.1, 0.2, 0.3])
    vertical_pos['middle'] = fuzz.trimf(vertical_pos.universe, [0.2, 0.5, 0.75])
    vertical_pos['mid_top'] = fuzz.trimf(vertical_pos.universe, [0.6, 0.8, 0.85])
    vertical_pos['top'] = fuzz.trimf(vertical_pos.universe, [0.8, 1, 1])

    # Membership functions for horizontal position
    horizontal_pos['left_edge'] = fuzz.trimf(horizontal_pos.universe, [0, 0, 0.3])
    horizontal_pos['middle'] = fuzz.trimf(horizontal_pos.universe, [0.2, 0.5, 0.8])
    horizontal_pos['right_edge'] = fuzz.trimf(horizontal_pos.universe, [0.7, 1, 1])

    # Membership functions for ball size (relative to frame height)
    ball_size['too_small'] = fuzz.trimf(ball_size.universe, [0, 0, 0.05])
    ball_size['good'] = fuzz.trimf(ball_size.universe, [0.02, 0.05, 0.40])
    ball_size['too_large'] = fuzz.trimf(ball_size.universe, [0.30, 1, 1])

    # Membership functions for person/ball ratio
    person_ball_ratio['low'] = fuzz.trimf(person_ball_ratio.universe, [0, 4, 4])
    person_ball_ratio['medium'] = fuzz.trimf(person_ball_ratio.universe, [3, 6, 9])
    person_ball_ratio['high'] = fuzz.trimf(person_ball_ratio.universe, [8, 15, 15])

    # Membership functions for aspect ratio (width/height)
    aspect_ratio['poor_small'] = fuzz.trimf(aspect_ratio.universe, [0, 0, 1])
    aspect_ratio['good'] = fuzz.trapmf(aspect_ratio.universe, [0.95, 0.97, 1.13, 1.15])
    aspect_ratio['poor_large'] = fuzz.trimf(aspect_ratio.universe, [1, 2, 2])

    # Membership functions for initial confidence
    initial_confidence['very_low'] = fuzz.trimf(initial_confidence.universe, [0, 0, 0.3])
    initial_confidence['low'] = fuzz.trimf(initial_confidence.universe, [0.2, 0.4, 0.5])
    initial_confidence['medium'] = fuzz.trimf(initial_confidence.universe, [0.4, 0.6, 0.8])
    initial_confidence['high'] = fuzz.trimf(initial_confidence.universe, [0.8, 0.9, 0.95])
    initial_confidence['very_high'] = fuzz.trimf(initial_confidence.universe, [0.85, 1, 1])


    # Membership functions for output confidence
    confidence['very_low'] = fuzz.trimf(confidence.universe, [0, 0, 0.25])
    confidence['low'] = fuzz.trimf(confidence.universe, [0.15, 0.35, 0.5])
    confidence['medium'] = fuzz.trimf(confidence.universe, [0.4, 0.6, 0.8])
    confidence['high'] = fuzz.trimf(confidence.universe, [0.7, 0.85, 0.95])
    confidence['very_high'] = fuzz.trimf(confidence.universe, [0.85, 1, 1])

    # Rules
    rules = [
        # # Vertical position rules
        ctrl.Rule(vertical_pos['top'], confidence['very_low']),
        ctrl.Rule(vertical_pos['mid_top'], confidence['low']),
        ctrl.Rule(vertical_pos['middle'], confidence['high']),
        ctrl.Rule(vertical_pos['mid_bottom'], confidence['high']),
        ctrl.Rule(vertical_pos['bottom'], confidence['very_low']),

        # Horizontal position rules
        ctrl.Rule(horizontal_pos['left_edge'] | horizontal_pos['right_edge'], confidence['very_low']),
        ctrl.Rule(horizontal_pos['middle'], confidence['high']),

        # Size rules
        # TODO: These rules are not good, need to be adjusted as these increases the confidence of incorrect ball objects
        # ctrl.Rule(ball_size['too_small'] | ball_size['too_large'], confidence['very_low']),
        # ctrl.Rule(ball_size['good'], confidence['medium']),

        # Person/ball ratio rules
        ctrl.Rule(person_ball_ratio['low'], confidence['very_low']),
        ctrl.Rule(person_ball_ratio['medium'], confidence['very_high']),
        ctrl.Rule(person_ball_ratio['high'], confidence['very_low']),

        # Aspect ratio rules
        ctrl.Rule(aspect_ratio['poor_small'] | aspect_ratio['poor_large'], confidence['very_low']),
        ctrl.Rule(aspect_ratio['good'], confidence['very_high']),

        # Initial confidence rules
        # ctrl.Rule(initial_confidence['high'] & ball_size['good'] & aspect_ratio['good'], confidence['very_high']),
        # ctrl.Rule(initial_confidence['low'] & (ball_size['too_small'] | ball_size['too_large']), confidence['very_low'])
        ctrl.Rule(initial_confidence['very_high'], confidence['very_high']),
        ctrl.Rule(initial_confidence['high'], confidence['high']),
        ctrl.Rule(initial_confidence['medium'], confidence['low']),
        ctrl.Rule(initial_confidence['low'], confidence['very_low']),
        ctrl.Rule(initial_confidence['very_low'], confidence['very_low'])
    ]

    # Create and return the control system
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def calculate_composite_confidence(initial_confidence, fuzzy_confidence):
    """
    Calculate the composite confidence score.
    
    Args:
        initial_confidence: Initial confidence score (0-1)
        fuzzy_confidence: Fuzzy system-adjusted confidence score (0-1)
    
    Returns:
        float: Composite confidence score (0-1)
    """
    # Weight for the fuzzy confidence
    # Ensure the fuzzy confidence is in the range [0, 1]
    fuzzy_confidence = max(0, min(1, fuzzy_confidence))
    
    # Calculate the composite confidence
    composite_confidence = initial_confidence + (fuzzy_confidence - 0.5) * 2 * (1 - initial_confidence)
    
    # Ensure the composite confidence is in the range [0, 1]
    composite_confidence = max(0, min(1, composite_confidence))
    return composite_confidence


def adjust_confidence(
    fuzzy_system,
    vertical_pos,
    horizontal_pos,
    ball_size,
    person_ball_ratio,
    aspect_ratio,
    initial_confidence
):
    """
    Adjust the confidence score using the fuzzy system.
    
    Args:
        fuzzy_system: The initialized fuzzy control system
        vertical_pos: Normalized vertical position (0-1, from top to bottom)
        horizontal_pos: Normalized horizontal position (0-1, from left to right)
        ball_size: Ball height relative to frame height (0-1)
        person_ball_ratio: Ratio of closest person height to ball height
        aspect_ratio: Ball bounding box width/height ratio
        initial_confidence: Initial YOLO confidence score (0-1)
    
    Returns:
        float: Adjusted confidence score (0-1)
    """
    # Input the values into the fuzzy system
    fuzzy_system.input['vertical_position'] = vertical_pos
    fuzzy_system.input['horizontal_position'] = horizontal_pos
    # fuzzy_system.input['ball_size'] = ball_size
    fuzzy_system.input['person_ball_ratio'] = person_ball_ratio
    fuzzy_system.input['aspect_ratio'] = aspect_ratio
    fuzzy_system.input['initial_confidence'] = initial_confidence
    
    # Compute the result
    try:
        fuzzy_system.compute()
        fuzzy_confidence = fuzzy_system.output['confidence']
        return calculate_composite_confidence(initial_confidence, fuzzy_confidence)
    except:
        # If computation fails (e.g., due to input out of bounds), return initial confidence
        # return initial_confidence
        raise ValueError("Computation failed in adjust_confidence function")

# Example usage
def process_yolo_detections(
    ball_detection,  # (x, y, w, h, conf)
    person_detections,  # list of (x, y, w, h, conf)
    image_height,
    image_width
):
    """
    Process YOLO detections and return adjusted confidence.
    
    Args:
        ball_detection: Tuple of (x, y, width, height, confidence) for ball
        person_detections: List of tuples (x, y, width, height, confidence) for persons
        image_height: Height of the image
        image_width: Width of the image
    
    Returns:
        float: Adjusted confidence score (0-1)
    """
    # Create fuzzy system
    fuzzy_system = create_football_fuzzy_system()
    
    # Extract ball information
    ball_x, ball_y, ball_w, ball_h, ball_conf = ball_detection
    
    # Calculate inputs for fuzzy system
    vertical_pos = ball_y / image_height
    horizontal_pos = ball_x / image_width
    ball_size = ball_h / image_height
    aspect_ratio = ball_w / ball_h if ball_h > 0 else 2
    print(aspect_ratio)
    
    # Find closest person and calculate ratio
    if person_detections:
        # TODO: Implement closest person to the ball calculation; this is wrong
        closest_person_h = max(person[3] for person in person_detections)
        person_ball_ratio = closest_person_h / ball_h if ball_h > 0 else 15
    else:
        person_ball_ratio = 6  # Default to medium ratio if no persons detected
    
    # Get adjusted confidence
    adjusted_conf = adjust_confidence(
        fuzzy_system,
        vertical_pos,
        horizontal_pos,
        ball_size,
        person_ball_ratio,
        aspect_ratio,
        ball_conf
    )
    
    return adjusted_conf