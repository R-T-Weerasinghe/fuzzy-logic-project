Fuzzy rules can be static or dynamic analysis rules.

Static rules only consider the current frame/image and do not take into account the history of the frames.

### Static rules

#### ball position rules
- if ball is in the bottom 1/2 of the frame, ball confidence high
- if ball is in top 1/5 of the frame, then ball confidence low
- if ball is in the middle rest of the frame, then ball confidence medium

- if ball in left or right 1/8 of the frame, then ball confidence low
- if ball in middle 3/4 of the frame, then ball confidence high

#### person-ball ratio rules
- if closest person to the ball and ratio of the ball size to the person is low, then ball confidence is low (person/ball is less than 3)
- if closest person to the ball and ratio of the ball size to the person is medium, then ball confidence is high (person/ball is around 5)
- if closest person to the ball and ratio of the ball size to the person is high, then ball confidence is medium (person/ball is more than 7)

#### person-ball confidence rules
- if person confidence is high and ball confidence is high, then ball confidence is high
- if person confidence is low and ball confidence is low, then ball confidence is high
- if person confidence is high and ball confidence is low, then ball confidence is low
- if person confidence is low and ball confidence is high, then ball confidence is high

#### ball size rules
- If ball size is unusually large (>1/10 of frame height) -> confidence low
- If ball size is unusually small (<1/50 of frame height) -> confidence low

#### Aspect ratio rules
- If ball bounding box aspect ratio deviates >20% from 1:1 -> confidence low (footballs should appear roughly circular)
- If ball bounding box aspect ratio is within 10% of 1:1 -> increase confidence


### Dynamic rules

Dynamic rules consider the history of the frames and can be used to detect changes in the scene. (i.e. tracking)