import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    stride = image_size / feature_size
    anchors = []
    
    for i in range(feature_size):        # rows (y direction)
        for j in range(feature_size):    # cols (x direction)
            cx = (j + 0.5) * stride      # center x
            cy = (i + 0.5) * stride      # center y
            
            for s in scales:
                for r in aspect_ratios:
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    
                    anchors.append([
                        cx - w/2,   # x1
                        cy - h/2,   # y1
                        cx + w/2,   # x2
                        cy + h/2    # y2
                    ])
    
    return anchors
