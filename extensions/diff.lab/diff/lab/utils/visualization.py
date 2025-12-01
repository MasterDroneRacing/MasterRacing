import numpy as np
import cv2
def visualize_depth(depth_map, max_depth=10.0, colored=False):
    depth_clipped = np.clip(depth_map, 0, max_depth)
    depth_normalized = (depth_clipped / max_depth * 255).astype(np.uint8)
    if colored:
        colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        # colored_depth = cv2.applyColorMap(255 - depth_normalized, cv2.COLORMAP_JET)
        return colored_depth
    else:
        return depth_normalized