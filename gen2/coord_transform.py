import numpy as np

def world_to_grid(wx, wy, center_x, center_y, scale, grid_size):
    """
    Transforms world coordinates (wx, wy) to grid coordinates (gx, gy).

    Args:
        wx: World X coordinate
        wy: World Y coordinate
        center_x: World X coordinate corresponding to the center of the grid
        center_y: World Y coordinate corresponding to the center of the grid
        scale: Scaling factor (pixels per world unit)
        grid_size: Size of the grid (width/height)

    Returns:
        (gx, gy) tuple of grid coordinates (int)
    """
    dx = wx - center_x
    dy = wy - center_y
    gx = int((dx * scale) + (grid_size / 2))
    gy = int((dy * scale) + (grid_size / 2))
    return gx, gy

def grid_to_world(gx, gy, center_x, center_y, scale, grid_size):
    """
    Transforms grid coordinates (gx, gy) to world coordinates (wx, wy).
    """
    dx = (gx - grid_size / 2) / scale
    dy = (gy - grid_size / 2) / scale
    wx = center_x + dx
    wy = center_y + dy
    return wx, wy
