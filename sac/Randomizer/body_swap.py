import numpy as np

def body_swap(cube, cylinder):

    if np.random.rand(1) > 0.5:

        target_xyz = cube.modify_xyz([0.05, 0.05, 0])
        target_orientation = cube.modify_euler([0, 0, 1.57])
        size = cube.modify_size([0.01, 0.01, 0.01])

        cylinder.isolate_object()

        

    else:
        target_xyz = cylinder.modify_xyz([0.05, 0.05, 0])
        target_orientation = cylinder.modify_euler([0, 0, 1.57])
        size = cylinder.modify_size([0.01, 0.01, 0])

        cube.isolate_object()

    return target_xyz, target_orientation, size
