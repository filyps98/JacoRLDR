import numpy as np

def body_swap(cube, cylinder):

    geom_id = 0
    if np.random.rand(1) > 0.5:

        target_xyz = cube.modify_xyz([0.075, 0.075, 0])
        target_orientation = cube.modify_euler([0, 0, 1.57])
        size = cube.modify_size([0.01, 0.01, 0.01])

        cylinder.isolate_object()
        geom_id = 2

        return geom_id, target_xyz, target_orientation, size


        

    else:
        target_xyz = cylinder.modify_xyz([0.075, 0.075, 0])
        target_orientation = cylinder.modify_euler([0, 0, 1.57])
        size = cylinder.modify_size([0.01, 0.01, 0])

        cube.isolate_object()
        geom_id = 3

        return geom_id, target_xyz.copy(), target_orientation, size
