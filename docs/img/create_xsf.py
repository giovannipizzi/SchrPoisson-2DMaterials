import numpy as np
import ase
import ase.io

x_positions_1 = np.arange(0.0,7.0,3.0)
x_positions_2 = np.arange(10.0,31.0,4.0)
x_positions_3 = np.arange(33.0, 41.0, 3.0)

x_pos = np.append(x_positions_1, x_positions_2)
x_pos = np.append(x_pos,x_positions_3)


cell = ase.Atoms('C'*x_pos.size,
                 pbc=True, 
                 cell = ( (x_pos[-1],0.,0.),
                          (0.,3.0,0.),
                          (0.,0.,15.0)))

cell.set_atomic_numbers([6, 6, 6,
                         16, 16, 16, 16, 16, 16, 
                         6, 6, 6])

pos = []
for i in range(x_pos.size):
    pos.append([x_pos[i],0.0,7.5])

cell.set_positions(pos)
ase.io.write("example.xsf",cell,"xsf")
