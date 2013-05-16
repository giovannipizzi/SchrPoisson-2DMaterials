#from __future__ import division
import numpy as n
import scipy
import scipy.linalg


#hbar^2/m0 in units of eV*ang*ang
HBAR2OVERM0=7.61996163

is_periodic = True

# grid spacing in ang; this is exact, and layer lengths are adapted to keep a constant
# step. This allows the second-derivative hamiltonian matrix to be symmetric
delta_x = 0.1

# Small threshold to check for charge neutrality
small_threshold = 1.e-6

# List of known materials
mat_properties = {
    'mat1': {
        'gap': 1.,          # gap, in eV
        'val_offset': 0.,   # offset in eV of the valence band from some reference state
        'condmass': 0.8,     # effective mass in the conduction, in units of m0
        'valmass':  0.9,     # effective mass in the valence, in units of m0
        'eps': 10.,         # relative dielectric constant
        'ndoping': 0.,      # the doping linear density, in e/cm
        },
    'mat2': {
        'gap': 0.7,         # gap, in eV
        'val_offset': -0.1, # offset in eV of the valence band from some reference state
        'condmass': 1.,     # effective mass in the conduction, in units of m0
        'valmass':  1.,     # effective mass in the valence, in units of m0
        'eps': 14.,         # relative dielectric constant
        'ndoping': 0.#-5.e5,      # the doping linear density, in e/cm
        },
    'shell': {
        'gap': 9.,          # gap, in eV
        'val_offset': -4.,  # offset in eV of the valence band from some reference state
        'condmass': 1.,     # effective mass in the conduction, in units of m0
        'valmass':  1.,     # effective mass in the valence, in units of m0
        'eps': 8.,          # relative dielectric constant
        'ndoping': 0.,      # the doping linear density, in e/cm
        },
    'n_deltadoping': {
        'ndoping': 3.e5,     # If a layer has thickness zero, only the doping needs to be defined
        },
    'p_deltadoping': {
        'ndoping': -3.e5,     # If a layer has thickness zero, only the doping needs to be defined
        }
    }


# List of layers, and their thickness in angstrom
# For non-periodic boundary conditions
layers_np = [
    ('shell',50.),
    ('mat1',10.),
    ('n_deltadoping',0.),
    ('mat2',10.),
    ('p_deltadoping',0.),
    ('mat1',10.),
    ('n_deltadoping',0.),
    ('mat2',10.),
    ('p_deltadoping',0.),
    ('mat1',10.),
    ('n_deltadoping',0.),
    ('mat2',10.),
    ('p_deltadoping',0.),
    ('mat1',10.),
    ('n_deltadoping',0.),
    ('mat2',10.),
    ('p_deltadoping',0.),
    ('mat1',10.),
    ('shell',50.),
    ]

# List of layers, and their thickness in angstrom
# For periodic boundary conditions
layers_p = [
    ('mat1',5.),
    ('n_deltadoping',0.),
    ('mat2',10.),
    ('p_deltadoping',0.),
    ('mat1',5.),
    ]

if is_periodic:
    layers = layers_p
else:
    layers = layers_np


class Slab(object):
    def __init__(self,layers):
        """
        Pass a suitable xgrid (containing the sampling points in units of angstroms) and
        a (in angstroms)
        """
        if len(layers) == 0:
            raise ValueError("layers must have at least one layer")

        self.max_step_size = 1. # in eV
        self._slope = 0. # in V/ang

        total_length = 0.
        # I create a list where each element is a tuple in the format
        # (full_material_properties, end_x_ang)
        # (the first layer starts at x=0.
        self._layers_range = []
        xgrid_pieces = []
        materials = []
        for layer_idx, l in enumerate(layers):
            nintervals = int(n.ceil(l[1]/delta_x))
            # I want always the same grid spacing, so I possibly increase (slightly) the
            # thickness
            layer_length = nintervals * delta_x
            grid_piece = n.linspace(total_length, total_length+layer_length,nintervals+1)

            # Note: the following works also for l[1]==0 (delta dopings), returning a length-1
            # array that then becomes array([]) [i.e., an empty list] when stripping the first
            # point with [1:]
            # The first layer should not be a delta-doping layer: this is checked later.
            if layer_idx == 0:
                # For the first layer I do not remove the first point
                xgrid_pieces.append(grid_piece)
            else:
                # I remove the first point, it is in the previous piece
                xgrid_pieces.append(grid_piece[1:])

            materials.append(mat_properties[l[0]])        
            #print sum(len(i) for i in xgrid_pieces[:-1]), sum(len(i) for i in xgrid_pieces)
            total_length += layer_length

        self._xgrid = n.concatenate(xgrid_pieces)

        # A check that all steps are equal; I calculate the error of each step w.r.t. the
        # expected step delta-x
        steps_error = (self._xgrid[1:] - self._xgrid[:-1]) - delta_x
        if abs(steps_error).max() > 1.e-10:
            raise AssertionError("The steps should be all equal to delta_x, but they aren't! "
                                 "max is: {}".format(abs(steps_error).max()))

        # Dielectric constants
        self._epsilon = n.zeros(len(self._xgrid))
        last_idx = 0
        for mat, grid_piece in zip(materials, xgrid_pieces):
            if len(grid_piece)!=0: # skip delta dopings
                self._epsilon[last_idx:last_idx+len(grid_piece)] = mat['eps']
            last_idx += len(grid_piece)

        # Conduction band and valence band profiles, in eV
        # effective masses, in units of the free electron mass
        self._condband = n.zeros(len(self._xgrid))
        self._valband = n.zeros(len(self._xgrid))
        self._valmass = n.zeros(len(self._xgrid))
        self._condmass = n.zeros(len(self._xgrid))
        last_idx = 0
        for mat, grid_piece in zip(materials, xgrid_pieces):
            if len(grid_piece)!=0: # skip delta dopings
                self._valband[last_idx:last_idx+len(grid_piece)] = mat['val_offset']
                self._condband[last_idx:last_idx+len(grid_piece)] = mat['val_offset'] + mat['gap']
                self._valmass[last_idx:last_idx+len(grid_piece)] = mat['valmass']
                self._condmass[last_idx:last_idx+len(grid_piece)] = mat['condmass']
            last_idx += len(grid_piece)

        # Doping; I also count total free holes and free electrons. In e/cm
        self._doping = n.zeros(len(self._xgrid))
        ## TODO CHANGE we do not need total holes and total electrons; only their difference is to be
        ## kept constant; we choose that we are at equilibrium and only 1 fermi energy in needed, and
        ## we find it by a bisection algorithm, until ne-nh = required value (that can be inferred from
        ## sum(self._doping)
        last_idx = 0
        for mat, grid_piece in zip(materials, xgrid_pieces):

            if len(grid_piece)!=0: # Finite thickness layer
                # The doping is distributed over the len(grid_piece) lines
                self._doping[last_idx:last_idx+len(grid_piece)] = mat['ndoping']/len(grid_piece)
            else:   # it is a delta doping
                if last_idx == 0:
                    raise ValueError("You cannot put a delta doping layer as the very first layer")
                self._doping[last_idx-1] += mat['ndoping']
            last_idx += len(grid_piece)

        # electrostatic potential in eV
        self._V = n.zeros(len(self._xgrid))

    def get_required_net_free_charge(self):
        """
        Return the net free charge needed (in e/cm) to compensate the doping.
        """
        # Minus sign because if I have a n-doping (self._doping > 0), I need a negative
        # charge to compensate
        return -n.sum(self._doping)

    def update_V(self,c_states, v_states, e_fermi, zero_elfield=True):
        """
        Both free_el_density and free_holes_density should be positive
        
        Return True upon convergence
        """
        import schrpoisson_wire

        V_conv_threshold = 1.e-4

        free_electrons_density = get_electron_density(c_states, e_fermi, self._condmass, self.npoints)
        free_holes_density = get_hole_density(v_states, e_fermi, self._valmass, self.npoints)

        total_charge_density =  self._doping - free_electrons_density + free_holes_density
        if is_periodic:
            new_V = schrpoisson_wire.v_of_rho_periodic(
                self._xgrid, total_charge_density, self._epsilon)
        else:
            new_V = schrpoisson_wire.v_of_rho_nonperiodic(
                self._xgrid, total_charge_density, self._epsilon)

        if zero_elfield:
            # in V/ang
            self._slope = (new_V[-1] - new_V[0])/(self._xgrid[-1] - self._xgrid[0])
            new_V -= self._slope*self._xgrid
        else:
            self._slope = 0.

        step = new_V - self._V
        # The following introduces steps in the curve.. discard
        # Keep some history: I don't move more than max_step_size, point by point
        #absstep=abs(step)
        #absstep[absstep>max_step_size] = max_step_size
        #self._V += n.sign(new_V - self._V) * absstep
            
        current_max_step_size = abs(step).max()
        print 'convergence param:', current_max_step_size
        # If current_max_step_size < max_step_size, I just add 'step', i.e.
        # self._V becomes new_V.
        # Otherwise, I rescale such that the maximum movement is max_step_size, globally
        if current_max_step_size != 0:
            self._V += step * min(self.max_step_size, current_max_step_size) / current_max_step_size

        #        self.max_step_size *= 0.9

        return current_max_step_size < V_conv_threshold
        
    def get_V(self):
        """
        Return the electrostatic potential in eV
        """
        return self._V

    def get_xgrid(self):
        """
        Return the x grid, in angstrom
        """
        return self._xgrid

    @property
    def npoints(self):
        return len(self._xgrid)

    def get_conduction_profile(self):
        """
        Return the conduction band profile (in absence of electric fields and/or charge bendings), in eV
        """
        return self._condband + self._V

    def get_valence_profile(self):
        """
        Return the conduction band profile (in absence of electric fields and/or charge bendings), in eV
        """
        return self._valband + self._V

def get_electron_density(c_states, e_fermi, c_mass_array, npoints):
    """
    Fill subbands with a 1D dos, at T=0 (for now; T>0 requires numerical integration)
    The first index of c_states must be the state energy in eV
    e_fermi in eV
    c_mass is array with the conduction effective mass (on the grid)
        in units of the free electron mass

    Return linear electron density, in e/cm
    """
    # The 1D DOS is (including the factor of 2 for the spin):
    # g(E) = sqrt(2 * effmass)/(pi*hbar) * 1/sqrt(E-E0)
    # where effmass is the band effective mass, E0 is the band edge.
    #
    # I rewrite it as g(E) = D * (meff/m0) / sqrt(E-E0)
    # where (meff/m0) is simply the effective mass in units of the electron free mass,
    # and D=sqrt(2) / pi / sqrt(HBAR2OVERM0) and will be in units of 1/ang/sqrt(eV)
    D = n.sqrt(2.) / n.pi / sqrt(HBAR2OVERM0)

    el_density = n.zeros(npoints)   
    for state_energy, state in c_states:
        if state_energy > e_fermi:
            continue
        square_norm = sum((state)**2)
        # I average the inverse of the effective mass
        # Both state and c_mass_array should have the same length
        averaged_eff_mass = 1./(sum(state**2 / c_mass_array) / sqrt(square_norm))

        # At T=0, integrating from E0 to Ef the DOS gives
        # D * meff * int_E0^Ef 1/(sqrt(E-E0)) dE =
        # D * meff * 2 * sqrt(Ef-E0)   [if Ef>E0, else zero]
        el_density += D * averaged_eff_mass * 2. * sqrt(e_fermi - state_energy) * (
            state**2 / square_norm)

    # Up to now, el_density is in 1/ang; we want it in 1/cm
    return el_density * 1.e8

def get_hole_density(v_states, e_fermi, v_mass_array, npoints):
    """
    For all documentation and comments, see get_electron_density
    
    v_mass_array should contain the mass of holes, i.e., positive

    Return a positive number
    """
    D = n.sqrt(2.) / n.pi / sqrt(HBAR2OVERM0)

    h_density = n.zeros(npoints)  
    for state_energy, state in v_states:
        # Note that here the sign is opposite w.r.t. the conduction case
        if state_energy < e_fermi:
            continue
        square_norm = sum((state)**2)
        averaged_eff_mass = 1./(sum(state**2 / v_mass_array) / sqrt(square_norm))

        h_density += D * averaged_eff_mass * 2. * sqrt(state_energy - e_fermi) * (
            state**2 / square_norm)

    return h_density * 1.e8

def find_efermi(c_states, v_states, c_mass_array, v_mass_array, npoints, target_net_free_charge):
    """
    Pass the conduction and valence states (and energies), 
    the conduction and valence mass arrays, 
    and the net charge to be used as a target (positive means that we want holes than electrons)
    """
    more_energy = 2. # in eV
    # TODO: we may want a precision also on the charge, not only on the energy
    energy_precision = 1.e-6 # eV

    all_states_energies = n.array([s[0] for s in c_states] + [s[0] for s in v_states])
    # I set the boundaries for the bisection algorithm; I could in principle
    # also extend these ranges
    ef_l = all_states_energies.min()-more_energy
    ef_r = all_states_energies.max()+more_energy

    electrons_l = n.sum(get_electron_density(c_states, ef_l, c_mass_array,npoints))
    holes_l = n.sum(get_hole_density(v_states, ef_l, v_mass_array, npoints))
    electrons_r = n.sum(get_electron_density(c_states, ef_r, c_mass_array, npoints))
    holes_r = n.sum(get_hole_density(v_states, ef_r, v_mass_array,npoints))
    
    net_l = holes_l - electrons_l
    net_r = holes_r - electrons_r
    if (net_l - target_net_free_charge) * (
        net_r - target_net_free_charge) > 0:
        raise ValueError("The net charge at the boundary of the bisection algorithm "
                         "range has the same sign! {}, {}, target={}; ef_l={}, ef_r={}".format(
                             net_l, net_r, target_net_free_charge,ef_l, ef_r))

    while (ef_r - ef_l) > energy_precision:
        ef = (ef_l + ef_r)/2.
        electrons = n.sum(get_electron_density(c_states, ef, c_mass_array,npoints))
        holes = n.sum(get_hole_density(v_states, ef, v_mass_array,npoints))
        net = holes - electrons
        if (net - target_net_free_charge) * (
          net_r - target_net_free_charge) > 0:
            net_r = net
            ef_r = ef
        else:
            net_l = net
            ef_l = ef

    return (ef_r + ef_l)/2.

def get_conduction_states_p(slab):
    """
    This function diagonalizes the matrix using the fact that the matrix is banded. This provides
    a huge speedup w.r.t. to a dense matrix.

    NOTE: _p stands for the periodic version
    """
    more_bands_energy = 0. # How many eV to to above the top of the conduction band

    # Ham matrix in eV; I store only the first two diagonals
    # H[2,:] is the diagonal,
    # H[1,1:] is the first upper diagonal,
    # H[0,2:] is the second upper diagonal
    H = n.zeros((3, slab.npoints))

    # Two arrays to map the old order of the matrix with the new one, and vice versa
    rangearray = n.arange(slab.npoints)
    # this is to be used as index when rebuilding the wavefunctions
    reordering = n.zeros(slab.npoints,dtype=int)
    reordering[rangearray <= (slab.npoints-1)//2] = (2 * rangearray)[rangearray <= (slab.npoints-1)//2]
    reordering[rangearray > (slab.npoints-1)//2] = (
    2 * slab.npoints - 2 * rangearray - 1)[rangearray > (slab.npoints-1)//2]
    #    # I don't know if I ever need this
    #    inverse_reordering = n.zeros(slab.npoints,dtype=int)
    #    inverse_reordering[::2] = (rangearray//2)[::2]
    #    inverse_reordering[1::2] = (slab.npoints - (rangearray+1)//2)[1::2]


    # Given the index i in the not-reordered matrix, for the matrix element (i-1)--(i),
    # return the indices in the 2 upper lines of the reordered, banded matrix.
    # Notes:
    # * the (i-1)--(i) matrix element is the i-th element of the superdiagonal array
    #   defined later
    # * applying reordering, we get that in the reordered matrix, the
    #   (i-1)--(i) element should occupy the j1--j2 matrix element.
    # * I resort j1, j2 such that j1 < j2 to fill the upper diagonal
    #   (NOTE! if this was a complex hermitian matrix, when reorderning I should also take
    #    the complex conjugate. Here the matrix is real and symmetric and I don't have to worry)
    # * j2-j1 can only be 1 or 2, if the reordering is correct
    # * if the j2-j1==1, I have to fill H[1,:], else H[0,:]: the first index is 2-(j2-j1)
    # * the second index, also due to the fact of that the superdiagonals are stored
    #   in H[0,2:] and H[1,1:] (see docs of scipy.linalg.eig_banded), is simply j2
    #
    # * What happens to the element 1-N (the one that gives periodicity) where N=matrixsize?
    #   * If I store it in superdiagonal[0], when calculating j1_list, for that element
    #     I get reordering[-1] == reordering[slab.npoints-1], that is
    #     what we want.
    j1_list = reordering[n.arange(slab.npoints)-1]
    j2_list = reordering[n.arange(slab.npoints)]
    temp_list = j2_list.copy()
    indices_to_swap = [j2_list<j1_list]
    j2_list[indices_to_swap] = j1_list[indices_to_swap]
    j1_list[indices_to_swap] = temp_list[indices_to_swap]
    reordering_superdiagonals = ( 2 - (j2_list - j1_list) , j2_list )
    # A check
    if any((j2_list - j1_list) < 1) or any((j2_list - j1_list) > 2):
        raise AssertionError("Wrong indices difference in j1/2_lists (cond)!")

    # On the diagonal, the potential: sum of the electrostatic potential + conduction band profile
    # This is obtained with get_conduction_profile()
    H[2,:][reordering] = slab.get_conduction_profile()
    min_energy = H[2,:].min()
    max_energy = H[2,:].max() + more_bands_energy

    # The zero-th element contains the periodicity term
    mass_differences = n.zeros(slab.npoints)
    mass_differences[0] = (slab._condmass[0] + slab._condmass[-1])/2.    
    mass_differences[1:] = (slab._condmass[1:] + slab._condmass[:-1])/2.
    
    # Finite difference method for 2nd derivatives. Remember that the equation with an effective
    # mass is:
    # d/dx ( 1/m(x) d/dx psi(x)), i.e. 1/m(x) goes inside the first derivative
    # I set the coefficient for the 2nd derivative on the diagonal
    # mass_differences[1] is the average mass between 0 and 1
    H[2,:][reordering]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
    # mass_differences[1+1] is the average mass between 1 and 2
    H[2,:][reordering]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences[
        (arange(slab.npoints)+1) % slab.npoints]

    # note! The matrix is symmetric only if the mesh step is identical for all steps
    # I use 1: in the second index because the upper diagonal has one element less, and
    # this is the format required by eig_banded
    # Note that this also sets superdiagonal[0] to the correct element that should be at
    # the corner of the matrix, at position (n-1, 0)
    superdiagonal = - (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
    H[reordering_superdiagonals] = superdiagonal

    w, v = scipy.linalg.eig_banded(H, lower=False, eigvals_only=False, 
                                   overwrite_a_band=True, # May enhance performance
                                   select='v', select_range=(min_energy,max_energy),
                                   max_ev=slab.npoints) # max_ev: max # of eigenvalues to expect
                                                        # I use the worst case scenario, where I get
                                                        # all of them
    result_to_reorder = zip(w, v.T)
    return tuple((w, v[reordering]) for w, v in result_to_reorder)


def get_valence_states_p(slab):
    """
    See discussion in get_conduction_states, plus comments here for what has changed
    
    NOTE: _p stands for the non-periodic version
    """
    more_bands_energy = 0. # How many eV to to below the bottom of the valence band

    H = n.zeros((3, slab.npoints))

    rangearray = n.arange(slab.npoints)
    reordering = n.zeros(slab.npoints,dtype=int)
    reordering[rangearray <= (slab.npoints-1)//2] = (2 * rangearray)[rangearray <= (slab.npoints-1)//2]
    reordering[rangearray > (slab.npoints-1)//2] = (
    2 * slab.npoints - 2 * rangearray - 1)[rangearray > (slab.npoints-1)//2]

    j1_list = reordering[n.arange(slab.npoints)-1]
    j2_list = reordering[n.arange(slab.npoints)]
    temp_list = j2_list.copy()
    indices_to_swap = [j2_list<j1_list]
    j2_list[indices_to_swap] = j1_list[indices_to_swap]
    j1_list[indices_to_swap] = temp_list[indices_to_swap]
    reordering_superdiagonals = ( 2 - (j2_list - j1_list) , j2_list )
    if any((j2_list - j1_list) < 1) or any((j2_list - j1_list) > 2):
        raise AssertionError("Wrong indices difference in j1/2_lists (valence)!")

    H[2,:][reordering] = slab.get_valence_profile()
    min_energy = H[2,:].min() - more_bands_energy
    max_energy = H[2,:].max() 

    # In the valence bands, it is as if the mass is negative
    mass_differences = n.zeros(slab.npoints)
    mass_differences[0]  = -(slab._valmass[0] + slab._valmass[-1])/2.    
    mass_differences[1:] = -(slab._valmass[1:] + slab._valmass[:-1])/2.
    
    H[2,:][reordering]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
    H[2,:][reordering]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences[
        (arange(slab.npoints)+1) % slab.npoints]

    superdiagonal = - (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
    H[reordering_superdiagonals] = superdiagonal

    w, v = scipy.linalg.eig_banded(H, lower=False, eigvals_only=False, 
                                   overwrite_a_band=True, # May enhance performance
                                   select='v', select_range=(min_energy,max_energy),
                                   max_ev=slab.npoints) # max_ev: max # of eigenvalues to expect
                                                        # I use the worst case scenario, where I get
                                                        # all of them
    result_to_reorder = zip(w, v.T)
    return tuple((w, v[reordering]) for w, v in result_to_reorder)

def get_conduction_states_np(slab):
    """
    This function diagonalizes the matrix using the fact that the matrix is banded. This provides
    a huge speedup w.r.t. to a dense matrix.

    NOTE: _np stands for the non-periodic version
    """
    more_bands_energy = 0. # How many eV to to above the top of the conduction band

    # Ham matrix in eV; I store only the first upper diagonal
    # H[1,:] is the diagonal,
    # H[0,1:] is the first upper diagonal
    H = n.zeros((2, slab.npoints))

    # On the diagonal, the potential: sum of the electrostatic potential + conduction band profile
    # This is obtained with get_conduction_profile()
    H[1,:] = slab.get_conduction_profile()
    min_energy = H[1,:].min()
    max_energy = H[1,:].max() + more_bands_energy

    mass_differences = (slab._condmass[1:] + slab._condmass[:-1])/2.

    # Finite difference method for 2nd derivatives. Remember that the equation with an effective
    # mass is:
    # d/dx ( 1/m(x) d/dx psi(x)), i.e. 1/m(x) goes inside the first derivative
    # I set the coefficient for the 2nd derivative on the diagonal
    H[1,1:] += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
    H[1,:-1]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences

    # note! The matrix is symmetric only if the mesh step is identical for all steps
    # I use 1: in the second index because the upper diagonal has one element less, and
    # this is the format required by eig_banded
    H[0,1:] = - (HBAR2OVERM0/2.) / delta_x**2 / mass_differences

    w, v = scipy.linalg.eig_banded(H, lower=False, eigvals_only=False, 
                                   overwrite_a_band=True, # May enhance performance
                                   select='v', select_range=(min_energy,max_energy),
                                   max_ev=slab.npoints) # max_ev: max # of eigenvalues to expect
                                                        # I use the worst case scenario, where I get
                                                        # all of them
    return zip(w, v.T)

def get_valence_states_np(slab):
    """
    See discussion in get_conduction_states, plus comments here for what has changed

    NOTE: _np stands for the non-periodic version
    """
    more_bands_energy = 0. # How many eV to to below the bottom of the valence band

    H = n.zeros((2, slab.npoints))

    H[1,:] = slab.get_valence_profile()
    min_energy = H[1,:].min() - more_bands_energy
    max_energy = H[1,:].max() 

    # In the valence bands, it is as if the mass is negative
    mass_differences = -(slab._valmass[1:] + slab._valmass[:-1])/2.

    H[1,1:] += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
    H[1,:-1]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences

    H[0,1:] = - (HBAR2OVERM0/2.) / delta_x**2 / mass_differences

    w, v = scipy.linalg.eig_banded(H, lower=False, eigvals_only=False, 
                                   overwrite_a_band=True, # May enhance performance
                                   select='v', select_range=(min_energy,max_energy),
                                   max_ev=slab.npoints) # max_ev: max # of eigenvalues to expect
                                                        # I use the worst case scenario, where I get
                                                        # all of them
    return zip(w, v.T)
  

if __name__ == "__main__":
    from pylab import *

    zoom_factor = 10. # To plot eigenstates
    max_steps = 50   # max number of steps of self-consistency

    print 'starting the code'
    slab = Slab(layers=layers)

    print 'slab created'

    try:
        converged = False
        for iteration in range(max_steps):
            print 'starting iteration {}...'.format(iteration)
            if is_periodic:
                c_states = get_conduction_states_p(slab)   
                v_states = get_valence_states_p(slab)
            else:
                c_states = get_conduction_states_np(slab)   
                v_states = get_valence_states_np(slab)
            
            e_fermi = find_efermi(c_states, v_states, slab._condmass, slab._valmass,
                                  npoints=slab.npoints,
                                  target_net_free_charge=slab.get_required_net_free_charge())
            print iteration, e_fermi

            zero_elfield = is_periodic
            converged = slab.update_V(c_states, v_states, e_fermi, zero_elfield=zero_elfield)
            # slab._slope is in V/ang; the factor to bring it to V/cm
            print 'Added E field: {} V/cm '.format(slab._slope * 1.e8) 
            if converged:
                break
    except KeyboardInterrupt:
        pass

    if not converged:
        print "****** WARNING! Calculation not converged ********"

    el_density = get_electron_density(c_states, e_fermi, slab._condmass, slab.npoints)
    hole_density = get_hole_density(v_states, e_fermi, slab._valmass,slab.npoints)
    print 'required net free charge={}'.format(slab.get_required_net_free_charge())
    print 'int. el. density: ', n.sum(el_density)
    print 'int. hole density:', n.sum(hole_density)

    plot(slab.get_xgrid(), slab.get_valence_profile(),'b',linewidth=2)
    plot(slab.get_xgrid(), slab.get_conduction_profile(),'r',linewidth=2)

    for w, v in c_states:
        plot(slab.get_xgrid(), w + zoom_factor * abs(v)**2,'r')
    for w, v in v_states:
        # Plot valence bands upside down
        plot(slab.get_xgrid(), w - zoom_factor * abs(v)**2,'b')
    
    plot(slab.get_xgrid(), ones(slab.npoints) * e_fermi, '--')
    plot(slab.get_xgrid(), slab.get_V(),'k')
    xlabel("x (ang)")
    ylabel("eV")

    ## For debugging purposes
    figure(2)
    #    plot(slab.get_xgrid(), slab._doping,label='d')
    plot(slab.get_xgrid(), el_density,label='e')
    plot(slab.get_xgrid(), hole_density,label='h')
    #    plot(slab.get_xgrid(), slab._epsilon)
    legend()
    
    xlabel("x (ang)")
    ylabel("el/hole density")
    show()
    
    
