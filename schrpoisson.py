import numpy as n
import scipy
import scipy.linalg

#hbar^2/m0 in units of eV*ang*ang
HBAR2OVERM0=7.61996163

# grid spacing in ang; this is exact, and layer lengths are adapted to keep a constant
# step. This allows the second-derivative hamiltonian matrix to be symmetric
delta_x = 0.1

# Thickness of the slabs along z, in angstrom
# Change a, with fixed linrho, and converge this parameter for a->0
a = 0.001

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
        'ndoping': 0.,      # the doping linear density, in e/cm
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
        'ndoping': 1.e5,     # If a layer has thickness zero, only the doping needs to be defined
        },
    'p_deltadoping': {
        'ndoping': -1.e5,     # If a layer has thickness zero, only the doping needs to be defined
        }
    }


# List of layers, and their thickness in angstrom
layers = [
    ('shell',100.),
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
    ('shell',100.),
    ]

class Slab(object):
    def __init__(self,layers,a):
        """
        Pass a suitable xgrid (containing the sampling points in units of angstroms) and
        a (in angstroms)
        """
        self._a = a

        if len(layers) == 0:
            raise ValueError("layers must have at least one layer")

        self.max_step_size = 1. # in eV

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

        V_conv_threshold = 1.e-2

        free_electrons_density = get_electron_density(c_states, e_fermi, self._condmass)
        free_holes_density = get_hole_density(v_states, e_fermi, self._valmass)

        total_charge_density =  self._doping - free_electrons_density + free_holes_density
        new_V = schrpoisson_wire.v_of_rho(self._xgrid, total_charge_density, self._epsilon, self._a)

        if zero_elfield:
            slope = (new_V[-1] - new_V[0])/(self._xgrid[-1] - self._xgrid[0])
            new_V -= slope*self._xgrid

        step = new_V - self._V
        # The following introduces steps in the curve.. discard
        # Keep some history: I don't move more than max_step_size, point by point
        #absstep=abs(step)
        #absstep[absstep>max_step_size] = max_step_size
        #self._V += n.sign(new_V - self._V) * absstep
            
        current_max_step_size = abs(step).max()
        # If current_max_step_size < max_step_size, I just add 'step', i.e.
        # self._V becomes new_V.
        # Otherwise, I rescale such that the maximum movement is max_step_size, globally
        self._V += step * min(self.max_step_size, current_max_step_size) / current_max_step_size

        self.max_step_size *= 0.9

        return abs(step).max() < V_conv_threshold
        
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

def get_electron_density(c_states, e_fermi, c_mass_array):
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

    el_density = 0.   
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
        el_density += D * averaged_eff_mass * 2. * sqrt(e_fermi - state_energy)

    # Up to now, el_density is in 1/ang; we want it in 1/cm
    return el_density * 1.e8

def get_hole_density(v_states, e_fermi, v_mass_array):
    """
    For all documentation and comments, see get_electron_density
    
    v_mass_array should contain the mass of holes, i.e., positive

    Return a positive number
    """
    D = n.sqrt(2.) / n.pi / sqrt(HBAR2OVERM0)

    h_density = 0.   
    for state_energy, state in v_states:
        # Note that here the sign is opposite w.r.t. the conduction case
        if state_energy < e_fermi:
            continue
        square_norm = sum((state)**2)
        averaged_eff_mass = 1./(sum(state**2 / v_mass_array) / sqrt(square_norm))

        h_density += D * averaged_eff_mass * 2. * sqrt(state_energy - e_fermi)

    return h_density * 1.e8

def find_efermi(c_states, v_states, c_mass_array, v_mass_array, target_net_free_charge):
    """
    Pass the conduction and valence states (and energies), 
    the conduction and valence mass arrays, 
    and the net charge to be used as a target (positive means that we want holes than electrons)
    """
    # TODO: we may want a precision also on the charge, not only on the energy
    energy_precision = 1.e-6 # eV

    all_states_energies = n.array([s[0] for s in c_states] + [s[0] for s in v_states])
    # I set the boundaries for the bisection algorithm; I could in principle
    # also extend these ranges
    ef_l = all_states_energies.min()
    ef_r = all_states_energies.max()

    electrons_l = get_electron_density(c_states, ef_l, c_mass_array)
    holes_l = get_hole_density(v_states, ef_l, v_mass_array)
    electrons_r = get_electron_density(c_states, ef_r, c_mass_array)
    holes_r = get_hole_density(v_states, ef_r, v_mass_array)
    
    net_l = holes_l - electrons_l
    net_r = holes_r - electrons_r
    if (net_l - target_net_free_charge) * (
        net_r - target_net_free_charge) > 0:
        raise ValueError("The net charge at the boundary of the bisection algorithm "
                         "range has the same sign!")

    while (ef_r - ef_l) > energy_precision:
        ef = (ef_l + ef_r)/2.
        electrons = get_electron_density(c_states, ef, c_mass_array)
        holes = get_hole_density(v_states, ef, v_mass_array)
        net = holes - electrons
        if (net - target_net_free_charge) * (
          net_r - target_net_free_charge) > 0:
            net_r = net
            ef_r = ef
        else:
            net_l = net
            ef_l = ef

    return (ef_r + ef_l)/2.

def get_conduction_states(slab):
    """
    This function diagonalizes the matrix using the fact that the matrix is banded. This provides
    a huge speedup w.r.t. to a dense matrix.
    """
    more_bands_energy = 0. # How many eV to to above the conduction band edge

    # Ham matrix in eV; I store only the diagonal (H[1,:] is the diagonal, and H[0,1:] is the upper diagonal)
    H = n.zeros((2, slab.npoints))

    # On the diagonal, the potential: sum of the electrostatic potential + conduction band profile
    H[1,:] = slab._condband + slab._V
    min_energy = H[1,:].min()
    max_energy = H[1,:].max() + more_bands_energy

    mass_differences = (slab._condmass[n.arange(slab.npoints)[1:]] + 
                        slab._condmass[n.arange(slab.npoints)[:-1]])/2.

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

def get_valence_states(slab):
    more_bands_energy = 0. # How many eV to to above the conduction band edge

    # Ham matrix in eV; I store only the diagonal (H[1,:] is the diagonal, and H[0,1:] is the upper diagonal)
    H = n.zeros((2, slab.npoints))

    # On the diagonal, the potential: sum of the electrostatic potential + conduction band profile
    H[1,:] = slab._valband + slab._V
    min_energy = H[1,:].min() - more_bands_energy
    max_energy = H[1,:].max() 

    # In the valence bands, it is as if the mass is negative
    mass_differences = -(slab._valmass[n.arange(slab.npoints)[1:]] + 
                         slab._valmass[n.arange(slab.npoints)[:-1]])/2.

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
  

if __name__ == "__main__":
    from pylab import *

    zoom_factor = 10. # To plot eigenstates
    max_steps = 50   # max number of steps of self-consistency

    slab = Slab(layers=layers, a=a)

    try:
        converged = False
        for iteration in range(max_steps):
            c_states = get_conduction_states(slab)   
            v_states = get_valence_states(slab)
            
            e_fermi = find_efermi(c_states, v_states, slab._condmass, slab._valmass, 
                                  target_net_free_charge=slab.get_required_net_free_charge())
            print iteration, e_fermi
            
            converged = slab.update_V(c_states, v_states, e_fermi)
            if converged:
                break
    except KeyboardInterrupt:
        pass

    if not converged:
        print "****** WARNING! Calculation not converged ********"

    print 'required net free charge={}'.format(slab.get_required_net_free_charge())
    print 'el. density: ', get_electron_density(c_states, e_fermi, slab._condmass)
    print 'hole density:', get_hole_density(v_states, e_fermi, slab._valmass)

    plot(slab.get_xgrid(), slab.get_valence_profile(),'b',linewidth=2)
    plot(slab.get_xgrid(), slab.get_conduction_profile(),'r',linewidth=2)

    for w, v in c_states:
        plot(slab.get_xgrid(), w + zoom_factor * abs(v)**2,'r')
    for w, v in v_states:
        plot(slab.get_xgrid(), w + zoom_factor * abs(v)**2,'b')
    
    plot(slab.get_xgrid(), ones(slab.npoints) * e_fermi, '--')
    plot(slab.get_xgrid(), slab.get_V(),'k')

    ## For debugging purposes
    #plot(slab.get_xgrid(), slab._doping)
    #plot(slab.get_xgrid(), slab._epsilon)
    
    xlabel("x (ang)")
    ylabel("eV")
    show()
    
    
