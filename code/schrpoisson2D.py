"""
Main executable of the Schroedinger--Poisson solver for 2D materials.
See PDF documentation for its usage.

You need to pass on the command line the path to two JSON files, the first
to the materials properties, and the second to the calculation input flags:

  python modulable_2Dschrpoisson.py {material_props}.json  {calc_input}.json

Note that to run the code you need first to compile the fortran code using 
'make'.


If you use this code in your work, please cite the following paper:

A. Bussy, G. Pizzi, M. Gibertini, Strain-induced polar discontinuities
in 2D materials from combined first-principles and Schroedinger-Poisson
simulations, Phys. Rev. B 96, 165438 (2017).

This code is released under a MIT license, see LICENSE.txt file in the main
folder of the code repository, hosted on GitHub at
https://github.com/giovannipizzi/schrpoisson_2dmaterials
"""
import numpy as n
import scipy
import scipy.linalg
import time
import scipy.special
import os
from operator import add
import sys
import json

try:
    import schrpoisson_wire as spw        
except ImportError:
    raise ImportError("You need to have the schrpoisson_wire module.\n"
                      "To obtain it, you have to compile the Fortran code using f2py.\n"
                      "If you have f2py already installed, you will most probably need only to\n"
                      "run 'make' in the same folder as the code.")

# Python code version
__version__ = "1.1.0"


#hbar^2/m0 in units of eV*ang*ang
HBAR2OVERM0=7.61996163
# periodicity, should remain true in this case
is_periodic = True

# Small threshold to check for charge neutrality
small_threshold = 1.e-6
    
# If True, in run_simulation redirect part of the output to /dev/null
reduce_stdout_output = True


class ValidationError(Exception):
    pass

class InternalError(Exception):
    pass

def read_input_materials_properties(matprop):
    """
    Build a suitable dictionary containing the materials properties starting from an external json file
    """
    suitable_mat_prop = {}
    num_of_mat = len(matprop)
    valence_keys = []
    conduction_keys = []
    try:
         a_lat = matprop["0.00"]["x_lat"]
         b_lat = matprop["0.00"]["y_lat"]
    except KeyError:
         raise ValidationError("Error: lattice parameters not set up correctly in file '%s'" %json_matprop)
         
    #looping over the first input dict entry to get the keys
    try:
         for key in matprop["0.00"].keys():
             if "valence" in key:
                  valence_keys.append(key)
             if "conduction" in key:
                  conduction_keys.append(key)
         if len(valence_keys) == 0 or len(conduction_keys) == 0:
             raise KeyError          
    except KeyError:
        raise ValidationError("Error: The material properties json file (%s) must contain unstrained data with key '0.00' and subdictionaries must contain valence/conduction in their names" %json_matprop)
    
    #looping over the different strains
    try:
         for strain in matprop.keys():
             condenergies = []
             valenergies = []
             condmass = []
             valmass = []
             conddosmass = []
             valdosmass = []
             conddegeneracy = []
             valdegeneracy = []
             
             #generalities
             suitable_mat_prop[strain] = {}
             suitable_mat_prop[strain]["alpha"] = 4.*n.pi* 0.0055263496 * matprop[strain]["alpha_xx"] # 4 pi epsilon0 in units of e/(V*ang)
             suitable_mat_prop[strain]["ndoping"] = 0. #no doping allowed for now
             suitable_mat_prop[strain]["val_offset"] = 0.
             suitable_mat_prop[strain]["pol_charge"] = matprop[strain]["polarization_charge"]*1./b_lat * 1e8 #in units of e/cm
             
             #energy extrema specifics
             for key in valence_keys:
                  valenergies.append(matprop[strain][key]["energy"]-matprop[strain]["vacuum_level"]+matprop["0.00"]["vacuum_level"]) #need to align bands be substracting the vaccum level
                  valmass.append(matprop[strain][key]["conf_mass"])
                  valdosmass.append(matprop[strain][key]["DOS_mass"])
                  valdegeneracy.append(matprop[strain][key]["degeneracy"])
             for key in conduction_keys:
                  condenergies.append(matprop[strain][key]["energy"]-matprop[strain]["vacuum_level"]+matprop["0.00"]["vacuum_level"]) #need to align bands be substracting the vaccum level
                  condmass.append(matprop[strain][key]["conf_mass"])
                  conddosmass.append(matprop[strain][key]["DOS_mass"])
                  conddegeneracy.append(matprop[strain][key]["degeneracy"])
         
             #building the dictionary
             suitable_mat_prop[strain]["valenergies"] = valenergies
             suitable_mat_prop[strain]["condenergies"] = condenergies
             suitable_mat_prop[strain]["valmass"] = valmass
             suitable_mat_prop[strain]["condmass"] = condmass
             suitable_mat_prop[strain]["valdosmass"] = valdosmass
             suitable_mat_prop[strain]["conddosmass"] = conddosmass
             suitable_mat_prop[strain]["valdegeneracy"] = valdegeneracy
             suitable_mat_prop[strain]["conddegeneracy"] = conddegeneracy
                 
    except KeyError:
         raise ValidationError("Error: The material properties json file (%s) is not rightly organized: some dictionary keys might be missing" %json_matprop)
    
    #creating the delta doping layers that will appear at interfaces
    #looping over the strains, add a n and p delta doping for each
    
    for strain in suitable_mat_prop.keys():
         #creating n_deltadoping
         suitable_mat_prop[strain+"_n_deltadoping"] = {}
         suitable_mat_prop[strain+"_n_deltadoping"]["ndoping"] = suitable_mat_prop[strain]["pol_charge"]
         #creating p_deltadoping
         suitable_mat_prop[strain+"_p_deltadoping"] = {}
         suitable_mat_prop[strain+"_p_deltadoping"]["ndoping"] = -suitable_mat_prop[strain]["pol_charge"]
         
    return suitable_mat_prop, a_lat, b_lat


def update_mat_prop_for_new_strain(mat_prop, new_strain, plot_fit = False):
    """
    For a given strain, the relevant information such masses and energies are interpolated and the 
    material properties dictionary is updated
    """
    
    # Preliminary plot, if needed
    if plot_fit:
        import matplotlib.pyplot as plt

    # initializing arrays containing valence and conduction properties
    valenergies = n.zeros((len(mat_prop)/3,len(mat_prop["0.00"]["valenergies"]))) # len(mat_prop)/3 because for each strain there are 2 delta doping
    condenergies = n.zeros((len(mat_prop)/3,len(mat_prop["0.00"]["condenergies"])))
    valmass = n.zeros((len(mat_prop)/3,len(mat_prop["0.00"]["valmass"])))
    condmass = n.zeros((len(mat_prop)/3,len(mat_prop["0.00"]["condmass"])))
    valdosmass = n.zeros((len(mat_prop)/3,len(mat_prop["0.00"]["valdosmass"])))
    conddosmass = n.zeros((len(mat_prop)/3,len(mat_prop["0.00"]["conddosmass"])))
    pol_charge = n.zeros(len(mat_prop)/3)
    alpha = n.zeros(len(mat_prop)/3)
    
    # the strain array vs which evry physical quantity will be fitted (not sorted yet)
    strain = n.zeros(len(mat_prop)/3)
    
    # looping on the material properties versus strain
    i = 0
    for key in mat_prop.keys():
         if "doping" not in key:
             strain[i] = float(key)
             valenergies[i,:] = mat_prop[key]["valenergies"] 
             condenergies[i,:] = mat_prop[key]["condenergies"]
             valmass[i,:] = mat_prop[key]["valmass"]
             condmass[i,:] = mat_prop[key]["condmass"]
             valdosmass[i,:] = mat_prop[key]["valdosmass"]
             conddosmass[i,:] = mat_prop[key]["conddosmass"]
             pol_charge[i] = mat_prop[key]["pol_charge"]
             alpha[i] = mat_prop[key]["alpha"]
             i += 1
    
    # sorting the arrays so that they correspond to strain in increasing order
    order = n.argsort(strain)
    strain = strain[order]
    valenergies = valenergies[order,:]
    condenergies = condenergies[order,:]
    valmass = valmass[order,:]
    condmass = condmass[order,:]
    valdosmass = valdosmass[order,:]
    conddosmass = conddosmass[order,:]
    pol_charge = pol_charge[order]
    alpha = alpha[order]
    
    # the future dictionary entry for the new strain
    new_strain_prop = {}
    
    #actually fitting with optional visualization
    
    #polarization charge
    p = n.polyfit(strain,pol_charge,3)
    new_strain_prop["pol_charge"] = p[0]*new_strain**3  + p[1]*new_strain**2 + p[2]*new_strain + p[3]
    if plot_fit:
         plt.ion()
         plt.figure(1)
         plt.title("Polarization charge fit")
         plt.plot(strain,pol_charge,'kx')
         x = n.arange(0.,max(max(strain),new_strain)+0.001,0.001)
         y = p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
         plt.plot(x,y )
         plt.xlabel("Strain")

    # alpha 
    p = n.polyfit(strain,alpha,3)
    new_strain_prop["alpha"] = p[0]*new_strain**3  + p[1]*new_strain**2 + p[2]*new_strain + p[3]
    if plot_fit:
         plt.figure(2)
         plt.title("Alpha fit")
         plt.plot(strain,alpha,'kx')
         x = n.arange(0.,max(max(strain),new_strain)+0.001,0.001)
         y = p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
         plt.plot(x,y )
         plt.xlabel("Strain")

    
    # valence band energies
    new_valenergies = []
    if plot_fit:
         plt.figure(3)
         plt.title("Valence energies fit")
         plt.xlabel("Strain")
    for j in range(len(mat_prop["0.00"]["valenergies"])):
         p = n.polyfit(strain,valenergies[:,j],2)
         new_valenergies.append( p[0]*new_strain**2  + p[1]*new_strain + p[2])
         if plot_fit:
             plt.plot(strain,valenergies[:,j],'kx')
             x = n.arange(0.,max(max(strain),new_strain)+0.001,0.001)
             y = p[0]*x**2 + p[1]*x + p[2]
             plt.plot(x,y)
    new_strain_prop["valenergies"] = new_valenergies
    
    # conduction band energies
    new_condenergies = []
    if plot_fit:
         plt.figure(4)
         plt.title("Conduction energies fit")
         plt.xlabel("Strain")
    for j in range(len(mat_prop["0.00"]["condenergies"])):
         p = n.polyfit(strain,condenergies[:,j],2)
         new_condenergies.append( p[0]*new_strain**2  + p[1]*new_strain + p[2])
         if plot_fit:
             plt.plot(strain,condenergies[:,j],'kx')
             x = n.arange(0.,max(max(strain),new_strain)+0.001,0.001)
             y = p[0]*x**2 + p[1]*x + p[2]
             plt.plot(x,y)
    new_strain_prop["condenergies"] = new_condenergies
    
    # valence band confinement masses
    new_valmass = []
    if plot_fit:
         plt.figure(5)
         plt.title("Valence confinement (inverse) mass fit")
         plt.xlabel("Strain")
    for j in range(len(mat_prop["0.00"]["valmass"])):
         p = n.polyfit(strain,1./valmass[:,j],2)
         new_valmass.append( 1./(p[0]*new_strain**2  + p[1]*new_strain + p[2]))
         if plot_fit:
             plt.plot(strain,1./valmass[:,j],'kx')
             x = n.arange(0.,max(max(strain),new_strain)+0.001,0.001)
             y = (p[0]*x**2 + p[1]*x + p[2])
             plt.plot(x,y)
    new_strain_prop["valmass"] = new_valmass
    
    # conduction band confinement masses
    new_condmass = []
    if plot_fit:
         plt.figure(6)
         plt.title("Conduction confinement (inverse) mass fit")
         plt.xlabel("Strain")
    for j in range(len(mat_prop["0.00"]["condmass"])):
         p = n.polyfit(strain,1./condmass[:,j],2)
         new_condmass.append( 1./(p[0]*new_strain**2  + p[1]*new_strain + p[2]))
         if plot_fit:
             plt.plot(strain,1./condmass[:,j],'kx')
             x = n.arange(0.,max(max(strain),new_strain)+0.001,0.001)
             y = p[0]*x**2 + p[1]*x + p[2]
             plt.plot(x,y)
    new_strain_prop["condmass"] = new_condmass
    
    # valence band DOS masses
    new_valdosmass = []
    if plot_fit:
         plt.figure(7)
         plt.title("Valence DOS (inverse) mass fit")
         plt.xlabel("Strain")
    for j in range(len(mat_prop["0.00"]["valdosmass"])):
         p = n.polyfit(strain,1./valdosmass[:,j],2)
         new_valdosmass.append( 1./(p[0]*new_strain**2  + p[1]*new_strain + p[2]))
         if plot_fit:
             plt.plot(strain,1./valdosmass[:,j],'kx')
             x = n.arange(0.,max(max(strain),new_strain)+0.001,0.001)
             y = p[0]*x**2 + p[1]*x + p[2]
             plt.plot(x,y)
    new_strain_prop["valdosmass"] = new_valdosmass
    
    # conduction band DOS masses
    new_conddosmass = []
    if plot_fit:
         plt.figure(8)
         plt.title("Conduction DOS (inverse) mass fit")
         plt.xlabel("Strain")
    for j in range(len(mat_prop["0.00"]["conddosmass"])):
         p = n.polyfit(strain,1./conddosmass[:,j],2)
         new_conddosmass.append(1./(p[0]*new_strain**2  + p[1]*new_strain + p[2]))
         if plot_fit:
             plt.plot(strain,1./conddosmass[:,j],'kx')
             x = n.arange(0.,max(max(strain),new_strain)+0.001,0.001)
             y = p[0]*x**2 + p[1]*x + p[2]
             plt.plot(x,y)
    new_strain_prop["conddosmass"] = new_conddosmass
    
    new_strain_prop["ndoping"] = 0. #no doping allowed for now
    new_strain_prop["val_offset"] = 0.
    
    #plotting the fits if need be
    if plot_fit:
         plt.show()
         print >> sys.stderr, ("Press any key (+ENTER) to close all plots and continue")
         raw_input()
         plt.close("all")
         plt.ioff()
    
    # it should never happen as it should be tested before hand
    for key in mat_prop.keys():
         if "doping" not in key:
             if float(key) == new_strain:
                  mat_prop[str(new_strain)] = mat_prop[key]
                  mat_prop[str(new_strain)+"_n_deltadoping"] = {}
                  mat_prop[str(new_strain)+"_n_deltadoping"]["ndoping"] = mat_prop[key]["pol_charge"]
                  mat_prop[str(new_strain)+"_p_deltadoping"] = {}
                  mat_prop[str(new_strain)+"_p_deltadoping"]["ndoping"] = -mat_prop[key]["pol_charge"]
                  return None
                  
    # updating the dictionary passed as an argument
    mat_prop[str(new_strain)] = new_strain_prop
    
    # also adding delta dopings
    # creating n_deltadoping
    mat_prop[str(new_strain)+"_n_deltadoping"] = {}
    mat_prop[str(new_strain)+"_n_deltadoping"]["ndoping"] = mat_prop[str(new_strain)]["pol_charge"]
    # creating p_deltadoping
    mat_prop[str(new_strain)+"_p_deltadoping"] = {}
    mat_prop[str(new_strain)+"_p_deltadoping"]["ndoping"] = -mat_prop[str(new_strain)]["pol_charge"]
         
    
class Slab(object):
    """
    General class to represent a system composed of multiple stripes
    """
    def __init__(self, layers, materials_properties, delta_x, smearing, beta_eV):
        """
        Pass a suitable xgrid (containing the sampling points in units of angstroms) and
        a (in angstroms)
        """
        if len(layers) == 0:
            raise ValueError("layers must have at least one layer")

        self.max_step_size = 0.8
        self._slope = 0. # in V/ang
        self.delta_x = delta_x
        self.smearing=smearing
        self.beta_eV=beta_eV

        total_length = 0.
        # I create a list where each element is a tuple in the format
        # (full_material_properties, end_x_ang) 
        # (the first layer starts at x=0.
        self._layers_range = []
        xgrid_pieces = []
        materials = []
        for layer_idx, l in enumerate(layers):
            nintervals = int(n.ceil(l[1]/self.delta_x))
            # I want always the same grid spacing, so I possibly increase (slightly) the
            # thickness
            layer_length = nintervals * self.delta_x
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

            materials.append(materials_properties[l[0]])        
            #print sum(len(i) for i in xgrid_pieces[:-1]), sum(len(i) for i in xgrid_pieces)
            total_length += layer_length

        self._xgrid = n.concatenate(xgrid_pieces)

        # A check that all steps are equal; I calculate the error of each step w.r.t. the
        # expected step delta-x
        steps_error = (self._xgrid[1:] - self._xgrid[:-1]) - self.delta_x
        if abs(steps_error).max() > 1.e-10:
            raise AssertionError("The steps should be all equal to delta_x, but they aren't! "
                                 "max is: {}".format(abs(steps_error).max()))

        # Polarizability
        self._alpha = n.zeros(len(self._xgrid))
        last_idx = 0
        for mat, grid_piece in zip(materials, xgrid_pieces):
            if len(grid_piece)!=0: # skip delta dopings
                self._alpha[last_idx:last_idx+len(grid_piece)] = mat['alpha']
            last_idx += len(grid_piece)

        # Conduction band and valence band profiles, in eV
        # effective masses and DOS masses, in units of the free electron mass
        # degeneracy is unitless and integer
        # Need the total number of valence and conduction extrema
        
        
        self._ncond_min = len(materials[0]['condenergies'])
        self._nval_max = len(materials[0]['valenergies'])
        
        # since all materials are requiered to have the same degeneracy for corresponding energy maxima
        self._conddegen = materials[0]['conddegeneracy']
        self._valdegen = materials[0]['valdegeneracy']
        
        self._condband = n.zeros((self._ncond_min,len(self._xgrid)))
        self._valband = n.zeros((self._nval_max,len(self._xgrid)))
        self._valmass = n.zeros((self._nval_max,len(self._xgrid)))
        self._condmass = n.zeros((self._ncond_min,len(self._xgrid)))
        self._valdosmass = n.zeros((self._nval_max,len(self._xgrid)))
        self._conddosmass = n.zeros((self._ncond_min,len(self._xgrid)))
        
        # conduction band
        for i in range(self._ncond_min):
            last_idx = 0
            for mat, grid_piece in zip(materials, xgrid_pieces):
                if len(grid_piece)!=0: # skip delta dopings
                    self._condband[i,last_idx:last_idx+len(grid_piece)] = mat['val_offset'] + mat['condenergies'][i]
                    self._condmass[i,last_idx:last_idx+len(grid_piece)] = mat['condmass'][i]
                    self._conddosmass[i,last_idx:last_idx+len(grid_piece)] = mat['conddosmass'][i]
                last_idx += len(grid_piece)

        # valence band
        for j in range(self._nval_max):
            last_idx = 0
            for mat, grid_piece in zip(materials, xgrid_pieces):
                if len(grid_piece)!=0: # skip delta dopings
                    self._valband[j,last_idx:last_idx+len(grid_piece)] = mat['val_offset'] + mat['valenergies'][j]
                    self._valmass[j,last_idx:last_idx+len(grid_piece)] = mat['valmass'][j]
                    self._valdosmass[j,last_idx:last_idx+len(grid_piece)] = mat['valdosmass'][j]
                last_idx += len(grid_piece)  


        # Doping; I also count total free holes and free electrons. In e/cm
        self._doping = n.zeros(len(self._xgrid))
         
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
        
        # memory for converging algorithm
        self._old_V = n.zeros(len(self._xgrid))
        self._indicator = n.zeros(2)
        self._counter = 0
        self._subcounter = 0
        self._Ef = n.zeros(2)
        self._E_count = 0
        self._max_ind = 0
        self._finalV_check = 0.0
        self._finalE_check = 0.0
         
        # Add an atribute that tells how much time is spent doing different tasks for optimization purposes
        # The tasks of interests are Solving Poisson equation, computing states (i.e. Hamiltonian digonalization), finding the Fermi energy
        # All times in seconds
        self._time_Poisson = 0.0
        self._time_Fermi = 0.0
        self._time_Hami = 0.0
         
    def get_computing_times(self):
         return self._time_Poisson, self._time_Fermi, self._time_Hami
    
    def update_computing_times(self,process,value):
         """
         updates the time spent on a numerical process (string) either "Fermi", "Hami", "Poisson"
         """
         if process == "Fermi":
             self._time_Fermi += value
         elif process == "Poisson":
             self._time_Poisson += value
         elif process == "Hami":
             self._time_Hami += value
         else:
             warnings.warn(process+" is not a valid process for performance monitoring")
         
         
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
        self._counter += 1
        self._Ef[self._counter%2] = e_fermi
        
        max_iteration = 5000

        V_conv_threshold = 2.e-4
             
        Ef_conv_threshold = 1.e-6 # if fermi energy does not change from one iteration to another, converged

        free_electrons_density = get_electron_density(c_states, e_fermi, self._conddosmass, self.npoints, self._conddegen, smearing=self.smearing, beta_eV=self.beta_eV)
        free_holes_density = get_hole_density(v_states, e_fermi, self._valdosmass, self.npoints, self._valdegen, smearing=self.smearing, beta_eV=self.beta_eV)
         
        total_charge_density =  self._doping - free_electrons_density + free_holes_density
         
        #updating the time spent solving Poisson
        start_t = time.time()
        if is_periodic:         
            new_V = -1.*spw.periodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0] # minus 1 because function returns electrostatic potential, not energy
        else:
            new_V = -1.*spw.nonperiodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0]
        end_t = time.time()

        self._time_Poisson += end_t-start_t
         
        new_V -= n.mean(new_V)
        
        if self._counter == 1:
            self._max_ind = n.argmax(new_V)
                  
        if zero_elfield:
            # in V/ang
            self._slope = (new_V[-1] - new_V[0])/(self._xgrid[-1] - self._xgrid[0])
            new_V -= self._slope*self._xgrid
        else:
            self._slope = 0.
                
        #we want ot avoid oscillations in converging algorithm
        #one has to stock new_V for comparison purposes
        
        
        self._indicator[self._counter%2] = new_V[self._max_ind]-self._V[self._max_ind] #need to keep track of oscillations when converging

        
        if self._indicator[0]*self._indicator[1] < 0:
            #oscillation, take the middle ground
            print "OSCILLATION"
            self._subcounter = 0
            self.max_step_size *= 0.1
            if self.max_step_size <= 0.1*V_conv_threshold:
                                    self.max_step_size = 0.1*V_conv_threshold
            
        else:
            self._subcounter += 1
        
        if self._subcounter == 20:
            self.max_step_size *= 1.4
            self._subcounter = 0
                    
        step = new_V - self._V
        current_max_step_size = n.max(n.abs(step))
      
        #convergence check
        self._over = False
        if current_max_step_size < V_conv_threshold:
                           start_t = time.time()
                           if is_periodic:
                                    check_V = -1.*spw.periodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0] # minus 1 because function returns electrostatic potential, not energy

                           else:
                                    check_V = -1.*spw.nonperiodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0]
                           end_t = time.time()
                           self._time_Poisson += end_t-start_t
                           check_val = n.max(n.abs(check_V-self._V))
                           self._indicator[self._counter%2] = check_V[self._max_ind]-self._V[self._max_ind]
                           if check_val > 5*V_conv_threshold:
                                    current_max_step_size = check_val
                                    step = check_V - self._V
                                    #self.max_step_size *= 0.5
                           else:
                                             
                                    self._over = True

                                    
                           
        print 'convergence param:', current_max_step_size         
        
        if current_max_step_size != 0 and  self._over == False:
                           #self._V += step * min(self.max_step_size, current_max_step_size) #/ (current_max_step_size)
                           self._V += step * self.max_step_size
                           self._old_V = self._V.copy()
        elif current_max_step_size == 0 and  self._over == False:
                           self._V = new_V
                           self._old_V = self._V.copy()
                  
        elif self._over == True:
            self._V = self._old_V
            print "Final convergence parameter: ", check_val
            self._finalV_check = check_val
                  
        if n.abs(self._Ef[0]-self._Ef[1]) <= Ef_conv_threshold and current_max_step_size < 10*V_conv_threshold:
            self._E_count += 1
            
            if self._E_count  == 4:
                print "Convergence of Fermi energy: ", n.abs(self._Ef[0]-self._Ef[1])
                current_max_step_size = 0.1*V_conv_threshold # froced convergence if that happens 4 times in a row
                if is_periodic:
                    check_V = -1.*spw.periodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0] # minus 1 because function returns electrostatic potential, not energy

                else:
                    check_V = -1.*spw.nonperiodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0]
                check_val = n.max(n.abs(check_V-self._V))
                print "Final convergence parameter: ", check_val
                self._finalV_check = check_val
        else:
                           self._E_count = 0    
       
        self._finalE_check = n.abs(self._Ef[0]-self._Ef[1])
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
        Return the conduction band profile in eV
        """
        # need dimensions to agree
        V = n.zeros((self._ncond_min,len(self._xgrid)))
        for i in range(self._ncond_min):
                           V[i,:] = self._V
        return self._condband + V

    def get_valence_profile(self):
        """
        Return the valence band profile in eV
        """
        # need dimensions to agree
        V = n.zeros((self._nval_max,len(self._xgrid)))
        for i in range(self._nval_max):
                           V[i,:] = self._V
        return self._valband + V
        
    def get_band_gap(self):    
        """
        Scans valence and conduction profiles in order to find the absolute conduction minimum and the absolute valence band maximum and returns the difference
        """
        conduction = self.get_conduction_profile()
        valence = self.get_valence_profile()
        
        cond_min = n.min(conduction)
        val_max = n.max(valence)
        
        return cond_min - val_max

    
def MV_smearing(E,beta,mu):
    """
    Marzari-Vanderbilt smearing function to be integrated in conjuction with the density of states
    Be careful: units of beta, E and mu must be consistent
    """
    
    return 0.5*scipy.special.erf(-beta*(E-mu)-1./n.sqrt(2)) + 1./n.sqrt(2.*n.pi)*n.exp(-(beta*(E-mu)+1./n.sqrt(2))**2) + 0.5
    

def get_electron_density(c_states, e_fermi, c_mass_array, npoints,degen, smearing, beta_eV, band_contribution = False, avg_eff_mass = False):
    """
    Fill subbands with a 1D dos, at T=0 (for now; T>0 requires numerical integration)
    The first index of c_states must be the state energy in eV
    e_fermi in eV
    c_mass is array with the conduction DOS mass (on the grid)
        in units of the free electron mass
    degen is the array containing the degeneracy of each conduction band minimum  

    Return linear electron density, in e/cm
    
    if avg_eff_mass, an array containing the average effective mass for each state and the state energy is returned
    """
    # The 1D DOS is (including the factor of 2 for the spin):
    # g(E) = sqrt(2 * effmass)/(pi*hbar) * 1/sqrt(E-E0)
    # where effmass is the band effective mass, E0 is the band edge.
    #
    # I rewrite it as g(E) = D * sqrt(meff/m0) / sqrt(E-E0)
    # where (meff/m0) is simply the effective mass in units of the electron free mass,
    # and D=sqrt(2) / pi / sqrt(HBAR2OVERM0) and will be in units of 1/ang/sqrt(eV)
    D = n.sqrt(2.) / n.pi / n.sqrt(HBAR2OVERM0)

    el_density = n.zeros(npoints)   
    
    contrib = n.zeros(len(degen))    
    
    avg_mass = n.zeros((1,3))
    
    # All the conduction band minima have to be taken into account with the appropriate degeneracy
    for j in range(len(degen)): #number of minima
        deg = degen[j]
        if j > 0 and avg_eff_mass == True:
             avg_mass = n.append(avg_mass,[[0.,0.,0.]],axis=0) # so that bands are separated by a line of zeros
        for state_energy, state in c_states[j]:
            energy_range = 20. # eV, to be very safe
             
            #if state_energy > e_fermi:
            #    continue
            square_norm = sum((state)**2)
            # I average the inverse of the effective mass
            # Both state and c_mass_array should have the same length
            # NOT SURE: square_norm or sqrt(square_norm) ? AUGU: I'm pretty sure it's square_norm and I changed it
            averaged_eff_mass = 1./(sum(state**2 / c_mass_array[j]) / square_norm)
            if avg_eff_mass == True:
                avg_mass = n.append(avg_mass,[[state_energy,state_energy-e_fermi,averaged_eff_mass]],axis=0)
             
            if not smearing and state_energy < e_fermi:
                # At T=0, integrating from E0 to Ef the DOS gives
                # D * sqrt(meff) * int_E0^Ef 1/(sqrt(E-E0)) dE =
                # D * sqrt(meff) * 2 * sqrt(Ef-E0)   [if Ef>E0, else zero]
                el_density += deg * D * n.sqrt(averaged_eff_mass) * 2. * n.sqrt(e_fermi - state_energy) * (
                    state**2 / square_norm)
                contrib[j] += n.sum(deg * D * n.sqrt(averaged_eff_mass) * 2. * n.sqrt(e_fermi - state_energy) * (
                    state**2 / square_norm))
            
            elif smearing and state_energy-e_fermi < energy_range: # more than enough margin
                # Need to numerically integrate the density of state times the occupation given by MV_smearing
                # to compute the integral, one uses the trick explained there to avoid singularities: http://math.stackexchange.com/questions/1351734/numerical-integration-of-divergent-function
                n_int = 10000. # number of intervals
                # change of variable E = state_energy + t**2
                dt = n.sqrt(energy_range)/n_int 
                t = n.arange(0,n.sqrt(energy_range),dt) 
                #plt.figure(1)
                #plt.plot(energy,deg * D * sqrt(averaged_eff_mass) * 1./n.sqrt(energy-state_energy),"b")
                #plt.plot(energy,MV_smearing(energy,beta_eV,e_fermi),"r")
                #plt.plot(energy,g_times_f,"k")
                #plt.title("%s"%e_fermi)
                #plt.show()
                temp_dens = 2*deg * D * n.sqrt(averaged_eff_mass) * n.trapz(MV_smearing(state_energy+t**2,beta_eV,e_fermi),dx=dt) * (state**2 / square_norm)
                el_density += temp_dens
                contrib[j] += n.sum(temp_dens)
                  
    # Up to now, el_density is in 1/ang; we want it in 1/cm
    if band_contribution == False:
        return el_density * 1.e8
    elif band_contribution == True and avg_eff_mass == False:
        return el_density * 1.e8, contrib  * 1.e8
    else:
         return el_density * 1.e8, contrib  * 1.e8, avg_mass

def update_doping(materials_props,material,doping):
    """
    materials_props is a dictionnary containing the materials properties (should be a copy of the original one)
    material is an entry (string) of materials_properties for which the doping must be updated
    doping is the updated value of the doping in e/cm. It must be negative for holes    
    """
    materials_props[material]['ndoping'] = doping
    
def get_energy_gap(c_states,v_states,c_degen,v_degen):
    """
    returns the difference between the lowest conduction electron state and the highest valence hole state
    c_degen and v_degen are the degeneracy lists
    """    
    all_val_states_energies = n.zeros(1)
    all_cond_states_energies = n.zeros(1)

    for i in range(len(c_degen)):
        all_cond_states_energies = n.append(all_cond_states_energies,[s[0] for s in c_states[i]])
    for j in range(len(v_degen)):
        all_val_states_energies = n.append(all_val_states_energies, [s[0] for s in v_states[j]])
    
    #suppressing the first entries of the arrays
    return n.min(n.delete(all_cond_states_energies,0)) - n.max(n.delete(all_val_states_energies,0))

def get_hole_density(v_states, e_fermi, v_mass_array, npoints, degen, smearing, beta_eV, band_contribution = False, avg_eff_mass = False):
    """
    For all documentation and comments, see get_electron_density
    
    v_mass_array should contain the DOS mass of holes, i.e., positive
    
    degen is the array containing the degeneracy of each valence band maximum

    Return a positive number
    """
    D = n.sqrt(2.) / n.pi / n.sqrt(HBAR2OVERM0)
    
    h_density = n.zeros(npoints)  
    
    avg_mass = n.zeros((1,3))
    
    contrib = n.zeros(len(degen))
    for j in range(len(degen)):
        deg = degen[j]
        if j > 0 and avg_eff_mass == True:
            avg_mass = n.append(avg_mass,[[0.,0.,0.]],axis=0) # so that bands are separated by a line of zeros
        for state_energy, state in v_states[j]:
            energy_range = 20. # eV to be extra safe
             
            # Note that here the sign is opposite w.r.t. the conduction case
            #if state_energy < e_fermi:
            #    continue
            square_norm = sum((state)**2)
            averaged_eff_mass = 1./(sum(state**2 / v_mass_array[j]) / square_norm)
            if avg_eff_mass == True:
                avg_mass = n.append(avg_mass,[[state_energy,state_energy-e_fermi,averaged_eff_mass]],axis=0)
                  
            if not smearing and state_energy > e_fermi:
                h_density += deg * D * n.sqrt(averaged_eff_mass) * 2. * n.sqrt(state_energy - e_fermi) * (
                    state**2 / square_norm)
                contrib[j] += n.sum(deg * D * n.sqrt(averaged_eff_mass) * 2. * n.sqrt(state_energy - e_fermi) * (
                    state**2 / square_norm))    
                      
             
            elif smearing and e_fermi-state_energy < energy_range:
                
                n_int = 10000. # number of intervals
                # change of variable E = state_energy + t**2
                dt = n.sqrt(energy_range)/n_int 
                t = n.arange(0,n.sqrt(energy_range),dt) 
                temp_dens = 2*deg * D * n.sqrt(averaged_eff_mass) * n.trapz(MV_smearing(2*e_fermi-state_energy+t**2,beta_eV,e_fermi),dx=dt) * (state**2 / square_norm)
                h_density += temp_dens
                contrib[j] += n.sum(temp_dens)
                
                # to keep a trace of old work
                """
                n_int = 100000. # 500 intervals
                delta_E = energy_range/n_int 
                energy = n.arange(state_energy-energy_range,state_energy,delta_E)
                energy -= 1.e-8 # to avoid dividing by zero
                g_times_f = deg * D * sqrt(averaged_eff_mass) * 1./n.sqrt(state_energy-energy) * MV_smearing(2*e_fermi-energy,beta_eV,e_fermi)
                #plt.figure(2)
                #plt.plot(energy,deg * D * sqrt(averaged_eff_mass) * 1./n.sqrt(state_energy-energy),"b")
                #plt.plot(energy,MV_smearing(2*e_fermi-energy,beta_eV,e_fermi),"r")
                #plt.plot(energy,g_times_f,"k")
                #plt.title("%s"%e_fermi)
                #plt.show()
                h_density += n.trapz(g_times_f,dx=delta_E) * (state**2 / square_norm)
                contrib[j] += sum(n.trapz(g_times_f,dx=delta_E) * (state**2 / square_norm))
                """ 
    if band_contribution == False:
        return h_density * 1.e8
    elif band_contribution == True and avg_eff_mass == False:
        return h_density * 1.e8, contrib * 1.e8
    else:
         return  h_density * 1.e8, contrib * 1.e8, avg_mass
    

def find_efermi(c_states, v_states, c_mass_array, v_mass_array, c_degen, v_degen, npoints, target_net_free_charge, smearing, beta_eV):
    """
    Pass the conduction and valence states (and energies), 
    the conduction and valence DOS mass arrays, 
    and the net charge to be used as a target (positive means that we want holes than electrons)
    """
    more_energy = 1. # in eV
    # TODO: we may want a precision also on the charge, not only on the energy
    # DONE
    energy_precision = 1.e-8 # eV
    charge_precision = 1. # out of the 10**7 it is still very precise 
    all_states_energies = n.zeros(1)
    for i in range(len(c_degen)):
        all_states_energies = n.append(all_states_energies,[s[0] for s in c_states[i]])
    for j in range(len(v_degen)):
        all_states_energies = n.append(all_states_energies, [s[0] for s in v_states[j]])
    all_states_energies = n.delete(all_states_energies,0)
    # I set the boundaries for the bisection algorithm; I could in principle
    # also extend these ranges
    ef_l = all_states_energies.min()-more_energy
    ef_r = all_states_energies.max()+more_energy

    electrons_l = n.sum(get_electron_density(c_states, ef_l, c_mass_array,npoints,c_degen, smearing=smearing, beta_eV=beta_eV))
    holes_l = n.sum(get_hole_density(v_states, ef_l, v_mass_array, npoints,v_degen, smearing=smearing, beta_eV=beta_eV))
    electrons_r = n.sum(get_electron_density(c_states, ef_r, c_mass_array, npoints,c_degen, smearing=smearing, beta_eV=beta_eV))
    holes_r = n.sum(get_hole_density(v_states, ef_r, v_mass_array,npoints,v_degen, smearing=smearing, beta_eV=beta_eV))
    
    net_l = holes_l - electrons_l
    net_r = holes_r - electrons_r
    if (net_l - target_net_free_charge) * (
        net_r - target_net_free_charge) > 0:
        raise ValueError("The net charge at the boundary of the bisection algorithm "
                         "range has the same sign! {}, {}, target={}; ef_l={}, ef_r={}".format(
                             net_l, net_r, target_net_free_charge,ef_l, ef_r))
    absdiff = 10*charge_precision
    en_diff = ef_r - ef_l
    while en_diff > energy_precision: 
        ef = (ef_l + ef_r)/2.
        electrons = n.sum(get_electron_density(c_states, ef, c_mass_array,npoints,c_degen, smearing=smearing, beta_eV=beta_eV))
        holes = n.sum(get_hole_density(v_states, ef, v_mass_array,npoints,v_degen, smearing=smearing, beta_eV=beta_eV))
        net = holes - electrons
        absdiff = abs(net - target_net_free_charge)
        if (net - target_net_free_charge) * (
          net_r - target_net_free_charge) > 0:
            net_r = net
            ef_r = ef
        else:
            net_l = net
            ef_l = ef
        en_diff =  ef_r - ef_l   
        #check on the charge precision
        if  absdiff > charge_precision:
            #need to go over the loop once more
            en_diff = 10*energy_precision
    return ef #(ef_r + ef_l)/2. it changes everything since the bissection algorithm worked for ef and not for (ef_r + ef_l)/2.

def get_conduction_states_p(slab):
    """
    This function diagonalizes the matrix using the fact that the matrix is banded. This provides
    a huge speedup w.r.t. to a dense matrix.

    NOTE: _p stands for the periodic version
    """
    more_bands_energy = 0.2 # How many eV to to above the top of the conduction band
    
    # The list containing the tuples that will be returned
    res = []

    # Ham matrix in eV; I store only the first two diagonals
    # H[2,:] is the diagonal,
    # H[1,1:] is the first upper diagonal,
    # H[0,2:] is the second upper diagonal
    # However, many possible energy minima in conduction band => self._ncond_min similar matrices, all contained in the same H
    H = n.zeros((slab._ncond_min,3, slab.npoints))

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
         
         #Need to loop over all conduction band minima and construct their Hamiltonian
    for i in range(slab._ncond_min): 
         
        # On the diagonal, the potential: sum of the electrostatic potential + conduction band profile
        # This is obtained with get_conduction_profile()
        H[i,2,:][reordering] = slab.get_conduction_profile()[i,:]
        min_energy = H[i,2,:].min()
        max_energy = H[i,2,:].max() + more_bands_energy

        # The zero-th element contains the periodicity term
        mass_differences = n.zeros(slab.npoints)
        mass_differences[0] = (slab._condmass[i,0] + slab._condmass[i,-1])/2.    
        mass_differences[1:] = (slab._condmass[i,1:] + slab._condmass[i,:-1])/2.
    
        # Finite difference method for 2nd derivatives. Remember that the equation with an effective
        # mass is:
        # d/dx ( 1/m(x) d/dx psi(x)), i.e. 1/m(x) goes inside the first derivative
        # I set the coefficient for the 2nd derivative on the diagonal
        # mass_differences[1] is the average mass between 0 and 1
        H[i,2,:][reordering]  += (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences
        # mass_differences[1+1] is the average mass between 1 and 2
        H[i,2,:][reordering]  += (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences[
            (n.arange(slab.npoints)+1) % slab.npoints]

        # note! The matrix is symmetric only if the mesh step is identical for all steps
        # I use 1: in the second index because the upper diagonal has one element less, and
        # this is the format required by eig_banded
        # Note that this also sets superdiagonal[0] to the correct element that should be at
        # the corner of the matrix, at position (n-1, 0)
        superdiagonal = - (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences
        H[i][reordering_superdiagonals] = superdiagonal

        w, v = scipy.linalg.eig_banded(H[i], lower=False, eigvals_only=False, 
                                    overwrite_a_band=True, # May enhance performance
                                    select='v', select_range=(min_energy,max_energy),
                                    max_ev=slab.npoints) # max_ev: max # of eigenvalues to expect
                                                         # I use the worst case scenario, where I get
                                                         # all of them
        result_to_reorder = zip(w, v.T)
        res.append(tuple((w, v[reordering]) for w, v in result_to_reorder))
        
    return res


def get_valence_states_p(slab):
    """
    See discussion in get_conduction_states, plus comments here for what has changed
    
    NOTE: _p stands for the periodic version
    """
    more_bands_energy = 0.2 # How many eV to to below the bottom of the valence band
    
    # The list containing the tuples that will be returned
    res = []
    
    H = n.zeros((slab._nval_max,3, slab.npoints))

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
    
    for i in range(slab._nval_max):
        H[i,2,:][reordering] = slab.get_valence_profile()[i,:]
        min_energy = H[i,2,:].min() - more_bands_energy
        max_energy = H[i,2,:].max() 

        # In the valence bands, it is as if the mass is negative
        mass_differences = n.zeros(slab.npoints)
        mass_differences[0]  = -(slab._valmass[i,0] + slab._valmass[i,-1])/2.    
        mass_differences[1:] = -(slab._valmass[i,1:] + slab._valmass[i,:-1])/2.
    
        H[i,2,:][reordering]  += (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences
        H[i,2,:][reordering]  += (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences[
            (n.arange(slab.npoints)+1) % slab.npoints]

        superdiagonal = - (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences
        H[i][reordering_superdiagonals] = superdiagonal

        w, v = scipy.linalg.eig_banded(H[i], lower=False, eigvals_only=False, 
                                    overwrite_a_band=True, # May enhance performance
                                    select='v', select_range=(min_energy,max_energy),
                                    max_ev=slab.npoints) # max_ev: max # of eigenvalues to expect
                                                         # I use the worst case scenario, where I get
                                                         # all of them
        result_to_reorder = zip(w, v.T)
        res.append(tuple((w, v[reordering]) for w, v in result_to_reorder))
        
    return res

def get_conduction_states_np(slab):
    """
    This function diagonalizes the matrix using the fact that the matrix is banded. This provides
    a huge speedup w.r.t. to a dense matrix.

    NOTE: _np stands for the non-periodic version
    """
    more_bands_energy = 0.2 # How many eV to to above the top of the conduction band
    
    # The list containing the tuples that will be returned
    res = []
    
    # Ham matrix in eV; I store only the first upper diagonal
    # H[1,:] is the diagonal,
    # H[0,1:] is the first upper diagonal
    H = n.zeros((slab._ncond_min,2, slab.npoints))
    
    for i in range(slab._ncond_min): 
            
        # On the diagonal, the potential: sum of the electrostatic potential + conduction band profile
        # This is obtained with get_conduction_profile()
        H[i,1,:] = slab.get_conduction_profile()[i]
        min_energy = H[i,1,:].min()
        max_energy = H[i,1,:].max() + more_bands_energy
            
        mass_differences = (slab._condmass[i,1:] + slab._condmass[i,:-1])/2.

        # Finite difference method for 2nd derivatives. Remember that the equation with an effective
        # mass is:
        # d/dx ( 1/m(x) d/dx psi(x)), i.e. 1/m(x) goes inside the first derivative
        # I set the coefficient for the 2nd derivative on the diagonal
        H[i,1,1:] += (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences
        H[i,1,:-1]  += (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences

        # note! The matrix is symmetric only if the mesh step is identical for all steps
        # I use 1: in the second index because the upper diagonal has one element less, and
        # this is the format required by eig_banded
        H[i,0,1:] = - (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences

        w, v = scipy.linalg.eig_banded(H[i], lower=False, eigvals_only=False, 
                                    overwrite_a_band=True, # May enhance performance
                                    select='v', select_range=(min_energy,max_energy),
                                    max_ev=slab.npoints) # max_ev: max # of eigenvalues to expect
                                                         # I use the worst case scenario, where I get
                                                         # all of them
        res.append(zip(w, v.T))
        
    return res

def get_valence_states_np(slab):
    """
    See discussion in get_conduction_states, plus comments here for what has changed

    NOTE: _np stands for the non-periodic version
    """
    more_bands_energy = 0.2 # How many eV to to below the bottom of the valence band
    
    # The list containing the tuples that will be returned
    res = []
    
    H = n.zeros((slab._nval_max,2, slab.npoints))
    
    for i in range(slab._nval_max):
        
        H[i,1,:] = slab.get_valence_profile()[i]
        min_energy = H[i,1,:].min() - more_bands_energy
        max_energy = H[i,1,:].max() 

        # In the valence bands, it is as if the mass is negative
        mass_differences = -(slab._valmass[i,1:] + slab._valmass[i,:-1])/2.

        H[i,1,1:] += (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences
        H[i,1,:-1]  += (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences

        H[i,0,1:] = - (HBAR2OVERM0/2.) / slab.delta_x**2 / mass_differences

        w, v = scipy.linalg.eig_banded(H[i], lower=False, eigvals_only=False, 
                                    overwrite_a_band=True, # May enhance performance
                                    select='v', select_range=(min_energy,max_energy),
                                    max_ev=slab.npoints) # max_ev: max # of eigenvalues to expect
                                                         # I use the worst case scenario, where I get
                                                         # all of them
        res.append(zip(w, v.T))
        
    return res
  

def run_simulation(slab, max_steps, nb_states, smearing, beta_eV, b_lat, delta_x):
    """
    This function launches the self-consistant Schroedinger--Poisson calculations and returns all the relevant data
    """ 
    it = 0

    # I don't want the terminal to be flooded with every single step
    if reduce_stdout_output:
        sys.stdout = open(os.devnull, "w")

    converged = False
    try:
        for iteration in range(max_steps):
            print 'starting iteration {}...'.format(iteration)
            it += 1
            start_t = time.time()
            if is_periodic:
                c_states = get_conduction_states_p(slab)   
                v_states = get_valence_states_p(slab)
            else:
                c_states = get_conduction_states_np(slab)   
                v_states = get_valence_states_np(slab)
            end_t = time.time()
            slab.update_computing_times("Hami", end_t-start_t)
             
            start_t = time.time()
            e_fermi = find_efermi(c_states, v_states, slab._conddosmass, slab._valdosmass,
                                  slab._conddegen, slab._valdegen, npoints=slab.npoints,
                                  target_net_free_charge=slab.get_required_net_free_charge(),
                                  smearing=smearing, beta_eV=beta_eV)
            end_t = time.time()
            slab.update_computing_times("Fermi", end_t-start_t)
            print iteration, e_fermi

            zero_elfield = is_periodic
            converged = slab.update_V(c_states, v_states, e_fermi, zero_elfield=zero_elfield)
            # slab._slope is in V/ang; the factor to bring it to V/cm
            print 'Added E field: {} V/cm '.format(slab._slope * 1.e8) 
            if converged:
                break
    except KeyboardInterrupt:
        pass
    if reduce_stdout_output:
        sys.stdout = sys.__stdout__
    if not converged:
        raise InternalError("****** ERROR! Calculation not converged ********")
    
    #should print all results at each step in files
    el_density, el_contrib, avg_cond_mass = get_electron_density(c_states, e_fermi, slab._conddosmass, slab.npoints, slab._conddegen, smearing=smearing, beta_eV=beta_eV, band_contribution = True, avg_eff_mass = True)
    hole_density, hole_contrib, avg_val_mass = get_hole_density(v_states, e_fermi, slab._valdosmass,slab.npoints, slab._valdegen, smearing=smearing, beta_eV=beta_eV, band_contribution = True, avg_eff_mass =True)
    tot_el_dens = n.sum(el_density)
    tot_hole_dens = n.sum(hole_density)
    tot_el_dens_2 = tot_el_dens * b_lat*1.e-8 # in units of e/b
    tot_hole_dens_2 = tot_hole_dens * b_lat*1.e-8 # in units of e/b
    el_density_per_cm2 = el_density * 1./(slab.delta_x*1.e-8) 
    hole_density_per_cm2 = hole_density * 1./(slab.delta_x*1.e-8) 
    
    #contribution in %
    if tot_el_dens != 0: 
        el_contrib /= tot_el_dens
    if tot_hole_dens != 0:    
        hole_contrib /= tot_hole_dens
         #print "El contrib: ", el_contrib
         #print "Hole contrib: ", hole_contrib
         
    matrix = [slab.get_xgrid(),n.ones(slab.npoints) * e_fermi]
    zoom_factor = 10. # To plot eigenstates
    
    # adding the potential profile of each band
         
    for k in range(slab._ncond_min):
        i=0
        matrix.append(slab.get_conduction_profile()[k])
        for w, v in c_states[k]:
            if i >= nb_states:
                break
            matrix.append(w + zoom_factor * n.abs(v)**2)
            matrix.append(n.ones(slab.npoints) * w)
            i+=1
             
    for l in range(slab._nval_max):   
        j=0     
        matrix.append(slab.get_valence_profile()[l])
        for w, v in v_states[l][::-1]:
            if j >= nb_states:
                  break
            # Plot valence bands upside down
            matrix.append(w - zoom_factor * n.abs(v)**2)
            matrix.append(n.ones(slab.npoints) * w)
            j+= 1
   
    #Keeping the user aware of the time spent on each main task
    print "Total time spent solving Poisson equation: ", slab.get_computing_times()[0], " (s)"
    print "Total time spent finding the Fermi level(s): ", slab.get_computing_times()[1], " (s)"
    print "Total time spent computing the electronic states: ", slab.get_computing_times()[2] , " (s)"
    
    return [[it, slab._finalV_check, slab._finalE_check, delta_x, e_fermi,  tot_el_dens, tot_hole_dens ,tot_el_dens_2,tot_hole_dens_2 ],
             [matrix],
             [slab.get_xgrid(),el_density_per_cm2,hole_density_per_cm2]]
                
 

def main_run(matprop, input_dict):
    """
    Main loop to run the code.

    :param matprop: dictionary with the content of the matprop json used in input
    :param input_dict: dictionary with the content of the input_dict json used in input
    """
    out_files = {}

    mat_properties, a_lat, b_lat = read_input_materials_properties(matprop)

    # Check and set smearing
    smearing = input_dict["smearing"]
    KbT = input_dict["KbT"]*13.6 # from Ry to eV, most often this precision is sufficient...
    beta_eV = 1./KbT

    #calculation type
    calculation_type = input_dict["calculation"]

    #max number of step for each self-consistent cycle
    max_steps = input_dict["max_iteration"]

    #do we plot the fits for new strains ?
    plot_fit = input_dict["plot_fit"]

    if calculation_type == "single_point":
         print("\n")
         print("Starting single-point calculation...")
         #updating the mat_properties dict in case there are non registered strains in the setup
         plotting_the_fit = False
         if plot_fit == True:
             plotting_the_fit = True
         for key in input_dict["setup"]:
             update_mat_prop_for_new_strain(mat_prop = mat_properties, new_strain = input_dict["setup"][key]["strain"], plot_fit = plotting_the_fit)
             plotting_the_fit = False
             
         #constructing the slab
         if len(input_dict["setup"]) < 3:
             raise ValidationError("Error: There must be at least three entries in the setup subdictionary of json file '%s'" %calc_input)
             
         #the first and last layers must be entered manually
         layers_p = []
         layers_p.append((str(input_dict["setup"]["slab1"]["strain"]),input_dict["setup"]["slab1"]["width"]))
         if input_dict["setup"]["slab1"]["polarization"] == "positive":
             layers_p.append((str(input_dict["setup"]["slab1"]["strain"])+"_p_deltadoping",0.0))
         else: 
             layers_p.append((str(input_dict["setup"]["slab1"]["strain"])+"_n_deltadoping",0.0))
         for i in range(len(input_dict["setup"])-2):
             slab_key = "slab"+str(i+2)
             #delta doping before layer
             if input_dict["setup"][slab_key]["polarization"] == "positive":
                  layers_p.append((str(input_dict["setup"][slab_key]["strain"])+"_n_deltadoping",0.0))
             else: 
                  layers_p.append((str(input_dict["setup"][slab_key]["strain"])+"_p_deltadoping",0.0))
             
             #actual layer
             layers_p.append((str(input_dict["setup"][slab_key]["strain"]),input_dict["setup"][slab_key]["width"]))
             
             #delta doping after the layer
             if input_dict["setup"][slab_key]["polarization"] == "positive":
                  layers_p.append((str(input_dict["setup"][slab_key]["strain"])+"_p_deltadoping",0.0))
             else: 
                  layers_p.append((str(input_dict["setup"][slab_key]["strain"])+"_n_deltadoping",0.0))
             
         #the last slab
         slab_key = "slab"+str(len(input_dict["setup"]))
         if input_dict["setup"][slab_key]["polarization"] == "positive":
             layers_p.append((str(input_dict["setup"][slab_key]["strain"])+"_n_deltadoping",0.0))
         else: 
             layers_p.append((str(input_dict["setup"][slab_key]["strain"])+"_p_deltadoping",0.0))
         layers_p.append((str(input_dict["setup"][slab_key]["strain"]),input_dict["setup"][slab_key]["width"]))
         
         delta_x = input_dict["delta_x"]
         slab = Slab(layers_p, materials_properties=mat_properties, delta_x = delta_x,
             smearing=smearing, beta_eV=beta_eV)
         
         res = run_simulation(slab = slab, max_steps = max_steps, nb_states = input_dict["nb_of_states_per_band"], smearing=smearing, beta_eV=beta_eV, b_lat=b_lat, delta_x=delta_x)

         print("\n")
         print("Convergence reached after %s iterations." %res[0][0])
         print("Voltage convergence parameter: %s" % res[0][1])
         print("Fermi energy convergence parameter: %s" % res[0][2])
         print("Total number of free electrons: %s (1/cm)" % res[0][5])
         print("Total number of free holes: %s (1/cm)" % res[0][6])
         print("\n")
                  
         #writing results into files
         out_files['general_info'] = {
             'filename': 'general_info.txt',
             'description': "General information",
             'data': n.atleast_2d(res[0]),
             'header': '1: Nb of iterations, 2: Voltage conv param, 3: Fermi energy conv param, 4: delta_x (ang), 5: Fermi energy (eV), 6: Total free electron density (1/cm), 7: Total free holes density (1/cm), 8: Total free electron density (1/b), 9: Total free holes density (1/b)'
             }

         out_files['band_data'] = {
             'filename': 'band_data.txt',
             'description': "Band data",
             'data': n.transpose(res[1][0]),
             'header': "1: position (ang), 2: Fermi energy (eV), the rest is organized as follow for each band:\n  First column is the potential profile of the band (in eV). The next pairs of columns are the wave function and the energy (eV) of the band's states"
             }

         out_files['density_profile'] = {
             'filename': 'density_profile.txt',
             'description': "Free-carrier density",
             'data': n.transpose(res[2]),
             'header': "1: position (ang), 2: Free electrons density (1/cm^2), 3: Free holes density (1/cm^2) "    
         }
    elif calculation_type == "map":
         print("\n")
         print("Starting map calculation...")
         print("\n")
         
         #build arrays containings the strains and the widths
         
         #strain
         strain_array = n.arange(input_dict["strain"]["min_strain"],input_dict["strain"]["max_strain"]+0.5*input_dict["strain"]["strain_step"],input_dict["strain"]["strain_step"])
         
         #width
         width_array = n.arange(input_dict["width"]["min_width"],input_dict["width"]["max_width"]+0.5*input_dict["width"]["width_step"],input_dict["width"]["width_step"])
         
         #updating the materials properties for the new strains
         plotting_the_fit = False
         if plot_fit == True:
             plotting_the_fit = True
             
         for strain in strain_array:
             update_mat_prop_for_new_strain(mat_prop = mat_properties, new_strain = strain, plot_fit = plotting_the_fit)
             plotting_the_fit = False
         
         #need to create a slab for each of those situation, run a simulation and retrieve the total carrier density
         data = n.zeros((strain_array.size*width_array.size,12))
         
         i = 0
         for strain in strain_array:
             for width in width_array:
             
                  layers_p = [("0.00",width/2.),(str(strain)+"_n_deltadoping",0.0), (str(strain),width*(1.+strain)), 
                               (str(strain)+"_p_deltadoping",0.0), ("0.00",width/2.)]
                  
                  #set delta_x so that the same number of steps are taken for each situation
                  delta_x = width * (2. + strain)/input_dict["nb_of_steps"]
                  if delta_x > input_dict["upper_delta_x_limit"]:
                      delta_x = input_dict["upper_delta_x_limit"]
                  
                  slab = Slab(layers_p, materials_properties=mat_properties, delta_x = delta_x,
                      smearing=smearing, beta_eV=beta_eV)
                  
                  print "Starting single-point calculation with strain = %s and width = %s ..." %(strain,width)
                  res = run_simulation(slab = slab, max_steps = max_steps, nb_states = 10, #arbitrary nb of states since not of interest here
                      smearing = smearing, beta_eV=beta_eV, b_lat=b_lat, delta_x=delta_x)
                  print "\n"
                  
                  data[i] = [strain, width , res[0][7], res[0][8], width*(1.+strain), res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]]
                  i += 1
         
         #saving the data in a file
         
         out_files['map_data'] = {
             'filename': 'map_data.txt',
             'description': "Map data",
             'data': data,
             'header': "1: strain, 2: width of unstrained slab (ang), 3: total electron density (1/b), 4: total holes density (1/b) 5: width of the strained slab (ang), 6: nb of iterations, 7: potential conv param, 8: Fermi energy conv param, 9: delta_x (ang), 10: Fermi energy (eV) 11: total electron density (1/cm), 12: total holes density (1/cm)"
             }
         
    else:
        raise ValidationError("The Calculation must either be 'single-point' or 'map'")

    return {'out_files': out_files}


if __name__ == "__main__":
    # Read files
    try:
        json_matprop = sys.argv[1]
        calc_input = sys.argv[2]
    except IndexError:
        print >> sys.stderr, ("Pass two parameters, containing the JSON files with the materials properties and the calculation input")
        sys.exit(1)
    try:
        with open(json_matprop) as f:
            matprop = json.load(f)
    except IOError:
         print >> sys.stderr, ("Error: The material properties json file (%s) passed as argument does not exist" % json_matprop)
         sys.exit(1)
    except ValueError:
         print >> sys.stderr, ("Error: The material properties json file (%s) is probably not a valid JSON file" % json_matprop)
         sys.exit(1)
    try:
        with open(calc_input) as f:
            input_dict = json.load(f)
    except IOError:
         print >> sys.stderr, ("Error: The calculation input json file (%s) passed as argument does not exist" % calc_input)
         sys.exit(1)
    except ValueError:
         print >> sys.stderr, ("Error: The material properties json file (%s) is probably not a valid JSON file" % json_matprop)
         sys.exit(1)
    
    try:
        retval = main_run(matprop=matprop, input_dict=input_dict)
    except ValidationError as e:
        print >> sys.stderr, "Validation error: {}".format(e)
        sys.exit(2)
    except InternalError as e:
        print >> sys.stderr, "Error: {}".format(e)
        sys.exit(3)

    out_files = retval['out_files']

    #folder where the output data will be printed
    out_folder = input_dict["out_dir"]
    if out_folder not in os.listdir(os.curdir):
        os.mkdir(out_folder)
  
    #writing results into files
    for file in sorted(out_files):
        filedata = out_files[file]
        fname = os.path.join(out_folder, filedata['filename'])

        n.savetxt(fname,filedata['data'],
            header=filedata['header'])
        print "{} saved in '{}'".format(
            filedata['description'],
            out_folder,
            fname,
            )

    # Disclaimer
    print "#"*72
    print "# If you use this code in your work, please cite the following paper:"
    print "# "
    print "# A. Bussy, G. Pizzi, M. Gibertini, Strain-induced polar discontinuities"
    print "# in 2D materials from combined first-principles and Schroedinger-Poisson"
    print "# simulations, Phys. Rev. B 96, 165438 (2017)."
    print "# DOI: 10.1103/PhysRevB.96.165438"
    print "#"*72
    


