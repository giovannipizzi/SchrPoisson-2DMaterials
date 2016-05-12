import numpy as n
import scipy
import scipy.linalg
import time
import scipy.special
import os

#Simulation specific parameters
#================================================================================================================

#Smearing, if yes how much ?
smearing = True
KbT = 0.005*13.6 #from Ry to eV
beta_eV = 1./KbT

#Strain. For now must aither be 0.03, 0.05 or 0.1
strain = 0.05

#the lattice constant in the y direction
b_lat = 4.288

#the polarization charge as a function of the strain in e/cm
if strain == 0.03 :
    delta_doping = 0.16841 * 1./b_lat * 1e8
    strained_material = 'Strained_3_SnSe'
elif strain == 0.05:
    delta_doping = 0.24717 * 1./b_lat * 1e8
    strained_material = 'Strained_5_SnSe'
elif strain == 0.1:
    delta_doping = 0.38643 * 1./b_lat * 1e8
    strained_material = 'Strained_10_SnSe'

#the list containg the mean slab length for computation
list_of_length = n.array([63.2548]) # can of course only contain 1 point

#Do you want a plot of the band and density profile at each step ?
plot_step = False

#folder where the detailed data of each step is printed
step_folder = "simulation_steps_data"
if step_folder not in os.listdir("./"):
    os.system("mkdir ./"+step_folder)

#folder where the summary data will be printed
summary_folder = "simulation_summary"
if summary_folder not in os.listdir("./"):
    os.system("mkdir ./"+summary_folder)


#Defining generalities
#================================================================================================================

#hbar^2/m0 in units of eV*ang*ang
HBAR2OVERM0=7.61996163
# periodicity, should remain true in this case
is_periodic = True

# grid spacing in ang; this is exact, and layer lengths are adapted to keep a constant
# step. This allows the second-derivative hamiltonian matrix to be symmetric
delta_x = 0.05

# Small threshold to check for charge neutrality
small_threshold = 1.e-6

#The code takes multiple bands into account, different materials must have the same number of bands and matchin degeneracy
#The lists given for the masses, band energies, degeneracy, etc must have the same order. Ex: the first element always refer to the Gamma point
mat_properties = { 
     'SnSe': {
        'condenergies': [0.6090962485,0.0902128713,0.05720314215],	# conduction energy levels in eV, in order: Gamma, Gamma-X, Gamma-Y
        'valenergies': [-1.508255819,-0.8832657543,-1.064317364],     	# valence energy levels in eV, in order: Gamma, Gamma-X, Gamma-Y
        'val_offset': 0.,  				                # offset in eV of the lowest valance state from some reference state
        'condmass':[2.741100107,0.110807373,0.131542031], 		# effective mass in the conduction states, in units of m0 
        'valmass':  [1.755925079, 0.125168454,0.109554071],     	# effective mass in the valence states, in units of m0 
        'conddosmass':[2.994158008,0.190387132,0.130378261], 		# mass used for the density of state calculation of the conduction states in unit of m0
        'valdosmass': [2.7330924,0.159070572,0.159511425], 		# mass used for the density of state calculation of the valence states in unit of m0
        'conddegeneracy' : [1,2,2],                       		# Degeneracy of the conduction band minima
        'valdegeneracy' : [1,2,2],                        		# Degeneracy of the valence band maxima
        'alpha': 4.*n.pi* 0.0055263496 * 10.22,                  	# the polarizabilit, 0.0055263496 = epsilon0 in e/(V*ang)
        'ndoping': 0 ,      			                	# the doping linear density, in e/cm
        },
    'Strained_3_SnSe': { # 3% strained SnSe
        'condenergies': [0.6005764308,0.098671565118,0.04524540703],	# conduction energy levels in eV, in order: Gamma, Gamma-X, Gamma-Y
        'valenergies': [-1.520311343,-1.034228618,-1.360574814],    	# valence energy levels in eV, in order: Gamma, Gamma-X, Gamma-Y
        'val_offset': 0.,  				                # offset in eV of the lowest valance state from some reference state
        'condmass':[1.790638381,0.132135785,0.151474703], 		# effective mass in the conduction states, in units of m0 
        'valmass':  [1.263123841, 0.149332816,0.142663316],     	# effective mass in the valence states, in units of m0 
        'conddosmass':[2.160064868,0.215656479,0.175190321], 		# mass used for the density of state calculation of the conduction states in unit of m0
        'valdosmass': [2.605644013,0.277787287,0.240072437], 		# mass used for the density of state calculation of the valence states in unit of m0
        'conddegeneracy' : [1,2,2],                       		# Degeneracy of the conduction band minima
        'valdegeneracy' : [1,2,2],                        		# Degeneracy of the valence band maxima
        'alpha': 4.*n.pi* 0.0055263496 * 8.09,                 		# the polarizability, 0.0055263496 = epsilon0 in e/(V*ang)
        'ndoping': 0 ,      			                	# the doping linear density, in e/cm
        },
    'Strained_5_SnSe': { # 5% strained SnSe
        'condenergies': [0.5973600272,0.09518500351,0.048095001],	# conduction energy levels in eV, in order: Gamma, Gamma-X, Gamma-Y
        'valenergies': [-1.530586988,-1.129146143,-1.52902204],     	# valence energy levels in eV, in order: Gamma, Gamma-X, Gamma-Y
        'val_offset': 0.,  				                # offset in eV of the lowest valance state from some reference state
        'condmass':[1.447647577,0.14613408,0.165506803], 		# effective mass in the conduction states, in units of m0 
        'valmass':  [1.072817073, 0.164708046,0.170014663],     	# effective mass in the valence states, in units of m0 
        'conddosmass':[1.901549047,0.215656479,0.209513646], 		# mass used for the density of state calculation of the conduction states in unit of m0
        'valdosmass': [2.5963443,0.396728963,0.314896267], 		# mass used for the density of state calculation of the valence states in unit of m0
        'conddegeneracy' : [1,2,2],                       		# Degeneracy of the conduction band minima
        'valdegeneracy' : [1,2,2],                       		# Degeneracy of the valence band maxima
        'alpha': 4.*n.pi* 0.0055263496 * 7.095,                  	# the polarizability, 0.0055263496 = epsilon0 in e/(V*ang)
        'ndoping': 0 ,      			                	# the doping linear density, in e/cm
        },
    'Strained_10_SnSe': { # 10% strained SnSe
        'condenergies': [0.5965998586,0.073992319300,0.082877535],	# conduction energy levels in eV, in order: Gamma, Gamma-X, Gamma-Y
        'valenergies': [-1.558452487,-1.350763729,-1.862302701],     	# valence energy levels in eV, in order: Gamma, Gamma-X, Gamma-Y
        'val_offset': 0.,  				                # offset in eV of the lowest valance state from some reference state
        'condmass':[0.98472771,0.182798895,0.205831494], 		# effective mass in the conduction states, in units of m0 
        'valmass':  [0.804425486, 0.205658144,0.271355674],     	# effective mass in the valence states, in units of m0 
        'conddosmass':[1.593144026,0.213362934,0.3196511], 		# mass used for the density of state calculation of the conduction states in unit of m0
        'valdosmass': [2.747762669,1.059071873,0.688994731], 		# mass used for the density of state calculation of the valence states in unit of m0
        'conddegeneracy' : [1,2,2],                       		# Degeneracy of the conduction band minima
        'valdegeneracy' : [1,2,2],                        		# Degeneracy of the valence band maxima
        'alpha': 4.*n.pi* 0.0055263496 * 5.40,                 		# the polarizability. The prefactor is in ang, 0.0055263496 = epsilon0 in e/(V*ang)
        'ndoping': 0 ,      			               		# the doping linear density, in e/cm
        },
    'n_deltadoping': {
        'ndoping': delta_doping,     # IN e/cm, If a layer has thickness zero, only the doping needs to be defined 
        },
    'p_deltadoping': {
        'ndoping': -delta_doping,     # In e/cm, If a layer has thickness zero, only the doping needs to be defined
        }
    }


#The general classes and methods

class Slab(object):
    def __init__(self,layers,materials_properties = mat_properties):
        """
        Pass a suitable xgrid (containing the sampling points in units of angstroms) and
        a (in angstroms)
        """
        if len(layers) == 0:
            raise ValueError("layers must have at least one layer")

        self.max_step_size = 0.8
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

            materials.append(materials_properties[l[0]])        
            #print sum(len(i) for i in xgrid_pieces[:-1]), sum(len(i) for i in xgrid_pieces)
            total_length += layer_length

        self._xgrid = n.concatenate(xgrid_pieces)

        # A check that all steps are equal; I calculate the error of each step w.r.t. the
        # expected step delta-x
        steps_error = (self._xgrid[1:] - self._xgrid[:-1]) - delta_x
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
        for i in range(self._ncond_min) :
            last_idx = 0
            for mat, grid_piece in zip(materials, xgrid_pieces):
                if len(grid_piece)!=0: # skip delta dopings
                    self._condband[i,last_idx:last_idx+len(grid_piece)] = mat['val_offset'] + mat['condenergies'][i]
                    self._condmass[i,last_idx:last_idx+len(grid_piece)] = mat['condmass'][i]
                    self._conddosmass[i,last_idx:last_idx+len(grid_piece)] = mat['conddosmass'][i]
                last_idx += len(grid_piece)
        """        
        print "condband : ", self._condband
        print "condmass : ", self._condmass  
        print 'conddosmass : ',   self._conddosmass
        """
        # valence band
        for j in range(self._nval_max) :
            last_idx = 0
            for mat, grid_piece in zip(materials, xgrid_pieces):
                if len(grid_piece)!=0: # skip delta dopings
                    self._valband[j,last_idx:last_idx+len(grid_piece)] = mat['val_offset'] + mat['valenergies'][j]
                    self._valmass[j,last_idx:last_idx+len(grid_piece)] = mat['valmass'][j]
                    self._valdosmass[j,last_idx:last_idx+len(grid_piece)] = mat['valdosmass'][j]
                last_idx += len(grid_piece)  
        """        
        print "valband : ", self._valband
        print "valmass : ", self._valmass  
        print 'valdosmass : ',   self._valdosmass             
        """ 

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
        
        # memory for convering algorithm
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
    
    def update_computing_times(self,process,value) :
	"""
	updates the time spent on a numerical process (string) either "Fermi", "Hami", "Poisson"
	"""
	if process == "Fermi":
	    self._time_Fermi += value
	elif process == "Poisson":
	    self._time_Poisson += value
	elif process == "Hami" :
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
        import schrpoisson_wire2 as spw        
        
        self._counter += 1
        self._Ef[self._counter%2] = e_fermi
        
        max_iteration = 5000

        V_conv_threshold = 2.e-4
	    
        Ef_conv_threshold = 1.e-6 # if fermi energy does not change from one iteration to another, converged

        free_electrons_density = get_electron_density(c_states, e_fermi, self._conddosmass, self.npoints, self._conddegen)
        free_holes_density = get_hole_density(v_states, e_fermi, self._valdosmass, self.npoints, self._valdegen)
	
        total_charge_density =  self._doping - free_electrons_density + free_holes_density
	
	#updating the time spent solving Poisson
	start_t = time.time()
        if is_periodic:
			new_V = -1.*spw.periodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0] # minus 1 because function returns electrostatic potential, not energy

        else:
			new_V = -1.*spw.nonperiodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0]
        end_t = time.time()
	self._time_Poisson += end_t-start_t
	
        new_V -= np.mean(new_V)
        
        if self._counter == 1 :
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
            
        else :
            self._subcounter += 1
        
        if self._subcounter == 20 :
            self.max_step_size *= 1.4
            self._subcounter = 0
                    
        step = new_V - self._V
        current_max_step_size = n.max(n.abs(step))
      
        #convergence check
        self._over = False
        if current_max_step_size < V_conv_threshold :
			start_t = time.time()
			if is_periodic:
				check_V = -1.*spw.periodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0] # minus 1 because function returns electrostatic potential, not energy

			else:
				check_V = -1.*spw.nonperiodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0]
			end_t = time.time()
			self._time_Poisson += end_t-start_t
			check_val = n.max(n.abs(check_V-self._V))
			self._indicator[self._counter%2] = check_V[self._max_ind]-self._V[self._max_ind]
			if check_val > 5*V_conv_threshold :
				current_max_step_size = check_val
				step = check_V - self._V
				#self.max_step_size *= 0.5
			else :
					
				self._over = True

				
			
        print 'convergence param:', current_max_step_size	
        
        if current_max_step_size != 0 and  self._over == False :
			#self._V += step * min(self.max_step_size, current_max_step_size) #/ (current_max_step_size)
			self._V += step * self.max_step_size
			self._old_V = self._V.copy()
        elif current_max_step_size == 0 and  self._over == False:
			self._V = new_V
			self._old_V = self._V.copy()
		
        elif self._over == True :
            self._V = self._old_V
            print "Final convergence parameter : ", check_val
            self._finalV_check = check_val
		
        if n.abs(self._Ef[0]-self._Ef[1]) <= Ef_conv_threshold and current_max_step_size < 10*V_conv_threshold:
            self._E_count += 1
            
            if self._E_count  == 4 :
                print "Convergence of Fermi energy : ", n.abs(self._Ef[0]-self._Ef[1])
                current_max_step_size = 0.1*V_conv_threshold # froced convergence if that happens 4 times in a row
                if is_periodic:
                    check_V = -1.*spw.periodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0] # minus 1 because function returns electrostatic potential, not energy

                else:
                    check_V = -1.*spw.nonperiodic_recursive_poisson(self._xgrid,total_charge_density,self._alpha,max_iteration)[0]
                check_val = n.max(n.abs(check_V-self._V))
                print "Final convergence parameter : ", check_val
                self._finalV_check = check_val
        else :
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
        for i in range(self._ncond_min) :
			V[i,:] = self._V
        return self._condband + V

    def get_valence_profile(self):
        """
        Return the valence band profile in eV
        """
        # need dimensions to agree
        V = n.zeros((self._nval_max,len(self._xgrid)))
        for i in range(self._nval_max) :
			V[i,:] = self._V
        return self._valband + V
        
    def get_band_gap(self):    
        """
        Scans valence and conduction profiles in order to find the absolute conduction minimum and the absolute valence band maximum and returns the difference
        """
        conduction = self.get_conduction_profile()
        valence = self.get_valence_profile()
        
        cond_min = np.min(conduction)
        val_max = np.max(valence)
        
        return cond_min - val_max
        

def MV_smearing(E,beta,mu):
    """
    Marzari Vanderbilt smearing function to be integrated in conjuction with the density of states
    Carefull units of beta,E and mu must be consistent
    """
    
    return 0.5*scipy.special.erf(-beta*(E-mu)-1./n.sqrt(2)) + 1./n.sqrt(2.*n.pi)*n.exp(-(beta*(E-mu)+1./n.sqrt(2))**2) + 0.5
    

def get_electron_density(c_states, e_fermi, c_mass_array, npoints,degen, band_contribution = False, avg_eff_mass = False):
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
    D = n.sqrt(2.) / n.pi / sqrt(HBAR2OVERM0)

    el_density = n.zeros(npoints)   
    
    contrib = n.zeros(len(degen))    
    
    avg_mass = np.zeros((1,3))
    
    # All the conduction band minima have to be taken into account with the appropriate degeneracy
    for j in range(len(degen)) : #number of minima
        deg = degen[j]
	if j > 0 and avg_eff_mass == True:
	    avg_mass = n.append(avg_mass,[[0.,0.,0.]],axis=0) # so that bands are separated by a line of zeros
        for state_energy, state in c_states[j]:
	    energy_range = 10. # eV, to be very safe
	    
            #if state_energy > e_fermi:
            #    continue
            square_norm = sum((state)**2)
            # I average the inverse of the effective mass
            # Both state and c_mass_array should have the same length
            # NOT SURE: square_norm or sqrt(square_norm) ? AUGU : I'm pretty sure it's square_norm and I changed it
            averaged_eff_mass = 1./(sum(state**2 / c_mass_array[j]) / square_norm)
	    if avg_eff_mass == True:
		avg_mass = n.append(avg_mass,[[state_energy,state_energy-e_fermi,averaged_eff_mass]],axis=0)
	    
	    if not smearing and state_energy < e_fermi:
		# At T=0, integrating from E0 to Ef the DOS gives
		# D * sqrt(meff) * int_E0^Ef 1/(sqrt(E-E0)) dE =
		# D * sqrt(meff) * 2 * sqrt(Ef-E0)   [if Ef>E0, else zero]
		el_density += deg * D * sqrt(averaged_eff_mass) * 2. * sqrt(e_fermi - state_energy) * (
		    state**2 / square_norm)
		contrib[j] += sum(deg * D * sqrt(averaged_eff_mass) * 2. * sqrt(e_fermi - state_energy) * (
		    state**2 / square_norm))
	   
	    elif smearing and state_energy-e_fermi < energy_range : # more than enough margin
		# Need to numerically integrate the density of state times the occupation given by MV_smearing
		# to compute the integral, one uses the trick explained there to avoid singularities : http://math.stackexchange.com/questions/1351734/numerical-integration-of-divergent-function
		n_int = 5000. # number of intervals
		# change of variable E = state_energy + t**2
		dt = n.sqrt(energy_range)/n_int 
		t = n.arange(0,n.sqrt(energy_range),dt) 
		#plt.figure(1)
		#plt.plot(energy,deg * D * sqrt(averaged_eff_mass) * 1./n.sqrt(energy-state_energy),"b")
		#plt.plot(energy,MV_smearing(energy,beta_eV,e_fermi),"r")
		#plt.plot(energy,g_times_f,"k")
		#plt.title("%s"%e_fermi)
		#plt.show()
		temp_dens = 2*deg * D * sqrt(averaged_eff_mass) * n.trapz(MV_smearing(state_energy+t**2,beta_eV,e_fermi),dx=dt) * (state**2 / square_norm)
		el_density += temp_dens
		contrib[j] += sum(temp_dens)
		
		
    # Up to now, el_density is in 1/ang; we want it in 1/cm
    if band_contribution == False :
        return el_density * 1.e8
    elif band_contribution == True and avg_eff_mass == False :
        return el_density * 1.e8, contrib  * 1.e8
    else :
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
    return n.min(np.delete(all_cond_states_energies,0)) - n.max(np.delete(all_val_states_energies,0))

def get_hole_density(v_states, e_fermi, v_mass_array, npoints, degen, band_contribution = False, avg_eff_mass = False):
    """
    For all documentation and comments, see get_electron_density
    
    v_mass_array should contain the DOS mass of holes, i.e., positive
    
    degen is the array containing the degeneracy of each valence band maximum

    Return a positive number
    """
    D = n.sqrt(2.) / n.pi / sqrt(HBAR2OVERM0)
    
    h_density = n.zeros(npoints)  
    
    avg_mass = np.zeros((1,3))
    
    contrib = n.zeros(len(degen))
    for j in range(len(degen)) :
        deg = degen[j]
	if j > 0 and avg_eff_mass == True:
	    avg_mass = n.append(avg_mass,[[0.,0.,0.]],axis=0) # so that bands are separated by a line of zeros
        for state_energy, state in v_states[j]:
	    energy_range = 10. # eV to be extra safe
	    
            # Note that here the sign is opposite w.r.t. the conduction case
            #if state_energy < e_fermi:
            #    continue
            square_norm = sum((state)**2)
            averaged_eff_mass = 1./(sum(state**2 / v_mass_array[j]) / square_norm)
	    if avg_eff_mass == True:
		avg_mass = n.append(avg_mass,[[state_energy,state_energy-e_fermi,averaged_eff_mass]],axis=0)
		
	    if not smearing and state_energy > e_fermi:
		h_density += deg * D * sqrt(averaged_eff_mass) * 2. * sqrt(state_energy - e_fermi) * (
		    state**2 / square_norm)
		contrib[j] += sum(deg * D * sqrt(averaged_eff_mass) * 2. * sqrt(state_energy - e_fermi) * (
		    state**2 / square_norm))    
		    
	    
	    elif smearing and e_fermi-state_energy < energy_range:
		
		n_int = 5000. # number of intervals
		# change of variable E = state_energy + t**2
		dt = n.sqrt(energy_range)/n_int 
		t = n.arange(0,n.sqrt(energy_range),dt) 
		temp_dens = 2*deg * D * sqrt(averaged_eff_mass) * n.trapz(MV_smearing(2*e_fermi-state_energy+t**2,beta_eV,e_fermi),dx=dt) * (state**2 / square_norm)
		h_density += temp_dens
		contrib[j] += sum(temp_dens)
		
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
    if band_contribution == False :
        return h_density * 1.e8
    elif band_contribution == True and avg_eff_mass == False:
        return h_density * 1.e8, contrib * 1.e8
    else:
	return  h_density * 1.e8, contrib * 1.e8, avg_mass
    

def find_efermi(c_states, v_states, c_mass_array, v_mass_array, c_degen, v_degen, npoints, target_net_free_charge):
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
    all_states_energies = np.delete(all_states_energies,0)
    # I set the boundaries for the bisection algorithm; I could in principle
    # also extend these ranges
    ef_l = all_states_energies.min()-more_energy
    ef_r = all_states_energies.max()+more_energy

    electrons_l = n.sum(get_electron_density(c_states, ef_l, c_mass_array,npoints,c_degen))
    holes_l = n.sum(get_hole_density(v_states, ef_l, v_mass_array, npoints,v_degen))
    electrons_r = n.sum(get_electron_density(c_states, ef_r, c_mass_array, npoints,c_degen))
    holes_r = n.sum(get_hole_density(v_states, ef_r, v_mass_array,npoints,v_degen))
    
    net_l = holes_l - electrons_l
    net_r = holes_r - electrons_r
    if (net_l - target_net_free_charge) * (
        net_r - target_net_free_charge) > 0:
        raise ValueError("The net charge at the boundary of the bisection algorithm "
                         "range has the same sign! {}, {}, target={}; ef_l={}, ef_r={}".format(
                             net_l, net_r, target_net_free_charge,ef_l, ef_r))
    absdiff = 10*charge_precision
    en_diff = ef_r - ef_l
    while en_diff > energy_precision : 
        ef = (ef_l + ef_r)/2.
        electrons = n.sum(get_electron_density(c_states, ef, c_mass_array,npoints,c_degen))
        holes = n.sum(get_hole_density(v_states, ef, v_mass_array,npoints,v_degen))
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
        if  absdiff > charge_precision :
            #need to go over the loop once more
            en_diff = 10*energy_precision
    return ef #(ef_r + ef_l)/2. it changes everything since the bissection algorithm worked for ef and not for (ef_r + ef_l)/2.

def get_conduction_states_p(slab):
    """
    This function diagonalizes the matrix using the fact that the matrix is banded. This provides
    a huge speedup w.r.t. to a dense matrix.

    NOTE: _p stands for the periodic version
    """
    more_bands_energy = 0.5 # How many eV to to above the top of the conduction band
    
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
    for i in range(slab._ncond_min) : 
	
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
        H[i,2,:][reordering]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
        # mass_differences[1+1] is the average mass between 1 and 2
        H[i,2,:][reordering]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences[
            (arange(slab.npoints)+1) % slab.npoints]

        # note! The matrix is symmetric only if the mesh step is identical for all steps
        # I use 1: in the second index because the upper diagonal has one element less, and
        # this is the format required by eig_banded
        # Note that this also sets superdiagonal[0] to the correct element that should be at
        # the corner of the matrix, at position (n-1, 0)
        superdiagonal = - (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
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
    
    NOTE: _p stands for the non-periodic version
    """
    more_bands_energy = 0.5 # How many eV to to below the bottom of the valence band
    
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
    
        H[i,2,:][reordering]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
        H[i,2,:][reordering]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences[
            (arange(slab.npoints)+1) % slab.npoints]

        superdiagonal = - (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
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
    more_bands_energy = 0.5 # How many eV to to above the top of the conduction band
    
    # The list containing the tuples that will be returned
    res = []
    
    # Ham matrix in eV; I store only the first upper diagonal
    # H[1,:] is the diagonal,
    # H[0,1:] is the first upper diagonal
    H = n.zeros((slab._ncond_min,2, slab.npoints))
    
    for i in range(slab._ncond_min) : 
            
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
        H[i,1,1:] += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
        H[i,1,:-1]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences

        # note! The matrix is symmetric only if the mesh step is identical for all steps
        # I use 1: in the second index because the upper diagonal has one element less, and
        # this is the format required by eig_banded
        H[i,0,1:] = - (HBAR2OVERM0/2.) / delta_x**2 / mass_differences

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
    more_bands_energy = 0.5 # How many eV to to below the bottom of the valence band
    
    # The list containing the tuples that will be returned
    res = []
    
    H = n.zeros((slab._nval_max,2, slab.npoints))
    
    for i in range(slab._nval_max) :
        
        H[i,1,:] = slab.get_valence_profile()[i]
        min_energy = H[i,1,:].min() - more_bands_energy
        max_energy = H[i,1,:].max() 

        # In the valence bands, it is as if the mass is negative
        mass_differences = -(slab._valmass[i,1:] + slab._valmass[i,:-1])/2.

        H[i,1,1:] += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences
        H[i,1,:-1]  += (HBAR2OVERM0/2.) / delta_x**2 / mass_differences

        H[i,0,1:] = - (HBAR2OVERM0/2.) / delta_x**2 / mass_differences

        w, v = scipy.linalg.eig_banded(H[i], lower=False, eigvals_only=False, 
                                    overwrite_a_band=True, # May enhance performance
                                    select='v', select_range=(min_energy,max_energy),
                                    max_ev=slab.npoints) # max_ev: max # of eigenvalues to expect
                                                         # I use the worst case scenario, where I get
                                                         # all of them
        res.append(zip(w, v.T))
        
    return res
  

def run_simulation(length1,length2, max_steps, material1, material2, b_lat, plot_step = False) :
    """
    This function launches the self/consistant Schroedinger-Poisson calculations and returns all the relevant data
    
    material1 and material2 are strings that indicate which material to choose from the dictionary
    
    Note : only used for periodic systems with delta-doping
    """ 
    
    it = 0
    mean_length = 0.5*(length1+length2)
    print 'Mean slab length : ', mean_length
    layers_p = [
    (material1,0.5*length1),
    ('n_deltadoping',0.0),
    (material2,length2),
	('p_deltadoping',0.0),
	(material1,0.5*length1),
    ]
    
    slab = Slab(layers_p)
    
    # I don't want the terminal to be flooded with every single step
    import sys
    import os
    
    sys.stdout = open(os.devnull, "w")
    
    try:
        converged = False
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
                                  target_net_free_charge=slab.get_required_net_free_charge())
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
    sys.stdout = sys.__stdout__
    if not converged:
        print "****** WARNING! Calculation not converged ********"    
    
    
    #should print all results at each step in files
    el_density, el_contrib, avg_cond_mass = get_electron_density(c_states, e_fermi, slab._conddosmass, slab.npoints, slab._conddegen, band_contribution = True, avg_eff_mass = True)
    hole_density, hole_contrib, avg_val_mass = get_hole_density(v_states, e_fermi, slab._valdosmass,slab.npoints, slab._valdegen, band_contribution = True, avg_eff_mass =True)
    tot_el_dens = n.sum(el_density)
    tot_hole_dens = n.sum(hole_density)
    tot_el_dens_2 = tot_el_dens * b_lat*1.e-8 # in units of e/unit cell, 
    tot_hole_dens_2 = tot_hole_dens * b_lat*1.e-8 # in units of e/unit cell
    
    #contribution in %
    if tot_el_dens != 0 : 
        el_contrib /= tot_el_dens
    if tot_hole_dens != 0 :    
        hole_contrib /= tot_hole_dens
	print "El contrib : ", el_contrib
	print "Hole contrib : ", hole_contrib
    matrix = [slab.get_xgrid(),ones(slab.npoints) * e_fermi]
    zoom_factor = 10. # To plot eigenstates
		
    i=0
    j=0
    for k in range(slab._ncond_min) :
        matrix.append(slab.get_conduction_profile()[k])
        for w, v in c_states[k]:
            matrix.append(w + zoom_factor * abs(v)**2)
            matrix.append(ones(slab.npoints) * w)
            i+=1
    for l in range(slab._nval_max):        
        matrix.append(slab.get_valence_profile()[l])
        for w, v in v_states[l]:
            # Plot valence bands upside down
            matrix.append(w - zoom_factor * abs(v)**2)
            matrix.append(ones(slab.npoints) * w)
            j+= 1
    
    if plot_step == True :

	# Plotting each band and corresponding states in a matching color
	plot(slab.get_xgrid(), slab.get_conduction_profile()[0],'r',linewidth=2)

	for w, v in c_states[0]:
	   plot(slab.get_xgrid(), w + zoom_factor * abs(v)**2,'r')     
	 
	plot(slab.get_xgrid(), slab.get_conduction_profile()[1],'g',linewidth=2)
    
	for w, v in c_states[1]:
	   plot(slab.get_xgrid(), w + zoom_factor * abs(v)**2,'g')
	    
	plot(slab.get_xgrid(), slab.get_conduction_profile()[2],'m',linewidth=2)
    
	for w, v in c_states[2]:
	   plot(slab.get_xgrid(), w + zoom_factor * abs(v)**2,'m')   
	    
	       
	 
	plot(slab.get_xgrid(), slab.get_valence_profile()[0],'b',linewidth=2)
		
	for w, v in v_states[0]:
		    # Plot valence bands upside down
	    plot(slab.get_xgrid(), w - zoom_factor * abs(v)**2,'b')     
	
	plot(slab.get_xgrid(), slab.get_valence_profile()[1],'c',linewidth=2)
		
	for w, v in v_states[1]:
		    # Plot valence bands upside down
	    plot(slab.get_xgrid(), w - zoom_factor * abs(v)**2,'c')    
	    
	plot(slab.get_xgrid(), slab.get_valence_profile()[2],'y',linewidth=2)
		
	for w, v in v_states[1]:
		    # Plot valence bands upside down
	    plot(slab.get_xgrid(), w - zoom_factor * abs(v)**2,'y')    
		    
	
	plot(slab.get_xgrid(), ones(slab.npoints) * e_fermi, 'k--')
	xlabel("x (ang)")
	ylabel("eV")
	
	# density profile plot
	
	figure(2)
	plot(slab.get_xgrid(), el_density,label='e')
	plot(slab.get_xgrid(), hole_density,label='h')

	legend()
    
	xlabel("x (ang)")
	ylabel("el/hole density (1/cm)")
	
	show()
    
    #Keeping the user aware of the time spend on each main task
    print "Total time spent solving Poisson equation : ", slab.get_computing_times()[0], " (s)"
    print "Total time spent finding the Fermi level(s) : ", slab.get_computing_times()[1], " (s)"
    print "Total time spent computing the electronic states : ", slab.get_computing_times()[2] , " (s)"
    
    #files saved in a subfolder  
    np.savetxt('./'+step_folder+'/SnSe_strain'+str(100*strain)+'_leng'+str(mean_length)+'_average_masses.txt',np.append(avg_val_mass,avg_cond_mass,axis=0),header='#1 : state energy (eV), 2: state energy - e_fermi (eV), average DOS effective mass (units of m0) \n #bands are separated by a line of zeros' )      
    np.savetxt('./'+step_folder+'/SnSe_strain'+str(100*strain)+'_leng'+str(mean_length)+'.txt',np.transpose(matrix),header='#1: position (ang), 2: fermi energy (eV), the rest is simulation dependent but contains the band profiles, the states and their energies')
    np.savetxt('./'+step_folder+'/SnSe_strain'+str(100*strain)+'_leng'+str(mean_length)+'_dens.txt', np.transpose([slab.get_xgrid(),el_density,hole_density,el_density* b_lat*1.e-8,hole_density* b_lat*1.e-8]),header='#1: position (ang), 2: electron density (1/cm), 3: hole density (1/cm), 4: electron density (1/a), 5: hole density (1/a)')
    
    return [it, slab._finalV_check, slab._finalE_check, tot_el_dens, tot_hole_dens,i,j,delta_x,mean_length,tot_el_dens_2,tot_hole_dens_2, hole_contrib[0], hole_contrib[1], hole_contrib[2], el_contrib[0], el_contrib[1], \
               el_contrib[2], e_fermi ]

if __name__ == "__main__":
    from pylab import *
    
    num = list_of_length.size
    matrix = n.zeros((18,num))
    i = 0
    for leng in list_of_length :  
        
	#update delta_x so that calculation time remains ok (min 200 pts)
	
	if leng < 30. :
	    delta_x = 0.2
	elif leng < 90 :
	    delta_x = 0.25
	elif leng < 150 :
	    delta_x = 0.4
	else :
	    delta_x = 0.5
	
	leng1 = 2*leng/(2+strain) #unstrained length
	leng2 = 2*leng*((1+strain)/(2+strain))
	#print leng1,leng2,leng1+leng2-2*leng

        res = run_simulation(leng1,leng2, 2000,'SnSe',strained_material,b_lat,plot_step)
        print res
        matrix[:,i] = res
	i += 1
    
    np.savetxt('./'+summary_folder+'/SnSe_strain'+str(100*strain)+'_from'+str(list_of_length[0])+'_to'+str(list_of_length[-1])+'.txt',np.transpose(matrix),header='1: Nb iterations, 2: V conv param, 3: E conv param, 4: tot el dens (1/cm), 5: tot hole dens (1/cm), 6: nb of el states, \
        7:nb of hole states, 8: step size (ang), 9: mean slab length (ang), 10: tot el dens (1/a), 11: tot hole dens (1/a), 12: contributioon of Gamma val band max, 13: contribution of Gamma-X val band max, \
        14: contribution of Gamma-Y val band max, 15: contributioon of Gamma cond band min, 16: contributioon of Gamma-X cond band min, 17: contributioon of Gamma-Y cond band min,  18: fermi energy (eV) ')
    
    
   
    

