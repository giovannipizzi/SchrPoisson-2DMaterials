import json

## Change here if needed
#type_of_calc = "single_point"
type_of_calc = "map"


name_of_json_file = "example_calc_input.json"

if type_of_calc == "single_point" :
    input_dict =  {
        "calculation": "single_point",
        "smearing" : True,
        "KbT" : 0.005, 
        "delta_x" : 0.5,
        "max_iteration" : 1000,
        "nb_of_states_per_band" : 2,
        "plot_fit" : False,
        "out_dir" : "single_point_output",
        "setup" : { "slab1" : { "strain" : 0.00,
                                "width" : 10.0,
                                "polarization" : "positive",
                            },
                    "slab2" : { "strain" : 0.10,
                                "width" : 20.0,
                                "polarization" : "positive",
                            },
                    "slab3" : { "strain" : 0.00,
                                "width" : 10.0,
                                "polarization" : "positive",
                            },
                },
    }
    
    elif type_of_calc == "map" :
        
        input_dict =  {"calculation": "map",
                       "smearing" : True,
                       "KbT" : 0.005, 
                       "max_iteration" : 1000,
                       "plot_fit" : False,
                       "nb_of_steps" : 100,
                       "upper_delta_x_limit" : 0.5,
                       "out_dir" : "map_output",
                       "strain" : { "min_strain" : 0.0,
                                    "max_strain" : 0.1,
                                    "strain_step" : 0.05
                                },
                       "width" : { "min_width" : 10.0,
                                   "max_width" : 20.0,
                                   "width_step" : 10.0
                               }
                   }
    else:
        raise ValueError("Invalid value of 'type_of_calc'")

    with open(name_of_json_file,'w') as f:
        json.dump(input_dict,f,indent=2)
        
