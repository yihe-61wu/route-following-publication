# Insect Mushroom Body Models for Embodied Visual Route Following
## Overview
Inspired by insect mushroom body (MB), a brain structure well-known for rapid associative learning,
we develop models that performs vision-based locomotion after one-shot or online visual learning. 
This repository contains our first model for visual route following
(or visual teach-and-repeat), that is, given a route learned by a robot,
if and how the robot can follow the route accurately using only visual inputs.
which has been validated on robots in the `iGibson` simulator ([demo](https://youtu.be/iT_WYdoe24I)) and in the real world.
The main results are summarised in the two preprints below:
1. Yihe, L., Maroun, R. A., & Webb, B. (2023).                                                                                            
[Vision-based route following by an embodied insect-inspired sparse neural network](https://doi.org/10.48550/arXiv.2303.08109).            
arXiv preprint arXiv:2303.08109.                                                                                                                                                                                                                                                   
2. Lu, Y., Cen, J., Maroun, R. A., & Webb, B. (2024).                                                                                     
[Embodied visual route following by an insect-inspired robot](https://doi.org/10.21203/rs.3.rs-4222706/v1).                                

The first preprint was accepted as a poster for ICLR Workshop on Sparsity in Neural Networks,
and the second one has recently been accepted by _Journal of Bionic Engineering_.   


## How to use
### Installation and dependencies
Please install all dependencies required by [iGibson 2.0](https://stanfordvl.github.io/iGibson/installation.html) first.
Additionally with `numpy` and `matplotlib`, one can reproduce simulations and visualisations (given simulated data).

A `conda` virtual environment is recommended, because
1. we have been using it throughout development and testing, and
2. iGibson is built on various common packages that a user might rely on in other projects, such as `PyBullet` and `OpenCV`.

All the code has been developed and tested on an `Ubuntu 22.04 LTS` laptop 
in a `Python 3.8.13` environment with all packages specified in `requirements.txt`.


### List of main files
Core files for route following:
  * `route_following.py` (`route_*.py`) - the script for the experiment of route following,
  * `robot_main.py` - models to be compared with Model 1,
  * `robot_steering.py` - Model 1 (with variations),
  * `mushroom_body.py` - MB circuits (with variations) in Model 1.

Auxilary files:
  * `\yaml_robots\*.yaml` - the robot parameters,
  * `igibson_api.py` - the functions and objects interfacing our models and the simulator,
  * `neuro_utils.py` - the common, basic objects for building MB circuits,
  * `robot_sensory.py` - the common, basic objects for sensory inputs to the models,
  * `simulation_main.py` - the common script for initialising simulation, robots and models, with respect to various keyword parameters,
  * `video_recorder.py` - the object for recording data,
  * `analysis` in a file name - indicates the file is for data analysis and visualisation.


### Running simulations
In addition to the scripts mentioned in the previous section for the main navigation tasks, 
there are auxillary scripts such as `examine_robot_property.py` available for secondary purposes.
Nevertheless, these scripts share the same workflow:
1. **Importing modules**: While all the simulations are to take place within the simulator, iGibson, 
most of the scripts do not import iGibson modules explicitly,
because the necessary iGibson modules are integrated in `igibson_api.py`,
which is imported into other files for model configuration, etc.
_Remark: double importing iGibson modules would typically lead to running errors._
2. **Preparing simulation, model(s) and recorder(s)**: Preparing simulation is composed of a sequence of typical steps of 
initialising simulator, scene and robot(s), 
which are wrapped under `SimulationOrganisor` in `igibson_api.py`.
Depending on the truth value of `headless`, viewers showing live videos for human users are to be additionally initialised.
The steps of configuration and initialisation of our models are wrapped in `simulation_main.py`,
and the steps for the data recorder under `Recorder` in `video_recorder.py`.
_Remark: each `Recorder` object can record data from only a single robot (the one under the control of an MB model)._
3. **Repeating trials**: In the route following experiments, training and test trials are separated.
Within every trial, the route is to firstly put the robot back at the starting location
and start a new session for data recording,
then to begin the simulation, a loop in which robot and model states are varied and recorded,
and finally to terminate the loop when the robot reaches its target position or the time is up,
followed by saving the recorded data to files (and generating a video if `headless==True`) and starting a new loop (trial).
4. **Ending simulation**: The simulator is to be disconnected, and auxillary figures generated.
_Remark: The disconnection of the simulator is essential, 
especially when a user attempts to repeat simulations by running the `main` function of a script.
Failure to do so would rapidly overload computer memories.
In contrast, the generation of the auxillary figures is efficient but not essential._
                                         
We recommend `headless` to be set as `False` when running early-stage simulations or producing video demos,
but as `True` when running repetitive simulations (for data analysis),
because showing, recording and saving live videos add a significant amount of overhead.

### Configuring simulator, scene and robot(s)
A user can customise properties of simulator, scene and robot(s) by specifying `**kwargs`
for the `main` function of a script.
The values are passed to `kwargs_main`.
If not specified, default values in `simulation_main.py` are used.

Typically in our main scripts, default values for the iGibson simulator (its physics engine and visual renderer) are used.

There are multiple indoor interactive scenes available from iGibson.
The choice of a scene can be specified by setting `scene_name`.
If `scene_name=='random'`, a scene will be chosen randomly.

There are multiple robot models available from iGibson.
We have considered only the ones under `\yaml_robots`, because they are all differential wheeled robots with a front camera.
The properties of sensory input and motor control can be customised such as their noise level.

If a user needs to modify properties beyond the existing ones in `kwargs_main`,
they should familiarise themselves with the underlying iGibson packages, the `*.yaml` files and our models.
 
### Customising models
See the **Main models** section.

### Reproducing data analysis and visualisation
For completeness and reproducibility, this repository has not been tidied up, 
containing all files of functions, objects, scripts (mostly as `*.py` files)
and analysis (mostly as `*.ipynb` files),
including those for testing, explorative, or comparison purposes and even those to be deprecated in the future.
We note, however, `jupyter` is delibrately omitted from the dependency requirements for running simulation,
because we use a different `conda` virtual environment for data analysis.

In addition, we do not provide our simulation data due to their synthetic nature and excessive quantity.
By default, all the simulation scripts will save data under `\records`.
A user trying to reuse the `*.ipynb` files is recommended to pay attention to their directory paths.
A copy of the data of the real-world route following experiment are kept here, though, under `\RobotData`,
because they are analysed in `paper_analysis_realworld.ipynb`.


## Main models
### Code Structure
A full model consists of modules from, in a bottom-up order:
* `utils.py` and `igibson_api` - generic functions and simulator API,
* `robot_sensory.py` and `neuro_utils.py` - basic components for sensory inputs and MB circuit,
* `mushroom_body.py` - MB circuits (for learning only),
* `robot_steering.py` - full MB models (from sensory inputs to control outputs) for route following.

### Mushroom body (MB) circuit for visual learning and memory
Insect MB, a shallow circuit, is capable of rapid associative learning.
The MB, as a visual memory in our models, is assumed to be a two-layered neural network,
consisting of 
- visual projection neurons (PN), receiving preprocessed visual inputs,
- Kenyon cells (KC), encoding any PN activity as a latent, sparse pattern, by multiplying the PN-KC weight matrix,
- MB output neurons (MBON), computing visual familiarity/novelty of the KC pattern, by multiplying the KC-MBON weight matrix.
      
where 
- the PN-KC matrix is randomly initialised to be binary and sparse, and fixed throughout a simulated experiment; and
- the KC-MBON matrix is initialised to be 
  - either zero, if learning is achieved by synaptic enhancement (i.e., increasing weights),
  - or one, if learning is achieved by synaptic depression (i.e., decreasing weights, which is more consistent with the real MB).

The MB circuit (with all the variations) can be found in `mushroom_body.py` and `mushroom_body_new.py`.
Other than the specific learning rule, the most import parameters of these objects are 
`N_pn` (the number of PN), `N_pn_perkc` (the sparsity of the PN-KC connectivity), `N_kc` (the number of KC) and `S_kc` (the sparsity of KC activity).

This model is relatively a low-level controller, 
because it takes visual inputs (only) as input, and return control commands of linear and angular velocities (only) as output.
Running `route_following.py` with `model_name='lamb'` simulates this model in a route following task.
Other variations of the model and different models for comparison can be found in `simulation_main.py`.


### Licenses
You are welcome to (re-)use everything in this repository for non-commercial purposes,
but please cite us and redistribute any derivative work with the same licenses.
More explicitly:
- The software is distributed with the **GNU GPL 3.0** license.
- The real-world experimental data is distributed with the **CC BY-NC-SA 4.0** license.