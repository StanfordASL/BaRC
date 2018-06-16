# BaRC: Backward Reachability Curriculum for Robotic Reinforcement Learning

## Programming Environment ##
```
git clone https://github.com/StanfordASL/BaRC.git
cd BaRC             # or wherever you cloned the repo

# openai/baselines requires Python 3.5, so we enforce it too.
conda create --name backreach python=3.5
source activate backreach
pip install numpy matplotlib scipy
```

## Dependencies ##
This repository depends on [OpenAI Gym](https://github.com/openai/gym), [OpenAI Baselines](https://github.com/openai/baselines), and [OpenMPI](https://www.open-mpi.org) (because of Baselines). Further, our code requires MATLAB as well as the [helperOC](https://github.com/HJReachability/helperOC) and [Level Set Methods](http://www.cs.ubc.ca/~mitchell/ToolboxLS) toolboxes for backward reachability computations.

For a minimal installation, you can install OpenAI Gym and Baselines like so:
```
cd gym
pip install -e '.[mujoco,atari,classic_control,robotics]'

cd ../code/baselines
pip install -e .
```

With those obtained, you must place our two gym environments `DrivingOrigin-v0` and `PlanarQuad-v0` located in the `gym/` folder into your gym installation. Instructions for how to do this can be found on the OpenAI gym website.

Then, place our modifications to the baselines PPO algorithm into your installation of OpenAI baselines. They are located under `code/baselines/`. For this, you simply replace the `ppo1` folder in the baselines repository with ours.

Now you're ready to use our code. `train.py` is the main runner interface which can be called with command line arguments.
