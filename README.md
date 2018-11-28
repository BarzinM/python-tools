# Python Tools

The scripts in this repository are a set of functions and classes that I need to have handy for different python programming projects. The repository is very fluid and functionalities keep getting added and remove, signatures keep changing, some libraries graduate to their own repositories, etc. So, basically, these tools are in alpha stage.

tl;dr: Everything is under development.

## Dependencies

Almost everything has been developed for `python3`. I don't plan to work on `python 2.7` compatibility. Everything is aimed for `3.5+`.

## Installation

At the moment, there is a `add_to_python3_user_site` script that adds this folder to the python path. This is so that I can continuously tweak with the scripts without worrying about packaging. In future, I might make a pip package out of this folder.

## Usage

Simply import the libraries into your python code.

## Summary of Tools

### `agents`

A collection of reinforcement learning agents and related function.

- `DQN`
- `DDPG`
- `PPO`
- `A3C`
- `base_agent`
- `value_function`

### `args`
`Args` class: for quickly setting up simple commandline argument parsing functionalities.

### `augmentation`
Dataset augmentation tools.

`mirrorFrame`: Mirrors image matrices horizontally.

`mirrorLidar`: Mirrors lidar readings.

`rotate`: Rotates an image

`scale`: Scales an image in x and y independently.

`transform`: All image transformation functions in one.

### `communication`

### `compare_plot`

`errorPlot`: Takes a set of vectors that are from repeating a test. Plots the average and std or average and min/max. The resulting plot is a shaded area that is common in showing the learning performance from different training attempts.

`runningAverage`

`exponential_moving_average`

### `data_structures`

`SumTree` class

### `environments`

Environments for experimenting with reinforcement learning agents.

`atari`

`fast_nav`

`t_maze`

### `exploration`

Agent exploration policies.

`LinearDecay` class

`OUExploration` class

### `filesystem`

File system tools.

`FileReport` class: fills a file with rows of given arguments.

`tailNumber` function: finds the numbering at the end of a file name and returns the number and raw filename.

`incrementName` function: increments the number at the end of a file name.

`mkd` function: makes a directory.

`rmd` function: remove a directory.

`rm` function: removes a file.

`here` function: git the path to the main executed file.

`ls` function: returns a list of files in a directory.

### `gan`

### `imagination`

### `logs`

Data logging tools.

`DataLog` class: similar to `filesystem.FileReport`, but only for numerical data. Made with the purpose of logging large volume of data into specific length saved batches.

### `memory`

Tools for storing agent experiences.

`Buffer` class: for storing in, loading, and randomly drawing from a ndarray.

`Memory` class: similar to `Buffer` class. It can work with multiple type and dimensionality of data at the same time. Example, saving image frames, lidar readings, actions, rewards at the same time.

`ContinuousMemory` class: similar to `Memory` class. It stores sequences of data and make the storage and retrival of data sequences easier. Practically reducing the memory size of experience replay buffer mechanism to half.

`PrioritizedExperienceReplay` class

### `monitor`

Tools for reporting.

`Figure` class: an easy way of generating `plot` or `imshow` figures that are updatable using a simple syntax.

`Tensorboard` class: an easy way of generating `Tensorboard` summaries.

`Display` class: This is useful if you want to have a string printed and updated over, without printing each new string in a new line of terminal emulator.

`boxPrint` function: prints a given string in a box for additional visibility.

### `networks`

Deep network functions.

`normalizeBatch` function

`flat` function

`fanInStd` function

`fanIn` function

`VAE` class

`MixtureDensityNetworks` class

`latent` function

`conv` function

`deconv` function

`fullyConneted` function

`convolution` function

`Convolutional` class

`Conv` class

`BaseNetwork` class

### `stats`

Statistical tools

`RunningStats` class: calculates a running average and standard deviation.

`RunningAverage` class

`Normality` class

`cosineSImilarity` function

### `tfmisc`

Tools for Tensorflow library.

`parameterCount` function

`selectFromRows` function

`clipGrads` function

`clipedOptimize` function

`applyGradient` function

`getScopeParameters` function

`copyScopeVars` function

### `vae_gan`

### `vae`