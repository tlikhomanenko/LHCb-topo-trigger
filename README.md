# LHCb-topo-trigger
LHCb RUN-II topological trigger upgrading

## LHCb trigger system RUN-I

<img src='https://raw.githubusercontent.com/tata-antares/LHCb-topo-trigger/master/img/triggers-system.PNG' width=250 alt='LHCb trigger system' />

## LHCb trigger system RUN-II: upgrading
For RUN-II new scheme is applied:

* HLT1 track
* HLT1 2-body 
* HLT2 n-body

<img src='https://raw.githubusercontent.com/tata-antares/LHCb-topo-trigger/master/img/sheme.png' width=350 alt='new topo scheme' />


## LHC
* Sample: one proton-proton bunches collision, called Event (40MHz)
* Event consists of the secondary vertices (SVR) or tracks, where particles are produced
* Features: an SVR, tracks and its products physical characteristics reconstructed from the detectors (momentum, mass, angles, impact parameter)

<img src='https://raw.githubusercontent.com/tata-antares/LHCb-topo-trigger/master/img/bdecayinjet.png' width=350 alt='LHC event' />

## Data
* Training data are set of SVRs for *HLT2 n-body* and *HLT1 2-body* or trakcs for *HLT1 track* all events
* Monte Carlo 2015 data (signal data) were simulated for various types of interesting events (different decays):
  * all decays are used in *HLT1 2-body* and *HLT1 track* training
  * six types of decays are used for *HLT2 n-body* training and all for testing
* Minimum bias data (real data for a small period of time) are used as background data
* Event is interesting from physical point of view if it contains at least one SVR, where searched decay happens

<img src='https://raw.githubusercontent.com/tata-antares/LHCb-topo-trigger/master/img/triggers-svg.png' width=250 alt='Event which passes trigger system' />

## ML problem
* Output rate is fixed, thus, false positive rate (FPR) for events is fixed
* Goal is to improve efficiency for each type of signal events 
* We improve true positive rate (TPR) for fixed FPR for events

<img src='https://raw.githubusercontent.com/tata-antares/LHCb-topo-trigger/master/img/roc_events.png' width=550 alt='ROC curve interpretation' />

## Production trigger system
There are two possibilities to speed up prediction operation for production stage:
* Bonsai boosted decision tree format (BBDT)
  * Features hashing using bins before training 
  * Converting decision trees to n-dimensional table (lookup table)
  * Table size is limited in RAM, thus count of bins for each features should be small
  * Discretization reduces quality
* Post-pruning (MatrixNet includes several thousand trees)
  * Train MatrixNet with several thousands trees
  * Reduce this amount of trees to a hundred
  * Greedily choose trees in a sequence from the initial ensemble to minimize a modified loss function (exploss for background and logloss for signal)
  * Change values in leaves (tree structure is preserved)

## Results
![Comparison HLT2 efficiency (HLT-high level trigger) relation to HLT1 between Run 1 and  new trigger system (without random forest trick). These channels are reconstructible signal decays with pt(B) > 2 GeV and tau(B) > 0.2 ps.](https://github.com/tata-antares/LHCb-topo-trigger/raw/master/img/LHCb_triggers.png)

## Reproducibility
* download root files to folder *datasets*
* run [preprocessing](https://github.com/tata-antares/LHCb-topo-trigger/blob/master/0_Preprocessing.ipynb) to create .csv files with tracks and SVRs 
* [HLT1 track](https://github.com/tata-antares/LHCb-topo-trigger/blob/master/HLT1-track.ipynb) creates models and plots for *HLT1 track* trigger
* [HLT1 2-body](https://github.com/tata-antares/LHCb-topo-trigger/blob/master/HLT1.ipynb) creates models and plots for *HLT1 2-body* trigger
* [HLT2 n-body] (https://github.com/tata-antares/LHCb-topo-trigger/blob/master/HLT2.ipynb) creates models and plots for *HLT2 n-body* trigger
* [BBDT and post-pruning](https://github.com/tata-antares/LHCb-topo-trigger/blob/master/HLT2-TreesPruning.ipynb) creates production models with BBDT format and post pruning for *HLT2 n-body*

## Requirements
* [rep](http://github.com/yandex/rep)
* [rep_ef](https://github.com/anaderi/REP_EF)

