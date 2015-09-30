# LHCb-topo-trigger
LHCb RUN-II topological trigger upgrading

## LHCb trigger system RUN-I
![LHCb trigger system](https://raw.githubusercontent.com/tata-antares/LHCb-topo-trigger/master/img/triggers-system.PNG=1000x1000)

## LHCb trigger system RUN-II: upgrading
For RUN-II new scheme is applyed:

* HLT1 track
* HLT1 2-body 
* HLT2 n-body
![new topo scheme](https://raw.githubusercontent.com/tata-antares/LHCb-topo-trigger/master/img/sheme.png)

## LHC
* Sample: one proton-proton bunches collision, called Event (40MHz)
* Event consists of the secondary vertices (SVR) or tracks, where particles are produced
* Features: an SVR, tracks and its products physical characteristics reconstructed from the detectors (momentum, mass, angles, impact parameter)

![LHC event](https://raw.githubusercontent.com/tata-antares/LHCb-topo-trigger/master/img/bdecayinjet.png)

## Data
* Training data are set of SVRs for *HLT2 n-body* and *HLT1 2-body* or trakcs for *HLT1 track* all events
* Monte Carlo 2015 data (signal data) were simulated for various types of interesting events (different decays):
  * all decays are used in *HLT1 2-body* and *HLT1 track* training
  * six types of decays are used for *HLT2 n-body* training and all for testing
* Minimum bias data (real data for a small period of time) are used as background data
* Event is interesting from physical point of view if it contains at least one SVR, where searched decay happens
![Event which passes trigger system](https://raw.githubusercontent.com/tata-antares/LHCb-topo-trigger/master/img/triggers-svg.png)

## ML problem
* Output rate is fixed, thus, false positive rate (FPR) for events is fixed
* Goal is to improve efficiency for each type of signal events 
* We improve true positive rate (TPR) for fixed FPR for events
![ROC curve interpretation](https://raw.githubusercontent.com/tata-antares/LHCb-topo-trigger/master/img/roc_events.png)

## Results
![Comparison HLT2 efficiency (HLT-high level trigger) relation to HLT1 between Run 1 and  new trigger system (without random forest trick). These channels are reconstructible signal decays with pt(B) > 2 GeV and tau(B) > 0.2 ps.](https://github.com/tata-antares/LHCb-topo-trigger/raw/master/img/LHCb_triggers.png)




