# IF-learn 
Learning algorithms for machine learning based estimation and inference on structural target functions, such as conditional average treatment effects, using influence functions.

This repository is structured as follows: 

(1) iflearn - contains sklearn-style implementations of model-agnostic learning algorithms (sometimes also referred to as Meta-learners), currently limited to estimation of conditional average treatment effects.

(2) paper_utils.if_paper - contains utils to replicate the simulation studies in Curth, Alaa & van der Schaar (2020). With a working rpy2 installation, these can be run using the notebook 'Simulation studies.ipynb'. 


This code was written by Alicia Curth during a research (dissertation) internship at the vanderschaar-lab at the University of Cambridge, and was used for a dissertation submitted in partial fulfilment of the degree of Master of Science in Statistical Science at the University of Oxford in September 2020. 


### Citing
If you use this Code, please cite
```
@article{curth2020estimating,
  title={Estimating Structural Target Functions using Machine Learning and Influence Functions},
  author={Curth, Alicia and Alaa, Ahmed M and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2008.06461},
  year={2020}
}
```
