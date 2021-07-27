# PatchPerPixMatch
Script collection for matching neuronal morphologies in 3d multicolor light microscopy images, described in [[1]](#1)

PatchPerPixMatch is based on PatchPerPix instance segmentations [[2]](#2). 
Moreover, it depends on NBLAST [[3]](#3) to score how well PatchPerPix fragments align with a given target neuron.

### Pipeline
1. remove small components
2. get all fragment colors
3. skeletonize
4. crop lm to em bounding box
5. nblast lm to em
6. find best matches
7. visualize matches

## References
<a id="1">[1]</a>
Lisa Mais, Peter Hirsch, Claire Managan, Kaiyu Wang, Konrad Rokicki, Robert R. Svirskas, Barry J. Dickson, Wyatt Korff, Gerald M. Rubin, Gudrun Ihrke, Geoffrey W. Meissner, Dagmar Kainmueller (2021) PatchPerPixMatch for Automated 3d Search of Neuronal Morphologies in Light Microscopy. bioRxiv, https://doi.org/10.1101/2021.07.23.453511

<a id="2">[2]</a> 
Mais L., Hirsch P., Kainmueller D. (2020) 
PatchPerPix for Instance Segmentation. 
In: Vedaldi A., Bischof H., Brox T., Frahm JM. (eds) Computer Vision – ECCV 2020. ECCV 2020. 
Lecture Notes in Computer Science, vol 12370. 
Springer, Cham. https://doi.org/10.1007/978-3-030-58595-2_18

<a id="3">[3]</a> 
Costa, M., Manton, J.D., Ostrovsky, A.D., Prohaska, S., Jefferis, G.S. (2016)
Nblast: rapid, sensitivecomparison of neuronal structure and construction of neuron family databases. 
Neuron 91(2), 293–311
