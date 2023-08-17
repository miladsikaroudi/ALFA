# Leveraging All Levels of Features Abstraction for Improving the Generalization to Unseen Hospitals

## Table of Content

1. [Introduction](#intro)
2. [Guideline](#guideline)
    - [To prepare](#prepare)
    - [To run experiments](#experiments)
    - [To visualize objective functions](#visualize)
    - [To plot feature representations](#plot)
    
## <a name="intro"></a> Introduction


This repository contains the implementation of our "exploting the All Level of Features Abstraction" (ALFA) method for MICCAI 2023. Our approach goes beyond traditional methods by leveraging not only domain-specific features but also incorporating SSL features with the goal of improving DG. Through the integration of SSL, our ALFA method can effectively learn and utilize additional features that are relevant, resulting in improved performance and accuracy. We are excited to present our findings at MICCAI 2023 and share how the ALFA method can advance the field of medical image analysis.


![framework](gallery/insight.png)

Details of the model and experimental results can be found in the submission:
```bibtex
in progress
```
**Please CITE** our paper if you find it useful in your research.

### <a name="visualize"></a> To visualize objective functions we have used wandb:

```sh
wandb.log({name_for_the_entity:entity}) 
```

### <a name="Losses plots"></a> Here is the losses plot for the ALFA:

<img src="gallery/L_ssl_new.png" width="50%" height="50%">
<img src="gallery/L_i_new.png" width="50%" height="50%">
<img src="gallery/Ls_new.png" width="50%" height="50%">
<img src="gallery/L_disentangle i-ssl_new.png" width="50%" height="50%">
<img src="gallery/L_disentangle i-s_new.png" width="50%" height="50%">
<img src="gallery/L_disentangle s-ssl_new.png" width="50%" height="50%">
<img src="gallery/L_c_new.png" width="50%" height="50%">

**Note:** Results are stored in `DomainBed/results/train_output/out.txt`

## License

This source code is released under the Apache-2.0 license, included [here](LICENSE).
