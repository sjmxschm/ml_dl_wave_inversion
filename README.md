# [Wave Inversion with Machine and Deep Learning](https://ssrn.com/abstract=4124083)

**Author:** [Maximilian Schmitz](https://sjmxschm.github.io/) @ Georgia Institute of Technology

**Email:** [mschmitz7@gatech.edu](mailto:mschmitz7@gatech.edu)

![alt text](resources/DL_inversion_image.png "Title")

## Introduction

This is the official repository of the publication
> Schmitz, Maximilian and Kim, Jin-Yeon and Jacobs, Laurence J., Machine and Deep Learning for Coating Thickness 
> Prediction Using Lamb Waves. Available at ScienceDirect: [here](https://www.sciencedirect.com/science/article/abs/pii/S0165212523000239?via%3Dihub) or 
> [http://dx.doi.org/10.2139/ssrn.4124083](https://doi.org/10.1016/j.wavemoti.2023.103137)

With this repository, you can
- convert phase speed - frequency representations into the frequency - wave number domain
- automatically simulate CAE models for finite element analysis in [Abaqus](https://www.3ds.com/products-services/simulia/products/abaqus/) 
- invert to the thickness of the coating using machine learning techniques
- invert to the uniformness of the coating using deep learning


## Repo Walk through

The project is subdivided into 3 separate main parts. Each part is contained in a
separate folder with respective naming which will be walked through in the following. There is a visualization
of the repo structure below too.

- `analysis_sklearn` contains the machine learning inversion procedure based on scikit-learn
- `analysis_disp_curves` incorporates the conversion of dispersion curves extracted from a 
dispersion curve calculator (like [DLR dispersion calculator](https://www.dlr.de/zlp/en/desktopdefault.aspx/tabid-14332/24874_read-61142/))
into the right format and domain, e.g. from a phase speed - frequency representation into a
frequency - wave number domain. 
- `create_model` includes all scripts and files to (1) create the CAE model with a Python script in [Abaqus](https://www.3ds.com/products-services/simulia/products/abaqus/)
(industry standard for finite element analysis (FEM)) and (2) to run the entire pipeline for creating several
simulations on a Linux system (like the Georgia Tech PACE cluster) in parallel. More detailed
information about how to use those scripts properly and how to install the respective
environment can be found in the `README.md` in the respective folder.
- `dl_code` contains the code and data for training and evaluating a deep learning based
inversion model. The folder contains several notebooks which are explained in the `README.md`.
- `extract_disp_history` is merged into the `create_model` folder and is not needed anymore
- `helper_scripts` include smaller scripts for minor tasks used during develpment
- `unit_tests` include Python unit tests which were run during development and can still be
used for testing any adjustments made. Most unit tests are called form the deep learning code.

## Repo Structure

```
ml_dl_wave_inversion
│   README.md  
│
└───analysis_sklearn
│   │   analysis_sklearn.ipynb
│   │   dispersion_feature_loader.py
│   └───data
|       |   ...
│   └───figures
│
└───analytical_disp_curves
│   |   analytical_curves.py
│   └───Cr_dispersion_curves
|      |   ...
│   └───Zy4Cr_dispersion_curves
|      |   ...
│   └───Zy4_dispersion_curves_DC
|      |   ...
│
└───create_model
│   |   README.md
|   |   __init__.py
|   |   at_cluster_conda.pbs
|   |   at_cluster_conda_postpro.pbs
|   |   auto_feat_ext.pbs
|   |   auto_feature_extraction.py
|   |   auto_image_extraction.py
|   |   auto_img_ext.pbs
|   |   create_model_script_v15.py
|   |   create_model_script_v15_no_gap.py
|   |   delete_unnecessary_files_from_cluster.py
|   |   extract_disp_history_max_v5.py
|   |   postprocess_2dfft_max_v15.py
|   |   qdel_automation.py
|   |   run_automated_simulations_cluster.py
|   |   run_automated_simulations_cluster_postp.py
|   |   run_parallel_on_cluster.pbs
|   |   run_parallel_simulations.py
|   |   run_simulation.py
|   |   utils.py
|   |   visualize_dataset.py
|   |   wave_env_cluster.yml
│   └───figures_param_space
|
└───dl_code
│   |   README.md
|   |   __init__.py
|   |   confusion_matrix.py
|   |   data_transforms.py
|   |   dl_utils.py
|   |   image_loader.py
|   |   maxWaveNet_local.ipynb
|   |   my_resnet.py
|   |   my_resnet_old_december.py
|   |   optimizer.py
|   |   runner.py
|   |   simple_net.py
|   |   simple_net_big.py
|   |   simple_net_final.py
|   |   stats_helper.py
|   |   test_cuda_on_cluster.ipynb
|   |   testing.py
|   |   train_network.pbs
|   |   train_network_cluster.py
|   |   validate_network.ipynb
|   |   visualize_training_history.py
|   |   wave_cnn_env_linux.yml
|   |   wave_cnn_env_linux_2.yml
│   └───data
|      └───test
|           |   ...
|      └───train
|           |   ...
│   └───figures
│   └───model_checkpoints
│   └───trained_models
│
└───helper_scripts
│   |   automate_calling_from_cmd.py
|   |   build_slack_notifications.py
│   |   calculate_group_velocity.py
|   |   calculate_material_constants.py
│   |   delete_only_blacklist_files.py
|   |   ft_triangular_func.png
│   |   triangle_fourier_transform.py
|   
└───resources
| 
└───unit_tests
│   |   __init__.py
|   |   model_test_utils.py
│   |   test_data_transforms.py
│   |   test_dl_utils.py
|   |   test_image_loader.py
│   |   test_my_resnet.py
│   |   test_simple_net.py
|   |   test_simple_net_final.py
│   |   test_stats_helper.py
|   |   utils.py
```

## Train/Test Data & Pretrained Weights

As by definition, the machine and deep learning part is based on simulated data. I have not uploaded the 
data used for this research (train and test images, pretrained model parameters) to providers like SourceForge, ... yet. 
If you are interested, feel free to send me an email and I am happy to provide you with the respective
data: [mschmitz7@gatech.edu](mailto:mschmitz7@gatech.edu)

## Citation

If you find this useful, please cite our paper.
 ```
@article{ml_wave_inversion,
  title = {Machine and deep learning for coating thickness prediction using Lamb waves},
  author = {Maximilian Schmitz and Jin-Yeon Kim and Laurence J. Jacobs},
  journal = {Wave Motion},
  volume = {120},
  pages = {103137},
  year = {2023},
  issn = {0165-2125},
  doi = {https://doi.org/10.1016/j.wavemoti.2023.103137},
  url = {https://www.sciencedirect.com/science/article/pii/S0165212523000239},
  keywords = {Machine learning, Deep learning, Inversion, Wave propagation, Lamb wave, Fourier transform},
  year={2022}
}
```
