# User's manual

## Requirements
<code>pip install -r requirements.txt</code>

## Config file
*config_DGCNN.json* has the following parameters:\
* **folder_name** - the name of the results folder
* **nr_nodes** - the number of graph's nodes
* **nr_bins_cam** - the size of the result of Grad-CAM (the results is an array).

## Input data
An example of input data can be found in *pre_processed_data* folder. All the graphs have the Networkx xml file format. If you want to change the format, then you have to create a new Python file for preprocessing all the data.

## Output data
An example of how you should save the results can be found in *data* folder.

## Getting started
* Create train/test splits\
<code>python preprocessing_data.py</code>
* Train the model\
<code>./run_DGCNN.sh  folder_to_read  nr_folds</code>
* Create importance frequency matrix\
<code>python  importance_frequency_algorithm.py</code>
* Create interpretability heatmaps\
<code>python  plot_interpretability_heatmaps.py</code>



