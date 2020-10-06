# User's manual

## Requirements
<code>pip install -r requirements.txt</code>

## Getting started
* Create train/test splits\
<code>python preprocessing_data.py</code>
* Train the model\
<code>./run_DGCNN.sh  folder_to_read  nr_folds</code>
* Create importance frequency matrix\
<code>python  importance_frequency_algorithm.py</code>
* Create interpretability heatmaps\
<code>python  plot_interpretability_heatmaps.py</code>



