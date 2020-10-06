# Brain Functional Networks - An Interpretable Graph Classification Solution

We started from a real use case scenario: identifying the regions of the brain which are most affectedby alcohol consumption. The brain can be represented as a functional network (a weighted complete graph): nodes are regions, while edges represents regions connections. Edge weights illustrate nodes correlations (how strong two brain regions are correlated), their range values being between 0 and 1. The outcome of the project was an analysis of these networks in two different states: alcohol and non-alcohol. The analysis was obtained by building an interpretable graph classification pipeline.

In Palcu et al. [1] we used Deep Graph Convolutional Networks (DGCNN) Zhang et al. [2], that takes as input graphs of arbitrary structure and builds a graph classification model by applying end-to-end gradient based training. The graph classification model is used for distinguishing between different functional network states: Control (initial state) and Alcohol. However, one major issue is the lack of interpretability of such a model.  To overcome this issue, we integrated an adapted version of the Grad-CAM Selvaraju et al. [3] method that enables the visualization of the feature-importance map, thus highlighting the discriminative features.


**References**

[1] **Liana-Daniela Palcu**, Marius Supuran,Camelia Lemnaru, Rodica Potolea, Mihaela Dinsoreanu, Raul Cristian Muresan. *Discovering Discriminative Nodes for Classification with Deep Graph Convolutional Methods*. International Workshop on New Frontiers in Mining ComplexPatterns (nFCMP 2019). https://link.springer.com/chapter/10.1007/978-3-030-48861-1_5

[2] Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, and DhruvBatra. *Grad-cam: Why did you say that? visual explanations from deep networks via gradient-based localization.* CoRR, abs/1610.02391, 2016. http://arxiv.org/abs/1610.02391.

[3] Muhan Zhang, Zhicheng Cui, Marion Neumann, and Yixin Chen. *An end-to-end deep learning architecture forgraph classification.* In AAAI, pages 4438–4445, 2018. https://muhanzhang.github.io/papers/AAAI_2018_DGCNN.pdf

