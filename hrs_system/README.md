# CombiGCN
This CombiGCN model, titled 'CombiGCN: An Effective GCN Model for Recommender System', embodies our novel research and is detailed in our paper (DOI: 10.1007/978-981-97-0669-3_11). While our model draws inspiration from the LightGCN framework, it introduces significant enhancements and optimizations tailored for the recommender system context, implemented using TensorFlow. Our codebase builds upon the foundational concepts from the LightGCN model, accessible at https://github.com/kuandeng/LightGCN.


## Introduction
By identifying the similarity weight of users through their interaction history, a key concept of CF, we endeavor to build a user-user weighted connection graph based on their similarity weight of them. We propose a recommendation framework CombiGCN that combine user-user weighted connection graph and user-item interaction graph.

## Environment Requirement
The code has been tested running under Python 3.10.9 The required packages are as follows:
* tensorflow == 2.11.0
* numpy == 1.24.3
* scipy == 1.9.0
* sklearn == 1.2.0

## Examples to run a 3-layer CombiGCN

### vcr dataset

* Command
```
python CombiGCN.py --dataset vcr5p_late_fusion --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 10000
```



NOTE : the duration of training and testing depends on the running environment.
## Dataset
We provide three processed datasets: Ciao, Epinions and Foursquare.
* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.
  
* `user_list.txt`
  * User file.
  * Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
  
* `item_list.txt`
  * Item file.
  * Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.

## Reference
This CombiGCN model is based on our research presented in the following paper: "CombiGCN: An Effective GCN Model for Recommender System" (10.1007/978-981-97-0669-3_11). If you use this model or our findings in your research, please cite:

@InProceedings{10.1007/978-981-97-0669-3_11,
author="Nguyen, Loc Tan
and Tran, Tin T.",
editor="H{\`a}, Minh Ho{\`a}ng
and Zhu, Xingquan
and Thai, My T.",
title="CombiGCN: An Effective GCN Model for Recommender System",
booktitle="Computational Data and Social Networks",
year="2024",
publisher="Springer Nature Singapore",
address="Singapore",
pages="111--119",
abstract="Graph Neural Networks (GNNs) have opened up a potential line of research for collaborative filtering (CF). The key power of GNNs is based on injecting collaborative signal into user and item embeddings which will contain information about user-item interactions after that. However, there are still some unsatisfactory points for a CF model that GNNs could have done better. The way in which the collaborative signal are extracted through an implicit feedback matrix that is essentially built on top of the message-passing architecture of GNNs, and it only helps to update the embedding based on the value of the items (or users) embeddings neighboring. By identifying the similarity weight of users through their interaction history, a key concept of CF, we endeavor to build a user-user weighted connection graph based on their similarity weight.",
isbn="978-981-97-0669-3"
}

=======
