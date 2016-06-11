- To create the train-test split, use ipython to read 'photo_id_to_business_id.json' and do the split
- Created 2 new models Places205CNN_yelp and places_GoogleNet_yelp via the prototxt.
- Set all the weights (bias_lr and weight_decay) = 0 except for fc8. This is just saying that I want to freeze the weights of the model for all layers except for fc8.

Notebook on multi label classification
http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/pascal-multilabel-with-datalayer.ipynb

## Note
Always use the attributes_list file to read in attributes. The order of labels in that file is to be considered the standard order.   

#### Create Python Data Layer
Look at caffe/examples/pycaffe/layers/pascal-multilabel-with-datalayer.py file to understand how to create a layer to pick up the dataset and the multilabels
Module name: `yelp_multilabel_datalayers`

### Papers/References
1. http://arxiv.org/pdf/1502.05988.pdf
2. http://arxiv.org/pdf/1406.5726.pdf
3. http://arxiv.org/pdf/1403.6382.pdf
4. http://arxiv.org/pdf/1406.5726.pdf
5. https://www.cs.toronto.edu/~hinton/science.pdf

# Issues
- Degeneration of results: Predicts everything to be zero.
  - Possibles fixes: Batch Normalization (Doersch paper)
  - Square error[5]

# Things to try:
- AlexNet Baseline
- Generate Mean files of training and testing data
- GoogleNet
- Siamese on GoogleNet and Places-GoogleNet using Contrastive Loss

I have created separate train_val and solver prototxts for each model. To convert a deploy to train_val, replace the prob layer with the loss layer. Be sure to modify the name and top/bottom names of the layers you want to finetune.

To train, run from src folder (example for GoogleNet):

    ../../caffe/build/tools/caffe train -solver models/solver_googlenet.prototxt -weights models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel -gpu all


Look at results.py to see how the final metric is calculated.
