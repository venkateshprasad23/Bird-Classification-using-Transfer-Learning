# Bird-Classification-using-Transfer-Learning

The project tests two pretrained models, namely ResNet18 and VGG16 on classifying the CalTech Birds Dataset.

## Description

The project is an attempt to understand the importance of transfer learning. Transfer Learning is incredibly crucial
when we are posed with the problem of a small dataset. Hence we use models pre-trained on the larger ImageNet dataset,
modify only their input and output layers and freeze their hidden layers. This saves time during training and still
gives decent performance. 

### Code Organisation

```
nntools.py   -- Contains abstract classes for implementing various networks, calculating performance,etc.
main.py      -- Code for training and testing.
tutorial.pdf -- A JuPyter Notebook file for explanation purposes.
```



