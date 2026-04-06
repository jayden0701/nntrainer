# Multi_Input example

- This example demonstrates how to use the `multi_input` layer. 
- The NNTrainer supports a network that takes multiple tensors as inputs. 
- Users can create multiple `input` layers for the network with their own names and build the network accordingly. 
- This code includes an example of training with...

```
                       +-----------+
                       |  output   |
                       +-----------+
                              |                  
    +---------------------------------------------------+  
    |                      flatten                      |
    +---------------------------------------------------+  
                              |                   
    +---------------------------------------------------+  
    |                      concat0                      |
    +---------------------------------------------------+  
        |                     |                  |
    +-----------+       +-----------+       +-----------+  
    |  input 0  |       |  input 1  |       |  input 2  |  
    +-----------+       +-----------+       +-----------+   

```
