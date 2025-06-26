This is a simple neural network for handwritten digits image recognition.

It's specificity is that it is written entirely in DPL, my homemade language.

The main file is neural_network.dpl. Since there isn't any way to read files
in dpl yet, I made the download_dataset.py script to generate mnist_train.dpl,
mnist_test.dpl and model_weights.dpl. These files are then imported using #include
in neural_networks.dpl. 

I neet to figure out a better way to create large arrays but it's fine for now.