# Advanced_ML
Advanced ML homework for Columbia University master's course

Advanced ML Github Directory:

Assignment #1: Predicting Happiness Scores for Countries
The data is a list of countries with a categorical happiness score ranging from very high to very low. There are five possible ratings: very high, high, average, low, very low. The goal of the assignment is to build an ML model to predict those scores based on a variety of tabular features, such as GDP per capita, social support, healthy life expectancy, and generosity among others.

For test accuracy, f1, precision, and recall, the best classifier was SVC with a linear kernel and a regularization parameter C=100. The raw accuracy score of 53.8% is good but not great. It may help to add additional features to the analysis. The precision score of 0.603 means that, when the classifier makes a classification, it is correct 60.3% of the time. Recall is a bit harder to interpret with 5 classes, but in general, recall returns a score that is the proportion of ground truth samples correctly classified.

The multilayer perceptron was the 2nd-best classifier, with a test accuracy score of 46.2%, and an f1 score of 0.450. The MLP opted for an alpha parameter (regularization) of 0.01 and hidden_layer_sizes of 64, 32, and 16, which is quite a large network given the limited number of training samples. Nevertheless it scored quite well; perhaps as a result of the regularization parameter working to keep it from overfitting.

The 3rd-best classifier was Logistic Regression with a regularization parameter C=100, which incidentally was the same as the SVC with linear kernel. The worst classifier of the four tried was Gradient Boosting with a learning rate of 0.316 and max_depth of 3 (which limits the depth of each tree in the ensemble to prevent overfitting). I was suprised Gradient Boosting didn't do better. Perhaps further tuning of the hyperparameters would have further assisted it, since it had the highest training score relative to its test metrics, which can indicate overfitting.

Link: https://github.com/seanmcalevey/Advanced_ML/blob/master/Advanced%20ML%20Homework%20%231%20(3).ipynb

Assignment #2: Identifying Brain Cancer in Radiological Scans (Visual Task)
It's critically important for the medical community to be able to classify whether or not radiological scans of the brain contain tumors. If there are machine learning models that are more accurate on average than their human counterparts, that would be a boon to the medical community since it would save signficant time and human labor, while also improving the accuracy of classification.

As you will see above, the first model I submitted was a simple feed-forward neural net with no dropout that had an accuracy of 72.5%, an F1 score of 0.706414, a precision of 0.766409, and a recall of 0.712963. It did fairly well considering its lack of sophistication but the scores could certainly be improved.

The second model I submitted was a feed-forward neural net with 0.5 dropout before each layer, meaning 50% of the inputs to each layer were randomly dropped, which helps regularize the model during training. The model had an accuracy of 86.3%, an F1 score of 0.86253, a precision of 0.862308, and a recall of 0.863426. These scores were very impressive, not least because it was trained for longer with heavier dropout, meaning it resisted overfitting better than the previous model while also training for a longer period of time.

The third model imported VGG19 as its initial set of layers, and then added a layer of 256 neurons with 50% dropout before outputting a dense layer of two neurons like the other models. This model had an accuracy of 84.3%, an F1 score of 0.841615, a precision of 0.845611, and a recall of 0.840278. Surprisingly this model scored slightly worse overall than the previous model, but nonetheless still scored 6th overall on the leaderboard in terms of accuracy. Clearly the imported VGG19 layers did a good job of bootstrapping the weights into an intially effective structure for understanding visual data such as the radiological scans in the exercise.

The best model from the leaderboard had a structure of three dense layers of 64 units each that are flattened into a nearly 3M dimensional vector and then connected to a softmax layer of that outputs a probability for each class. The surprising thing was that this model didn't use the dropout regularization of my best model but it still outperformed it significantly. But the model is relatively simple: it just uses three 64 unit layers and a softmax output to achieve its result. On the other hand my best model had an initial layer of 256 and then two 64 unit layers before its softmax output; it also had 50% dropout between each layer, which as noted above worked to counteract overfitting.

Also notable was that the optimization approach of the best model from the leaderboard was the same as my best model: both used standard stochastic gradient descent with a 0.001 learning rate without momentum. A different approach might use an optimizer like 'adam' or 'rmsprop', but seeing as the best model used simple stochastic gradient descent, perhaps the best approach is to keep optimization simple and not stray far from SGD without momentum. Nonetheless, my model's similarly simplistic approach (with dropout regularization, however) did not outperform the best model on the leaderboard.

Finally as you'll notice, I fit the best model's architecture to the training data and then scored it on the testing set. It scored a 78.4% accuracy on the test set, which was lower than the leaderboard score. This isn't surprising, however, since for small datasets a lot of the final testing score depends on how the data was split into train and test sets, as some test set combinations may be easier to classify than others.

Link: https://github.com/seanmcalevey/Advanced_ML/blob/master/Advanced_ML_HW.ipynb

Assignment #3: BBC Text Data
The third assignment revolves around the NLP task of processing BBC text data and using it to predict the subject of the discussion. There are five target subjects: tech, business, sport, entertainment, and politics. The average text length in words is 390.3; the minimum length is 90 words; the maximum length is 4,492. Each category is relatively evenly represented, with entertainment (the least common) having nearly 400 instances in the dataset while business (the most common) has just over 500 instances. In all there are roughly 2,500 samples in total.

There were five neural networks fit to the data. The first was a standard feed-forward neural network. Its structure was the following: a 256-unit embedding layer, a 256-unit dense layer, and a 5 unit output layer (one for each subject class). The best validation score was 81.8% after 17 epochs of training.

The second model was a convolutional model comprised of a 256-unit embedding layer, two 1D convolutional layers with max pooling after each, and finally a 256-unit dense layer connected to the 5-unit output layer. The combination of convolutional layers with max pooling vastly reduces the input dimensionality of the text data, helping the model process more effectively in the later layers. This model significantly improved on the score of the first model: the best validation score was 87.8% after 10 epochs.

The third model was a recurrent neural network. The embedding layer was the same, but instead of convolutional layers, this model had a 256-unit GRU layer that processed text in a linear fashion, going one word at a time and running it through the network before moving to the next. The best validation score was 81.6% after 17 epochs.

The fourth model was also a recurrent neural network, but this one had two successive 256-unit GRU layers before the dense and output layers. What this means is that the network still processes each word individually before moving to the next but the processed word data needs to pass through two GRU layers, and both of those layers pass hidden state information on to the next processed word. The best validation score was 72% after 14 epochs. The depth of this network could explain why the validation score was lower than the shallower network. The depth makes it harder for error signals to effectively travel back through the network and update weights in the first layer.

The fifth and final model was a bidirectional recurrent neural network with the exact same structure as the previous (fourth) model, except that each GRU layer runs through the text in both directions, forward and backward. So each layer still has the same number of units for each pass, but because there are two passes the total number of units are effectively doubled for each GRU layer. All of that information is then passed on to the final layers. The best validation score was 79% after 19 epochs, which could partially be explained by the network overfitting with the high number of units in the bidirectional layers.

Link: https://github.com/seanmcalevey/Advanced_ML/blob/master/Advanced_ML_HW_3.ipynb
