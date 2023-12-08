import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import MaxPooling1D,Dense, Conv1D, Flatten
from similarity_comparison.similarity import Similarity

class CNNSimilarity:
    """
    CNN Similarity class
    """
    def __init__(self):
        self.model = None
    
    def train(self, labeled_data):
        """
        This method is used for training the CNN. First we divide the labeled_data into features and labels. Then we train the CNN,
        we set the compile function for the model and then we fit it. 

        Args:
            labeled_data: The data used for training the CNN, it consists of tuples and each tuple 
            has this format: (features, class) and features is an array

        """
        # Extract features and labels from labeled_data
        features, labels = self.extract_features_labels(labeled_data)

        # Convert features and labels to numpy arrays
        features = np.array(features)
        labels = np.array(labels)

        # Reshape features to match the input shape of the CNN
        features = features.reshape(features.shape[0], features.shape[1], 1)

        #The number for the classes
        self.num_classes = 151
        
        # Create the CNN model
        # The Convolutional layers(Conv1D) are used for learning features from input data and also sets the input/output shape
        # The Max Pooling layers(MaxPooling1D) are used for reducing the spatial dimensions
        # The Fully Connected layers(Dense) are used for making the final predictions

        self.model = Sequential()
        self.model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(features.shape[1], 1)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(128, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        # First parameter is the loss function used during training
        # The optimizer parameter is used for updating the weights
        # Metrics represent the evaluation of the training
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        self.model.fit(features, labels, epochs=20, batch_size=64, validation_split=0.2)



    def calculate_class_probabilities(self, obj):
        """
        This method is used for getting the class with the biggest probability for an object

        Args:
            obj: The object for which we want to find the biggest probability
        
        Returns: 
            The maximum probability and also the class
        """
         # Convert obj to a numpy array and reshape it
        obj = np.array(obj).reshape(1, len(obj), 1)

        # Calculate the class probabilities using the trained model
        class_probabilities = self.model.predict(obj)[0]

        # Get the index of the class with the highest probability
        max_prob_index = np.argmax(class_probabilities)

        # Get the highest probability and its corresponding class label
        max_prob = class_probabilities[max_prob_index]
        class_label = max_prob_index

        return max_prob, class_label
    
    def calculate_probability_class(self, obj, cls):
        """
        This method is used to calculate the probability of an object belonging to a specific class.

        Args:
            obj: The object for which to calculate the class probabilities.
            cls: The class label for which to retrieve the probability.

        Returns:
            The probability of the object belonging to the specified class.
        """
        # Convert obj to a numpy array and reshape it
        obj = np.array(obj).reshape(1, len(obj), 1)

        # Calculate the class probabilities using the trained model
        class_probabilities = self.model.predict(obj)[0]

        # Get the probability for the specified class label
        class_index = int(cls)
        probability = class_probabilities[class_index]

        return probability

    def calculate_all_probabilities(self, obj, topK):
        """
        This method is used for getting the topK classes with the biggest probability for an object

        Args:
            obj: The object for which we want to find the biggest probability
            topK: An integer that represents the number of classes that we want to find
        
        Returns: 
            The indices of the classes with the biggest probabilities
        """

        # Convert obj to a numpy array and reshape it
        obj = np.array(obj).reshape(1, len(obj), 1)

        # Calculate the class probabilities using the trained model
        class_probabilities = self.model.predict(obj)[0]

        top_indices = np.argsort(class_probabilities)[-topK:]

        return top_indices
    
    def extract_features_labels(self, labeled_data):
        """
        This method is used for creating the features and labels using the labeled_data

        Args:
            labeled_data: The data used for training the CNN, it consists of tuples and each tuple 
            has this format: (features, class) and features is an array
        
        Returns: 
            This method should return a tuple of two lists: (features, labels)
        """

        features = []
        labels = []
        for data in labeled_data:
            features.append(data[0])
            labels.append(data[1])

        return features, labels