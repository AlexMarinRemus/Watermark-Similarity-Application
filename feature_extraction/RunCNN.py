from feature_extraction.CNN import CNNSimilarity
import ast

class RunCNNSImilarity:

    def __init__(self):
        None
    
    def RunCNN(self, obj):
        # Create a CNN
        cnn = CNNSimilarity()

        # Read the content from the file
        with open("features.txt", "r") as file:
            content = file.read()

        # Convert the text to a list of tuples
        labeled_data = ast.literal_eval(content)
        images = ast.literal_eval(content)
        # Train the CNN
        cnn.train(labeled_data) 

        # Calculate the similarity and the probability for and object
        (simProb, cls) = cnn.calculate_class_probabilities(obj)

        # For each image save the path and the probability of being part of that class
        output= [] 
        for image in images:
            output.append((cnn.calculate_probability_class(image[0],cls),image[1]))

        # Sort the output list based on the first term (probability) in descending order
        sorted_output = sorted(output, key=lambda x: x[0], reverse=True)    

        result= []
        # Print the sorted output
        for item in sorted_output:
            probability, path = item
            result.append(path)

        return result