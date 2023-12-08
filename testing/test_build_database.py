import pytest
from unittest.mock import patch, mock_open, call, MagicMock
from unittest import TestCase
import pandas as pd
import os
import pickle
import cv2
import argparse
import sys

import build_database as Build_DB
from feature_extraction.feature_extraction import FeatureExtraction
from harmonization.harmonization import Harmonization

class TestBuildDatabase:

    def test_open_loadDB(self):
        """
        Tests that the loadDB function opens the file with the
        provided file path, with mode being "rb"
        """

        m = mock_open(read_data="pickle_file")
        # Even though only opening the pkl file is being test
        # the dataframe still needs to be mocked in order for the
        # method to run
        df = pd.DataFrame({"image_path": ["image.png"], "features": [[0,1,2]]})

        with patch("builtins.open", m) as mock_open_file, \
            patch("build_database.pd.read_pickle", return_value=df) as mock_read_pickle:

            data = Build_DB.loadDB("database.pkl")
            # Asserts that when the open() method access the file path, the
            # expected data is read
            assert open("database.pkl", "rb").read() == "pickle_file"
            # Asserts that the mock is called with expected arguments
            mock_open_file.assert_called_with("database.pkl", "rb")

    def test_open_file_not_found_loadDB(self):
        """
        Tests that the loadDB function catches file not found exception and instead
        logs a message.
        """

        with TestCase.assertLogs(self) as captured:
            data = Build_DB.loadDB("database.pkl")

            assert data == None
            assert captured.records[0].getMessage() == "Loading the database from database.pkl"
            assert captured.records[1].getMessage() == "File database.pkl could not be found"


    def test_read_pickle_loadDB(self):
        """
        Tests that the loadDB function can access the dataframe from the
        pkl file and returns the dataframe information as expected.
        """

        m = mock_open(read_data="pickle_file")
        df = pd.DataFrame({"image_path": ["image.png"], "features": [[0,1,2]]})

        with patch("builtins.open", m) as mock_open_file, \
            patch("build_database.pd.read_pickle", return_value=df) as mock_read_pickle:

            data = Build_DB.loadDB("database.pkl")

            # Asserts that read_pickle method get"s called with the expected
            # file path
            assert mock_read_pickle.call_args.args[0].read() == "pickle_file"
            # Asserts that the data is retrieved from the dataframe as expected
            assert data == [("image.png", [0,1,2])]

    def test_logging_loadDB(self):
        """
        Tests that the loadDB function interacts with the log as expected.
        """

        m = mock_open(read_data="pickle_file")
        df = pd.DataFrame({"image_path": ["image.png"], "features": [[0,1,2]]})

        with patch("builtins.open", m) as mock_open_file, \
            patch("build_database.pd.read_pickle", return_value=df) as mock_read_pickle, \
            TestCase.assertLogs(self) as captured:

            Build_DB.loadDB("database.pkl")
            assert len(captured.records) == 2
            assert captured.records[0].getMessage() == "Loading the database from database.pkl"
            assert captured.records[1].getMessage() == "Database successfully loaded"

    def test_access_images_makeDB(self):
        """
        Test that images are accessed by os.listdir, that they are read, and
        that their features are extracted, when making the database.
        """

        # The three different image names, and the three different image arrays (each of size 1x3)
        image_names = ["image1.png", "image2.png", "image3.png"]
        images = [[0,0,255], [0,255,0], [255,0,0]]
        # The dataframe is also mocked so that the pkl file is not generated or interacted with,
        # since that is not what is being tested here.
        dataframe = pd.DataFrame({"test": []})

        # Note: return value is a static return value, and side effects returns different things
        # each time the mock is called. So since there are three images, imread uses side_effect
        with patch("build_database.os.listdir", return_value=image_names) as mock_listdir, \
                patch("build_database.cv2.imread", side_effect=images) as mock_read_image, \
                patch.object(dataframe, "memory_usage") as mock_memory_usage, \
                patch.object(dataframe, "to_pickle") as mock_to_pickle, \
                patch("build_database.os.path.exists") as mock_path, \
                patch.object(FeatureExtraction, "extract_features") as mock_feature_extraction, \
                patch.object(Harmonization, "harmonize") as mock_harmonize, \
                TestCase.assertLogs(self) as captured:

            mock_harmonize.side_effect = [[0,0,1], [0,1,0], [1,0,0]]


            Build_DB.makeDB("test_db.pkl", "images", is_traced=True)

            # Asserts that the listdir function is called with the expected input
            mock_listdir.assert_called_with("images")
            # Asserts that imread is called with three different calls, each to the different image path
            imread_calls = [call("images/image1.png", 0), call("images/image2.png", 0), call("images/image3.png", 0)]
            mock_read_image.assert_has_calls(imread_calls)
            print(mock_read_image.call_args_list)

            # Assert that the processing image message is logged for each image
            assert captured.records[0].getMessage() == "Processing image 1 of 3"
            assert captured.records[1].getMessage() == "Processing image 2 of 3"
            assert captured.records[2].getMessage() == "Processing image 3 of 3"

            # Assert that there are three harmonization calls, each assuming traced
            harmonization_calls = [call(True), call(True), call(True)]
            mock_harmonize.assert_has_calls(harmonization_calls)

            # Assert that there are three feature extraction calls, each to the respective (harmonized) image
            feature_extraction_calls = [call([0,0,1]), call([0,1,0]), call([1,0,0])]
            mock_feature_extraction.assert_has_calls(feature_extraction_calls)

    def test_makedb_image_not_read(self):
        """
        Test that if the image is None then the logger records the right
        message.
        """
        
        # Initialize the image that will be used.
        image_name = ["invalid_img.png"]
        image = None
        dataframe = pd.DataFrame({"test": []})

        # Mock methods that will be called.
        with patch("build_database.os.listdir", return_value=image_name) as mock_listdir, \
                patch("build_database.cv2.imread", return_value=image) as mock_read_image, \
                patch.object(dataframe, "memory_usage") as mock_memory_usage, \
                patch.object(dataframe, "to_pickle") as mock_to_pickle, \
                patch("build_database.os.path.exists") as mock_path, \
                patch.object(FeatureExtraction, "extract_features") as mock_feature_extraction, \
                patch.object(Harmonization, "harmonize") as mock_harmonize, \
                patch("logging.disable") as mock_log, \
                TestCase.assertLogs(self) as captured:
            
            # Execute makeDB with the set parameters.
            Build_DB.makeDB("test_db.pkl", "image", is_traced=True)
            mock_listdir.assert_called_with("image")

            # Assert that the logger registers the right messages and warning. 
            assert captured.records[0].getMessage() == "Processing image 1 of 1"
            assert captured.records[1].getMessage() == "The image invalid_img.png could not be read"

    def test_creation_of_pkl_makeDB(self):
        """
        Test that images are stored in the pkl in a way that is expected.
        """

        try:
            # The three different image names, and the three different image arrays (each of size 1x3)
            image_names = ["image1.png", "image2.png", "image3.png"]
            images = [[[0,0,255], [0,0,255], [0,0,255]],
                    [[0, 255, 0], [0, 255, 0], [0, 255, 0]],
                    [[255, 0, 0], [255, 0, 0], [255, 0, 0]]
                    ]

            # Note: return value is a static return value, and side effects returns different things
            # each time the mock is called. So since there are three images, imread uses side_effect
            with patch("build_database.os.listdir", return_value=image_names) as mock_listdir, \
                    patch("build_database.cv2.imread", side_effect=images) as mock_read_image, \
                    patch.object(FeatureExtraction, "extract_features") as mock_feature_extraction, \
                    patch.object(Harmonization, "harmonize") as mock_harmonize, \
                    TestCase.assertLogs(self) as captured:

                mock_harmonize.side_effect = [[0,0,1], [0,1,0], [1,0,0]]
                mock_feature_extraction.side_effect = [[0,1,2],[3,4,5],[6,7,8]]
                Build_DB.makeDB("test_db.pkl", "images", is_traced=True)

                print(captured.records)
                assert captured.records[3].getMessage() == "Database successfully created"
                assert "Saving the database to test_db.pkl" in captured.records[4].getMessage()


                with open("test_db.pkl", "rb") as f:
                    loaded_pkl = pickle.load(f)

                    assert loaded_pkl["image_path"][0] == "images/image1.png"
                    assert loaded_pkl["features"][0] == [0,1,2]
                    assert loaded_pkl["image_path"][1] == "images/image2.png"
                    assert loaded_pkl["features"][1] == [3,4,5]
                    assert loaded_pkl["image_path"][2] == "images/image3.png"
                    assert loaded_pkl["features"][2] == [6,7,8]

        finally:
            # The to_pickle method creates a pickle file, which is deleted here.
            # Note: this is not mocked because the contents of the pickle file are tested
            if os.path.exists("test_db.pkl"):
                os.remove("test_db.pkl")

    def test_pkl_appending_makeDB(self):
        """
        Test that images are appended in the pkl in a way that is expected.
        """

        try:
            # Create a pre-existing pkl file that will be appended to
            initial_db = {"image_path": ["images/image1.png", "images/image2.png", "images/image3.png"], \
                "features": [[0,1,2],[3,4,5],[6,7,8]]}
            initial_df = pd.DataFrame(initial_db)
            initial_df.to_pickle("test_db.pkl")

            # The three different image names, and the three different image arrays (each of size 1x3)
            image_names = ["image4.png", "image5.png", "image6.png"]
            images = [[0,0,255], [0,255,0], [255,0,0]]

            # Note: return value is a static return value, and side effects returns different things
            # each time the mock is called. So since there are three images, imread uses side_effect
            with patch("build_database.os.listdir", return_value=image_names) as mock_listdir, \
                    patch("build_database.cv2.imread", side_effect=images) as mock_read_image, \
                    patch.object(FeatureExtraction, "extract_features") as mock_feature_extraction, \
                    patch.object(Harmonization, "harmonize") as mock_harmonize, \
                    TestCase.assertLogs(self) as captured:

                mock_harmonize.side_effect = [[0,0,1], [0,1,0], [1,0,0]]

                mock_feature_extraction.side_effect = [[9,10,11],[12,13,14],[15,16,17]]
                Build_DB.makeDB("test_db.pkl", "images", is_traced=True)

                assert captured.records[3].getMessage() == "Data successfully added to database"
                assert "Saving the database to test_db.pkl" in captured.records[4].getMessage()

                with open("test_db.pkl", "rb") as f:
                    loaded_pkl = pickle.load(f)

                    # Asserts that all entries, both old and new, appear
                    assert loaded_pkl["image_path"][0] == "images/image1.png"
                    assert loaded_pkl["features"][0] == [0,1,2]
                    assert loaded_pkl["image_path"][1] == "images/image2.png"
                    assert loaded_pkl["features"][1] == [3,4,5]
                    assert loaded_pkl["image_path"][2] == "images/image3.png"
                    assert loaded_pkl["features"][2] == [6,7,8]
                    assert loaded_pkl["image_path"][3] == "images/image4.png"
                    assert loaded_pkl["features"][3] == [9,10,11]
                    assert loaded_pkl["image_path"][4] == "images/image5.png"
                    assert loaded_pkl["features"][4] == [12,13,14]
                    assert loaded_pkl["image_path"][5] == "images/image6.png"
                    assert loaded_pkl["features"][5] == [15,16,17]


        finally:
            # The to_pickle method creates a pickle file, which is deleted here.
            # Note: this is not mocked because the contents of the pickle file are tested
            if os.path.exists("test_db.pkl"):
                os.remove("test_db.pkl")

    def test_db_file_not_found_makeDB(self):
        """
        Tests that the logger has the expected exception method when a database
        file cannot be found.
        """

        # The three different image names, and the three different image arrays (each of size 1x3)
        image_names = ["image1.png", "image2.png", "image3.png"]
        images = [[0,0,255], [0,255,0], [255,0,0]]

        # Note: return value is a static return value, and side effects returns different things
        # each time the mock is called. So since there are three images, imread uses side_effect
        with patch("build_database.os.listdir", return_value=image_names) as mock_listdir, \
                patch("build_database.cv2.imread", side_effect=images) as mock_read_image, \
                patch.object(FeatureExtraction, "extract_features") as mock_feature_extraction, \
                patch.object(Harmonization, "harmonize") as mock_harmonize, \
                TestCase.assertLogs(self) as captured:

            mock_feature_extraction.side_effect = [[0,1,2],[3,4,5],[6,7,8]]
            Build_DB.makeDB("test_database/test_db.pkl", "images", is_traced=True)

            assert captured.records[3].getMessage() == \
                "The database path is invalid, make sure the path is correct and the program has the right permissions"

    def test_image_path_not_valid_makeDB(self):
        """
        Tests that the logger has the expected exception method when an image path does
        not exist.
        """

        # Note: return value is a static return value, and side effects returns different things
        # each time the mock is called. So since there are three images, imread uses side_effect
        with TestCase.assertLogs(self) as captured, \
                TestCase.assertRaises(self, SystemExit):
            Build_DB.makeDB("test_database/test_db.pkl", "images", is_traced=True)

            assert captured.records[0].getMessage() == "The provided image path does not exist"

    # Test the get_args method.
    
    def test_get_args(self):
        """
        Test that the default arguments are parsed and used when running
        build_database.py.
        """

        # Mock system argument to return the build_database.py file.
        with patch('sys.argv', return_value='build_database.py') as mock_sys:
            args = Build_DB.get_args()
            # Make sure that the result from get_args is a Namespace object.
            assert isinstance(args, argparse.Namespace)
            vars_args = vars(args)
            # Check that all arguments are present in the Namespace with their 
            # corresponding default values.
            assert len(vars_args) == 3
            assert ('input_path', \
                'dataset_images/original_dataset/Training/training_untraced') \
                    in vars_args.items()
            assert ('is_traced', False) in vars_args.items()
            assert ('db_name', 'db') in vars_args.items()


if __name__ == "__main__":
    pytest.main()  # Run this file with pytest
