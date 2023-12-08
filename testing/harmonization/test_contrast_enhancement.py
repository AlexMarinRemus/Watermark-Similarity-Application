"""
This file contains the class that tests all contrast enhancement methods.
Contrast enhancement methods can be found in 
'harmonization/contrast_enhancement.py'.
"""

import pytest
import numpy as np

import harmonization.contrast_enhancement as Contrast

class TestContrastEnhancement:
    """
    Class that tests all contrast enhancement methods.
    """

    def test_contrast_stretch(self):
        """
        Tests if an image that does not have intensities
        that cover the whole range of pixel values is
        contrast stretched properly. In other words, does
        the input image contain 0 and 255, if it didn't contain
        those originally.
        """

        input_img = np.array([[45, 20, 148],
                            [210, 5, 88],
                            [55, 100, 199]])
        stretched_img = Contrast.contrast_stretch(input_img)
        assert 0 in list(stretched_img.flatten())
        assert 255 in list(stretched_img.flatten())

    def test_contrast_stretch_no_change(self):
        """
        Tests that an image that already covers the whole range
        is not changed. This ensures that the contrast stretching
        operation doesn't warp the results of the input.
        """

        input_img = np.array([[45, 0, 148],
                            [210, 5, 88],
                            [255, 100, 199]])
        stretched_img = Contrast.contrast_stretch(input_img)
        assert (stretched_img == input_img).all()

    def test_normal_stretch(self):
        """
        Tests that the normal stretch method alters the input image.
        """

        input_img = np.array([[20, 100, 100, 100, 20],
                            [100, 200, 200, 200, 100],
                            [100, 200, 20, 200, 100],
                            [100, 200, 200, 200, 100],
                            [20, 100, 100, 100, 20]]).astype(np.uint8)
        stretched_img = Contrast.normal_stretch(input_img, median_size=3, \
            gaussian_size=(3,3), gaussian_sigma=3)
        assert (stretched_img != input_img).any()
        assert 0 in list(stretched_img.flatten())
        assert 255 in list(stretched_img.flatten())

    def test_stretch_flat_field(self):
        """
        Test stretch flat field method by asserting that the image
        is stretched as expected, and that the vignette effect
        is removed.
        """

        input_img = np.array([[20, 100, 100, 100, 20],
                            [100, 200, 200, 200, 100],
                            [100, 200, 20, 200, 100],
                            [100, 200, 200, 200, 100],
                            [20, 100, 100, 100, 20]]).astype(np.uint8)
        output_img = Contrast.stretch_flat_field(input_img, median_size=3, \
            gaussian_size=(3,3), gaussian_sigma=3)

        # Asserts that the image is stretched normally
        assert (output_img!= input_img).any()
        assert 0 in list(output_img.flatten())
        assert 255 in list(output_img.flatten())

        # Creates lists for input and output images that only contains values around borders
        border_input = [input_img[1][0], input_img[1][4], input_img[2][0], input_img[2][4], \
                        input_img[3][0], input_img[3][4]]
        border_input.extend(input_img[0])
        border_input.extend(input_img[4])
        border_output = [output_img[1][0], output_img[1][4], output_img[2][0], output_img[2][4], \
                        output_img[3][0], output_img[3][4]]
        border_output.extend(output_img[0])
        border_output.extend(output_img[4])
        # Asserts that the value around the border of the output is brighter than that of
        # the input.
        assert np.mean(border_input) < np.mean(border_output)


    def test_remove_shadows_values(self):
        """
        Tests that the output of removing shadows is a lighter
        image than the input image.
        """

        input_img = np.array([[100, 50, 50],
                            [255, 255, 50],
                            [255, 255, 100]]).astype(np.uint8)
        unshadowed_img = Contrast.remove_shadows(input_img, (2,2), 3)

        assert np.mean(input_img) < np.mean(unshadowed_img)

    def test_flat_field_with_vignette(self):
        """
        Flat field ensures that if the border of an image is darker/
        shadowed, then flat field correction removes that vignette
        effect. Test to see if the function makes the border lighter,
        but the inside remains
        """
        input_img = np.array([[0, 100, 100, 100, 0],
                            [100, 255, 255, 255, 100],
                            [100, 255, 0, 255, 100],
                            [100, 255, 255, 255, 100],
                            [0, 100, 100, 100, 0]]).astype(np.uint8)
        flat_field = Contrast.flat_field_correction(input_img)
        expected_center = np.array([[255, 255, 255],
                                    [255, 0, 255],
                                    [255, 255, 255]])

        # Ensures that teh center of the image is the same
        assert (flat_field[1:4,1:4] == expected_center).all()
        # Checks that border is lighter than the input, since
        # the input is the same.
        assert np.mean(input_img) < np.mean(flat_field)

    def test_flat_field_no_vignette(self):
        """
        Flat field ensures that if the border of an image is darker/
        shadowed, then flat field correction removes that vignette
        effect. Test to see if the function makes the border lighter,
        but the inside remains
        """
        input_img = np.array([[255, 255, 255, 255, 255],
                            [255, 255, 255, 255, 255],
                            [255, 255, 0, 255, 255],
                            [255, 255, 255, 255, 255],
                            [255, 255, 255, 255, 255]]).astype(np.uint8)
        flat_field = Contrast.flat_field_correction(input_img)

        assert (flat_field == input_img).all()

    def test_top_hat_on_dark_gray_image(self):
        """
        Tests that the top hat filter emphasizes an input image
        by taking the darker parts of the image and darkening them
        further, while keeping the lighter parts light.
        """
        input_img = np.array([[50, 50, 50],
                            [50, 255, 50],
                            [50, 50, 50]]).astype(np.uint8)
        top_hat = Contrast.top_hat(input_img)
        # Asserts that the border of the image is darker
        assert (top_hat[0] < 50).all()
        assert top_hat[1][0] < 50 and top_hat[1][2] < 50
        assert (top_hat[2] < 50).all()
        # Asserts that the center of the image (the point that should
        # be emphasized) is still bright.
        assert top_hat[1][1] > 200

    def test_top_hat_on_already_clear_image(self):
        """
        Tests that the top hat filter doesn't change
        an image that is already as emphasized as possible - 
        i.e. the dark parts are 0 and the light parts are 255
        """
        input_img = np.array([[0, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255]]).astype(np.uint8)
        top_hat = Contrast.top_hat(input_img)
        assert (input_img == top_hat).all()

    def test_top_hat_on_light_gray_image(self):
        """
        Tests that the top hat filter doesn't change
        an image that has very dark sections, but the
        light section is not as bright as possible.
        """
        input_img = np.array([[180, 0, 0],
                            [0, 180, 0],
                            [0, 0, 0]]).astype(np.uint8)
        top_hat = Contrast.top_hat(input_img)
        assert (input_img == top_hat).all()

    def test_CLAHE(self):
        """
        Tests that the CLAHE method outputs something different
        from the input image.
        """
        input_img = np.array([[10, 200, 10, 200, 10],
                            [200, 10, 200, 10, 200]]).astype(np.uint8)
        clahe_img = Contrast.CLAHE(input_img, tile_grid_size=(2,2))
        assert (input_img != clahe_img).any()

    
    def test_ameliorate_contrast_on_margins(self):
        """
        Test that the values larger than the most common value are set to the
        most common value.
        """
        input_img = np.array([[255, 255, 255, 255, 255],
                              [254, 255, 255, 255, 255],
                              [3, 3, 3, 3, 3],
                              [3, 3, 3, 3, 3],
                              [3, 3, 3, 3, 3],
                              [254, 254, 254, 255, 255]]).astype(np.uint8)
        image_ameliorated = Contrast.ameliorate_contrast_on_margins(input_img)
        result = np.array([[3, 3, 3, 3, 3],
                           [3, 3, 3, 3, 3],
                           [3, 3, 3, 3, 3],
                           [3, 3, 3, 3, 3],
                           [3, 3, 3, 3, 3],
                           [3, 3, 3, 3, 3]]).astype(np.uint8)
        assert np.array_equal(image_ameliorated, result)


if __name__ == '__main__':
    pytest.main()  # Run this file with pytest
