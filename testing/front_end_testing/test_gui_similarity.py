import pytest
import os
import signal
import time
import subprocess
from sys import platform
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from webdriver_manager.core.utils import ChromeType
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


class Test_GUI_Watermark_Similarity():

    driver = webdriver.Chrome(ChromeDriverManager(version="113.0.5672.63",
                                                  chrome_type=ChromeType.CHROMIUM).install())

    def _run_input_screen(self):
        driver = self.driver
        driver.find_element(By.ID, "upload_image").send_keys(os.getcwd() +
                                    "/dataset_images/original_dataset/Training/training_traced/1_1.jpg")
        driver.find_element(By.ID, "image_number_value").clear()
        driver.find_element(By.ID, "image_number_value").send_keys("5")
        checkbox = driver.find_element(By.ID, "traced_checkbox")
        if (checkbox.get_attribute("checked") is None):
            checkbox.click()
        driver.find_element(By.ID, "submit_button").click()

    def pytest_namespace():
        """
        Defines global test variables, since __init__ cannot be used with pytest.
        """
        return {"p": None}

    @pytest.fixture(scope='session', autouse=True)
    def driver_configuration(self):
        """
        Sets up the driver before all the tests are run, then yields for the tests, and finally
        breaks down the driver
        """

        # pytest.driver = webdriver.Chrome(ChromeDriverManager(version="113.0.5672.63", \
        #     chrome_type=ChromeType.CHROMIUM).install())

        if platform == "win32":
            pytest.p = subprocess.Popen("python app.py", stdout=subprocess.PIPE,
                                        shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            pytest.p = subprocess.Popen("python app.py", stdout=subprocess.PIPE,
                                        shell=True, preexec_fn=os.setsid)


        time.sleep(4)

        # Waits for all the pytests to complete before destroying the driver
        yield

    def test_title_header_input_page(self):
        """
        Tests that the expected elements are seen when accessing /input
        """

        driver = self.driver
        driver.get("http://localhost:5000/input")

        assert "Watermark Similarity" in driver.title

        title = driver.find_element(By.ID, "title_header")
        assert title.text == "Upload your image"

    def test_redirect_to_input_page(self):
        """
        Tests that accessing / will lead to /input
        """

        driver = self.driver
        driver.get("http://localhost:5000/")

        assert "Watermark Similarity" in driver.title

        title = driver.find_element(By.ID, "title_header")
        assert title.text == "Upload your image"

    def test_alert_if_no_image(self):
        """
        Tests that the correct alert pops up if no image is input (but a number of
        images to output is input)
        """

        driver = self.driver
        driver.get("http://localhost:5000/input")

        driver.find_element(By.ID, "image_number_value").clear()
        driver.find_element(By.ID, "image_number_value").send_keys("5")
        driver.find_element(By.ID, "submit_button").click()

        alert = driver.switch_to.alert
        assert alert.text == "You forgot to add the image!"
        alert.accept()

    def test_alert_if_no_database(self):
        """
        Tests that the correct alert pops up if no image is input (but a number of
        images to output is input)
        """

        driver = self.driver
        driver.get("http://localhost:5000/input")

        driver.find_element(By.ID, "upload_image").send_keys(os.getcwd() +
                                "/dataset_images/original_dataset/Training/training_traced/1_1.jpg")

        driver.find_element(By.ID, "database").clear()
        driver.find_element(By.ID, "database").send_keys("")
        driver.find_element(By.ID, "submit_button").click()

        alert = driver.switch_to.alert
        assert alert.text == "You forgot to add a database path!"
        alert.accept()

    def test_alert_if_wrong_database(self):
        """
        Tests that the correct alert pops up if no image is input (but a number of
        images to output is input)
        """

        driver = self.driver
        driver.get("http://localhost:5000/input")

        driver.find_element(By.ID, "upload_image").send_keys(os.getcwd() +
                                    "/dataset_images/original_dataset/Training/training_traced/1_1.jpg")

        driver.find_element(By.ID, "database").clear()
        driver.find_element(By.ID, "database").send_keys("database/man.pkl")
        driver.find_element(By.ID, "submit_button").click()

        WebDriverWait(driver, 60).until(
            EC.alert_is_present())

        alert = driver.switch_to.alert
        assert alert.text == "Invalid path!"
        alert.accept()

    def test_alert_if_number_out_of_range(self):
        """
        Tests that the correct alert pops up if the number of output images is less than 1
        or greater than 100
        """

        driver = self.driver
        driver.get("http://localhost:5000/input")

        driver.find_element(By.ID, "upload_image").send_keys(os.getcwd() +
                                "/dataset_images/original_dataset/Training/training_traced/1_1.jpg")

        driver.find_element(By.ID, "image_number_value").clear()
        driver.find_element(By.ID, "image_number_value").send_keys("0")
        driver.find_element(By.ID, "submit_button").click()

        alert = driver.switch_to.alert
        assert alert.text == "The number of images to output must be between 1 and 100!"
        alert.accept()

        driver.find_element(By.ID, "image_number_value").clear()
        driver.find_element(By.ID, "image_number_value").send_keys("101")
        driver.find_element(By.ID, "submit_button").click()

        alert = driver.switch_to.alert
        assert alert.text == "The number of images to output must be between 1 and 100!"
        alert.accept()

    def test_transition_page_when_input_correct(self):
        """
        Tests that when all input is entered correctly, the page is transitioned to the denoise page
        """

        driver = self.driver
        driver.get("http://localhost:5000/input")

        self._run_input_screen()

        # Since it takes a second for webpages to load, this line of code waits for the body
        # of the next page to come by waiting for the loading screen
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "denoise_header")))

        assert driver.current_url == "http://localhost:5000/denoise"
        assert driver.find_element(
            By.ID, "denoise_header").text == "Denoising"

    def test_denoise_page_elements(self):
        """
        Tests that the denoising screen appears as expected, and has its required elements
        """

        driver = self.driver
        driver.get("http://localhost:5000/input")

        self._run_input_screen()

        # Since it takes a second for webpages to load, this line of code waits for the body
        # of the next page to come
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.ID, "load_screen")))

        assert driver.current_url == "http://localhost:5000/denoise"

        # Here we wait for at most 60 seconds, until the loading screen is finished + 1 second to elt the images become visible
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "app")))

        # Checks that the images are displayed
        assert driver.find_element(By.ID, "image-option-0").is_displayed()
        assert driver.find_element(By.ID, "image-option-1").is_displayed()
        assert driver.find_element(By.ID, "image-option-2").is_displayed()
        assert driver.find_element(By.ID, "image-option-3").is_displayed()

        # Checks that the header text is there
        assert driver.find_element(
            By.ID, "denoise_header").text == "Denoising"

        # Checks that the quit button reroutes to the input page
        driver.find_element(By.ID, "home_button").click()
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.ID, "upload_image")))
        assert driver.current_url == "http://localhost:5000/input"

    def test_threshold_page_elements(self):
        """
        Tests that the thresholding screen appears as expected, and has its required elements
        """
        driver = self.driver
        driver.get("http://localhost:5000/input")

        self._run_input_screen()

        # Here we wait for the loading of the denoising image to finish
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "denoise_header")))

        # Wait for the denoising images to load and be clickable
        WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.ID, "image-option-0")))

        # Click one of the denoised images to progress to the next screen
        driver.find_element(By.ID, "image-option-0").click()

        # Since it takes a second for webpages to load, this line of code waits for the body
        # of the threshold page to come
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "threshold_header")))

        # Checks that once the body is present, the load screen is seen
        assert driver.current_url == "http://localhost:5000/threshold"
        assert driver.find_element(By.ID, "app").is_displayed()

        # Checks that the images are displayed
        assert driver.find_element(By.ID, "image-option-0").is_displayed()
        assert driver.find_element(By.ID, "image-option-1").is_displayed()
        assert driver.find_element(By.ID, "image-option-2").is_displayed()
        assert driver.find_element(By.ID, "image-option-3").is_displayed()
        assert driver.find_element(By.ID, "image-option-4").is_displayed()
        assert driver.find_element(By.ID, "image-option-5").is_displayed()

        # Checks that the header text is there
        assert driver.find_element(
            By.ID, "threshold_header").text == "Thresholding"

        # Checks that the quit button reroutes to the input page
        driver.find_element(By.ID, "home_button").click()
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.ID, "upload_image")))
        assert driver.current_url == "http://localhost:5000/input"

    def test_edit_image_screen(self):
        """
        Tests that the edit screen has paint, erase, undo, and restart functionality.
        """

        driver = self.driver
        driver.get("http://localhost:5000/input")

        self._run_input_screen()

       # Here we wait for the loading for the denoise screen to finish
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "denoise_header")))

        # Wait for the denoising images to load and be clickable
        WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.ID, "image-option-0")))

        # Click one of the denoised images
        driver.find_element(By.ID, "image-option-0").click()

        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "threshold_header")))

        # Wait for the loading of the threshold page to finish
        WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.ID, "image-option-0")))

        # Click one of the denoised images
        driver.find_element(By.ID, "image-option-0").click()

        # driver.execute_script(
        #     "arguments[0].click();", driver.find_element(By.ID, "image-option-0"))

        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "image_edit_header")))

        # Wait for the edit image page to load
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "edit-image-canvas")))

        if platform == "win32":
            time.sleep(5)
        
        # Get the canvas
        canvas = driver.find_element(By.ID, "edit-image-canvas")
        # Get the encoding of the image, this will be used to compare to changes made in the image
        canvas_original = driver.execute_script(
            "return arguments[0].toDataURL('image/png').substring(21);", canvas)

        # Used for moving around the mouse on the canvas
        action = ActionChains(driver)

        WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.ID, "paintbrush")))

        # First try painting on the canvas
        # driver.find_element(By.ID, "paintbrush").click()
        driver.execute_script(
            "arguments[0].click();", driver.find_element(By.ID, "paintbrush"))
        # Clicks and drags the mouse to specific coordinates with the paintbrush option enabled
        action.move_to_element(canvas).click_and_hold(
            canvas).move_by_offset(40, 45).release().perform()

        if platform == "win32":
            time.sleep(5)

        # Gets the encoded image that has been painted over, and asserts that it is not the same as the unpainted
        # image.
        canvas_painted = driver.execute_script(
            "return arguments[0].toDataURL('image/png').substring(21);", canvas)
        assert canvas_original != canvas_painted

        # Undoes the change and asserts that the image after the undo is the same as the original
        driver.find_element(By.ID, "undo").click()

        if platform == "win32":
            time.sleep(5)
        
        canvas_undone = driver.execute_script(
            "return arguments[0].toDataURL('image/png').substring(21);", canvas)
        assert canvas_original == canvas_undone

        # The eraser could not be clicked normally, so instead execute_script is used
        erase_button = driver.find_element(By.ID, "eraser")
        driver.execute_script("arguments[0].click();", erase_button)

        # Checks that when erasing the canvas, it is not the same as original unerased canvas, and also not the
        # same as the painted canvas
        action.move_to_element(canvas).click_and_hold(
            canvas).move_by_offset(40, 45).release().perform()
        
        if platform == "win32":
            time.sleep(5)

        canvas_erased = driver.execute_script(
            "return arguments[0].toDataURL('image/png').substring(21);", canvas)
        assert canvas_original != canvas_erased
        assert canvas_painted != canvas_erased

        # Clears the whole canvas and makes sure that it is the same as the original
        driver.find_element(By.ID, "clear").click()
        canvas_cleared = driver.execute_script(
            "return arguments[0].toDataURL('image/png').substring(21);", canvas)
        assert canvas_original == canvas_cleared

    def test_output_display_elements(self):
        """
        Tests that the output page has the expected elements.
        """

        driver = self.driver
        driver.get("http://localhost:5000/input")

        number_to_output = 5

        self._run_input_screen()

        # Here we wait for the loading for the denoise screen to finish
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "denoise_header")))

        # Wait for the denoising images to load and be clickable
        WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.ID, "image-option-0")))

        # Click one of the denoised images
        driver.find_element(By.ID, "image-option-0").click()

        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "threshold_header")))

        # Wait for the loading of the threshold page to finish
        WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.ID, "image-option-0")))

        # Click one of the denoised images
        driver.find_element(By.ID, "image-option-0").click()

        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "image_edit_header")))

        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "continue_button")))

        # Waits for the edit image body screen to load and immediately continues past it
        if platform == "win32":
            time.sleep(5)

        driver.execute_script(
            "arguments[0].click();", driver.find_element(By.ID, "continue_button"))

        # Waits for the final display screen to load
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.ID, "output_header")))

        if platform == "win32":
            time.sleep(5)

        # Makes sure that the url and elements are as expected
        assert driver.current_url == "http://localhost:5000/output"
        assert driver.find_element(By.ID, "input-pic").is_displayed()
        assert driver.find_element(By.ID, "harmonized-pic").is_displayed()

        # Waits for all the output images to load
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CLASS_NAME, "output-image")))

        # Checks that the number of output images is the expected value (the value that was input)
        images = driver.find_elements(By.CLASS_NAME, "output-image")
        assert len(images) == number_to_output

        # Gets the image sources
        image_sources = []
        for image in images:
            image_sources.append(image.get_attribute("src"))

        # Checks that each source was taken from the correct localhost port, and that
        # each source occurs in the list only once (meaning that each image is unique)
        for src in image_sources:
            assert "http://localhost:5000" in src
            assert image_sources.count(src) == 1

    def test_tear_down(self):
        # Resetting the driver prevents a weird request error
        driver = self.driver
        driver.get("http://localhost:5000/input")
        self.driver.close()
        self.driver.quit()

        if platform == "win32":
            pytest.p.send_signal(signal.CTRL_BREAK_EVENT)
            pytest.p.kill()
        else:
            os.kill(os.getpgid(pytest.p.pid), signal.SIGTERM)
