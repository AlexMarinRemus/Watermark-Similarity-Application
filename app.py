"""
The file for all backend GUI interactions. This handles all of the calculations for the pipeline
and sends all of the required information to the frontend.
"""

import base64
import logging
import os
import base64
from sys import platform
import os 

import cv2
import numpy as np
from flask import (Flask, abort, flash, make_response, redirect, render_template,
                   request, send_file, url_for)
from flask_cors import CORS
from PIL import Image

from database.extract_images import append_database, load_folder, load_image, reset_global_variables
from feature_extraction.feature_extraction import FeatureExtraction
from Pipeline import Pipeline

app = Flask(__name__)
app.debug = True
CORS(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.secret_key = "my_secret_key"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
db_path = "database/manual_db.pkl"
number_of_images = None
is_traced = False
index_value = 0

raw_image = None
input_image = None
denoised_images = []
denoised_image_chosen = None
wavelet_option = None
thresholded_images = []
threshold_image_chosen = None
similar_images = []
current_image_path = None


@app.route("/input")
def home():
    """
    Serves the index.html file, but also resets all of the variables, this ensures
    that there is not remains of previous runs in the new run.
    """

    global raw_image
    global index_value
    global number_of_images
    global is_traced
    global input_image
    global denoised_images
    global denoised_image_chosen
    global thresholded_images
    global threshold_image_chosen
    global similar_images
    global wavelet_option
    global db_path
    
    raw_image = None
    index_value = 0
    number_of_images = None
    is_traced = False
    input_image = None
    denoised_images = []
    denoised_image_chosen = None
    wavelet_option = None
    thresholded_images = []
    threshold_image_chosen = None
    similar_images = []
    db_path = "database/manual_db.pkl"
    
    return render_template("index.html")


@app.route("/")
def reroute_to_start():
    """
    Redirects the root to the start/input page
    """

    return redirect("http://localhost:5000/input")


@app.route("/output")
def output():
    """
    Serves the output file
    """
    return render_template("output.html")


@app.route("/edit_image")
def edit_image():
    """
    Serves the edit image file
    """
    return render_template("edit_image.html")


@app.route("/denoise")
def denoise():
    """
    Serves the denoise images file
    """
    return render_template("denoise.html")


@app.route("/threshold")
def threshold():
    """
    Serves the threshold images file
    """
    return render_template("threshold.html")


@app.after_request
def add_header(r):
    """
    Add header that ensures that the requests are not stored in cache. This was added because
    sometimes if different images were sent in succession, the images would not update due to
    caching.
    """
    r.headers["Cache-Control"] = "no-store"
    return r


@app.route("/build_database")
def database():
    """
    Endpoint for the building of the database start screen
    Returns: renders the database start screen when accessing this endpoint
    """
    reset_global_variables()
    return render_template("database_start.html")


@app.route("/process_folder", methods=["POST"])
def process():
    """
    Process the selected folder and store all image files' names from it in a global parameter
    Alerts if no folder found in the request
    Returns: forwards to the page for processing each dataset image
    """
    global db_path
    if request.files is None:
        flash("No folder input.")
    db_path = request.form["database"]
    if not os.path.exists(db_path) or not db_path.endswith(".pkl"):
        logger.info("Invalid path was entered")
        flash("Invalid path!")
        abort(400, "Invalid path")
    global APP_ROOT
    load_folder(request=request, root=APP_ROOT)
    return redirect(url_for("process_image"))


@app.route("/process_image")
def process_image():
    """
    This method sends the image whose turn it is to process
    from the database, processes it through the harmonization (future work)
    and in the end appends its feature vector to the database
    Returns: the rendered edit screen for this image
    """
    global raw_image
    global index_value
    global number_of_images
    global is_traced
    global input_image
    global denoised_images
    global denoised_image_chosen
    global thresholded_images
    global threshold_image_chosen
    global similar_images
    global wavelet_option

    raw_image = None
    index_value = 0
    number_of_images = None
    is_traced = False
    input_image = None
    denoised_images = []
    denoised_image_chosen = None
    wavelet_option = None
    thresholded_images = []
    threshold_image_chosen = None
    similar_images = []

    current_image_path, shouldTerminate = load_image()
    # If we have iterated over all the images we should go back to the beginning
    if shouldTerminate:
        current_image_path = None
        return render_template("database_start.html")

    img = Image.open(current_image_path)      # show image size (width, height)
    img = img.convert("L")   # convert to greyscale
    input_image = np.array(img)
    raw_image = input_image
    # Render the website for this current image and proceed to the next page
    return render_template("process_image.html", value=current_image_path)


@app.route("/load_denoised_images")
def load_denoised_images():
    """
    Takes the input image and then denoises it differently based on four different
    options. These are then stored in the denoised_image array.
    Note: These are not stored in the file structure because these images should not persist
    once the pipeline for these images ends.
    """
    try:
        global db_path
        global denoised_images
        denoised_images = []
        image = input_image
        for i in range(4):
            p = Pipeline(image, raw_image=raw_image, is_traced=is_traced, db_path=db_path)
            p.pre_process(i)
            p.denoise(i)

            denoised_images.append(p.get_image())
        logger.info("Denoised images loaded")
    except Exception as e:
        logger.error(e)
    return "Denoised images loaded"


@app.route("/get_denoised_image", methods=["POST"])
def get_denoised_image():
    """
    Gets the denoised image that is desired, converts it into a png then encodes
    it as a base64 string. This is then the text of the response. The image height
    and width is added in the response headers. The response is then sent to the frontend.
    """

    try:
        option = int(request.data)

        height = denoised_images[option].shape[0]
        width = denoised_images[option].shape[1]
        _, img_encoded = cv2.imencode(".png", denoised_images[option])
        png_as_text = base64.b64encode(img_encoded)

        response = make_response(png_as_text)
        response.headers["image-height"] = height
        response.headers["image-width"] = width

        logger.info(f"Denoised image {option+1}/4 sent")

        # Send the image file to the client
        return response

    except Exception as e:
        print(e)
        return "internal server error", 500


@app.route("/submit_denoised_image", methods=["POST"])
def submit_denoised_image():
    """
    The choice that was made by the user for the best denoised image
    is recieved and that specific image is then stored in a variable.
    """

    try:
        option = int(request.data)
        global denoised_image_chosen
        global wavelet_option
        denoised_image_chosen = denoised_images[option]
        wavelet_option = option + 1

        logger.info(f"Denoised image {option+1} chosen")

        return "Image denoised"

    except Exception as e:
        print(e)
        return "internal server error", 500


@app.route("/load_threshold_images")
def load_threshold_images():
    """
    Takes the input image and then thresholds it differently based on six different
    options. These are then stored in the thresholded_image array.
    Note: These are not stored in the file structure because these images should not persist
    once the pipeline for these images ends.
    """
    global db_path
    global thresholded_images
    thresholded_images = []
    for i in range(6):
        p = Pipeline(denoised_image_chosen,
                     raw_image=raw_image, is_traced=is_traced, db_path=db_path)
        p.threshold(i)
        p.post_process(i, wavelet_option=wavelet_option)

        thresholded_images.append(p.get_image())

    logger.info("Thresholded images loaded")

    return "Thresholded images loaded"


@app.route("/get_threshold_image", methods=["POST"])
def get_threshold_image():
    """
    Gets the threshold image that is desired, converts it into a png then encodes
    it as a base64 string. This is then the text of the response. The image height
    and width is added in the response headers. The response is then sent to the frontend.
    """

    try:
        option = int(request.data)

        height = thresholded_images[option].shape[0]
        width = thresholded_images[option].shape[1]
        _, img_encoded = cv2.imencode(".png", thresholded_images[option])
        png_as_text = base64.b64encode(img_encoded)

        response = make_response(png_as_text)
        response.headers["image-height"] = height
        response.headers["image-width"] = width

        logger.info(f"Thresholded image {option+1}/6 sent")

        # Send the image file to the client
        return response

    except Exception as e:
        print(e)
        return "internal server error", 500


@app.route("/submit_threshold_image", methods=["POST"])
def submit_threshold_image():
    """
    The choice that was made by the user for the best thresholded image
    is recieved and that specific image is then stored in a variable.
    """

    try:
        option = int(request.data)

        global threshold_image_chosen
        threshold_image_chosen = thresholded_images[option]

        logger.info(f"Thresholded image {option+1} chosen")

        return "Image thresholded"

    except Exception as e:
        print(e)
        return "internal server error", 500


@app.route("/get_input_image")
def get_input():
    """
    Gets the input image converts it into a png then encodes it as a base64 string.
    This is then the text of the response. The image height and width is added in the response headers.
    The response is then sent to the frontend.
    """

    global input_image
    height = input_image.shape[0]
    width = input_image.shape[1]
    _, img_encoded = cv2.imencode(".png", input_image)
    png_as_text = base64.b64encode(img_encoded)

    response = make_response(png_as_text)
    response.headers["image-height"] = height
    response.headers["image-width"] = width

    # Send the image file to the client
    return response


@app.route("/get_harmonized_image")
def get_harmonized():
    """
    Gets the harmonized image converts it into a png then encodes it as a base64 string.
    This is then the text of the response. The image height and width is added in the response headers.
    The response is then sent to the frontend.
    """

    height = threshold_image_chosen.shape[0]
    width = threshold_image_chosen.shape[1]
    _, img_encoded = cv2.imencode(".png", threshold_image_chosen)
    png_as_text = base64.b64encode(img_encoded)

    response = make_response(png_as_text)
    response.headers["image-height"] = height
    response.headers["image-width"] = width

    # Send the image file to the client
    return response


@app.route("/get_raw_image")
def get_raw():
    """
    Gets the raw image converts it into a png then encodes it as a base64 string.
    This is then the text of the response. The image height and width is added in the response headers.
    The response is then sent to the frontend.
    """

    height = raw_image.shape[0]
    width = raw_image.shape[1]
    _, img_encoded = cv2.imencode(".png", raw_image)
    png_as_text = base64.b64encode(img_encoded)

    response = make_response(png_as_text)
    response.headers["image-height"] = height
    response.headers["image-width"] = width

    # Send the image file to the client
    return response


@app.route("/get_alpha_image")
def get_editable_image():
    """
    Gets the harmonized image, then makes the black background transparent, makes the white foreground green,
    then reformats it and sends it to the front end. This is used for the edit image functionality, so that the
    edited watermark is more easily seen.
    """
    grayscale_image = threshold_image_chosen
    rgb_image = cv2.cvtColor(threshold_image_chosen, cv2.COLOR_GRAY2RGB)
    r, g, b = cv2.split(rgb_image)
    r = np.zeros(r.shape).astype(np.uint8)
    b = np.zeros(r.shape).astype(np.uint8)
    rgba = [r, g, b, grayscale_image]
    edit_image = cv2.merge(rgba, 4)

    height = edit_image.shape[0]
    width = edit_image.shape[1]
    _, img_encoded = cv2.imencode(".png", edit_image)
    png_as_text = base64.b64encode(img_encoded)

    response = make_response(png_as_text)
    response.headers["image-height"] = height
    response.headers["image-width"] = width
    # Send the image file to the client
    return response


@app.route("/send_edited_image", methods=["POST"])
def send_edited_image():
    """
    Sends the edited image from the frontend to the backend, then the image is opened
    converted to grayscale, and the threshold_image_chosen is overwritten with this new
    edited image.
    """

    try:
        global db_path
        global number_of_images
        image = request.files["image"]
        img = Image.open(image)      # show image size (width, height)
        img = img.convert("L")   # convert to greyscale
        global threshold_image_chosen
        global current_image_path
        threshold_image_chosen = np.array(img)

        # After the image is edited we run the feature extraction and append to the database
        if number_of_images is None:
            f = FeatureExtraction()
            features = f.extract_features(threshold_image_chosen)
            append_database(features)
            return "Database appended"
        return "Image received"

    except Exception as e:
        print(e)
        return "internal server error", 500


@app.route("/upload_image", methods=["POST"])
def upload_image():
    """
    Uploads the input image from the input page to the backend. Converts it to
    grayscale, and also stores whether the image is traced and how many output images are
    desired.
    """

    try:
        global input_image
        global raw_image
        image = request.files["image"]
        # open and convert to grayscale
        input_image = np.array(Image.open(image).convert("L"))
        raw_image = np.copy(input_image)

        global db_path
        global is_traced
        global number_of_images
        is_traced = request.form["is-traced"].lower() == "true"
        number_of_images = int(request.form["number-images"])
        db_path = request.form["database"]
        if not os.path.exists(db_path) or not db_path.endswith(".pkl"):
            logger.info("Invalid path was entered")
            abort(400, "Invalid path")
        
        logger.info(
            f"Uploaded image with: is_traced: {is_traced}, number_of_images: {number_of_images}")

        return "Image uploaded successfully"
    except Exception as e:
        logging.error(e)
        return "internal server error", 500


@app.route("/upload_database_image", methods=["POST"])
def upload_database_image():
    """
    Uploads the input image from the input page to the backend. Converts it to
    grayscale, and also stores whether the image is traced and how many output images are
    desired.
    """

    try:
        image = request.files["image"]
        img = Image.open(image)      # show image size (width, height)
        img = img.convert("L")   # convert to greyscale
        global input_image
        input_image = np.array(img)

        global is_traced
        is_traced = request.form["is-traced"].lower() == "true"
        return "Image uploaded successfully"
    except Exception as e:
        print(e)
        return "internal server error", 500


@app.route("/send_val")
def send_val():
    """
    Sends the number of images to output to the frontend.
    """

    global number_of_images
    return f"{number_of_images}"


@app.route("/add_image_to_output")
def send_image():
    """
    Sends a specific output image the the output page. The index of the image
    to send is determined by the index_value variable. The image name and similarity
    measure is also sent as a request header.
    """

    global index_value
    image_tuple = similar_images[index_value]
    index_value = index_value + 1
    # Get the path of the selected image
    image_path = ""
    if platform == "win32":
        image_path = image_tuple[0].replace("/", "\\")
    else:
        image_path = image_tuple[0]

    response = make_response(send_file(image_path, as_attachment=True))
    response.headers["Image-Name"] = str(image_tuple[0])
    response.headers["Similarity-Measure"] = str(image_tuple[4])
    # Send the image file
    return response


@app.route("/run_application")
def run_app():
    """
    Runs the feature extraction and similarity matching to find the similar images
    and their similarity scores. This is then stored in a similar_images array.
    """
    global db_path
    p = Pipeline(threshold_image_chosen,
                 raw_image=input_image, is_traced=is_traced, db_path=db_path)
    ranked_list = p.feature_similarity()
    global similar_images
    similar_images = ranked_list

    return "Ok"


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=False)
