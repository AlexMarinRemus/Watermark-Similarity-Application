// Gets the divs for the images
const inputImage = document.getElementById("input-pic");
const harmonizedImage = document.getElementById("harmonized-pic");
const imagesDiv = document.getElementById("images");
// Get the two screens - the loading screen and the screen that displays the images
const display = document.getElementById("app");
const load = document.getElementById("load_screen");

const add_image_to_output_path = "http://localhost:5000/add_image_to_output";
const send_number_of_output_path = "http://localhost:5000/send_val";
const get_input_image_path = "http://localhost:5000/get_input_image";
const get_harmonized_image_path = "http://localhost:5000/get_harmonized_image";
const run_application_path = "http://localhost:5000/run_application";

// While the images are being loaded, only the load screen is displayed not the
// display screen
rank_index = 0
display.style.display = "none";

output_images = []
temp_img_name = null
temp_similarity_measure = 0

// Waits for the application (the feature extraction and similarity testing) to finish running
run_application().then(() => {
    // Once the final images have been calculated remove the loading screen and replace with the
    // display screen
    load.style.display = "none";
    display.style.display = "";
    // Get the input and harmonized images from the backend
    get_input_image().then(() => {
        get_harmonized_image().then(() => {
            // Gets the number of images from the backend
            get_number_of_images().then(async (number_of_images) => {
                //takes the number of similar images needed and then add them to the images div
                for (let i = 0; i < number_of_images; i++) {
                    // Fetches each image that has been loaded and adds it to the image in the div
                    // Wrapped in a try catch in case more images are loaded than exist in the database.
                    try{
                        await add_image_to_output()
                        console.log("Image " + String(rank_index) + " loaded")
                    }
                    catch (error) {
                        console.log("An error has occured with loading the images: " + error)
                    }
                }
            }).then(() => {
                console.log(output_images)
            })
        });
    });
});

// Asynchronous function that asks for the number of images to generate
async function get_number_of_images() {
    return fetch(send_number_of_output_path)
        .then((response) => response.text())
        .then((data) => {
            return parseInt(data);
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

// Asynchronous function that asks for the input image
async function get_input_image() {
    return fetch(get_input_image_path)
        .then((response) => {
            // Once the desired input image is recieved, get its height and width
            // then resize the image so that its aspect ratio stays the same.
            let height_input = parseInt(response.headers.get("image-height"));
            let width_input = parseInt(response.headers.get("image-width"));

            let aspect_ratio_input = width_input / height_input;

            if (height_input > width_input) {
                height_input = 200;
                width_input = aspect_ratio_input * 200;
            } else {
                width_input = 200;
                height_input = (1 / aspect_ratio_input) * 200;
            }

            // Here the image itself is displayed and resized based on height and width
            response.text().then((data) => {
                inputImage.style.width = String(width_input) + "px";
                inputImage.style.height = String(height_input) + "px";
                inputImage.src = "data:image/jpg;base64," + data;
            });
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

// Asynchronous function that asks for the harmonized image
async function get_harmonized_image() {
    return fetch(get_harmonized_image_path)
        .then((response) => {
            // Once the desired input image is recieved, get its height and width
            // then resize the image so that its aspect ratio stays the same.
            let height_harmonize = parseInt(
                response.headers.get("image-height")
            );
            let width_harmonize = parseInt(response.headers.get("image-width"));

            let aspect_ratio_harmonize = width_harmonize / height_harmonize;

            if (height_harmonize > width_harmonize) {
                height_harmonize = 200;
                width_harmonize = aspect_ratio_harmonize * 200;
            } else {
                width_harmonize = 200;
                height_harmonize = (1 / aspect_ratio_harmonize) * 200;
            }
            // Here the image itself is displayed and resized based on height and width
            response.text().then((data) => {
                harmonizedImage.style.width = String(width_harmonize) + "px";
                harmonizedImage.style.height = String(height_harmonize) + "px";
                harmonizedImage.src = "data:image/jpg;base64," + data;
            });
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

// Runs the feature extraction and similarity matching on the backend
async function run_application() {
    return fetch(run_application_path)
        .then((response) => {})
        .catch((error) => {
            console.error("Error:", error);
        });
}

function goBackInput() {
    window.location.href = "input";
}

let quitButton = document.getElementById("quit");
if (quitButton) {
    quitButton.addEventListener("click", goBackInput);
}

async function loadMoreImages(){
    // Fetches addtionals images that have been loaded and adds it to the image in the div
    // Wrapped in a try catch in case more images are loaded than exist in the database.
    try{
        for (let i = 0; i < 10; i++) {
            await add_image_to_output()
            console.log("Image " + String(rank_index) + " loaded")
        }
    }
    catch (error) {
        console.log("An error has occured with loading more images: " + error)
    }
}

// Function for adding similar images to the output screen
async function add_image_to_output(){
    // Fetches each image that has been loaded and adds it to the image in the div
    return fetch(add_image_to_output_path)
        .then((response) => {
            // Get the similarity measure and the image name through the response header
            console.log(rank_index)
            console.log(response.headers.get("Image-Name"));
            console.log(
                response.headers.get("Similarity-Measure")
            );

            // These variables, the name and the measure, can't be returned with the blob
            // since the blob needs to be handled asynchronously. Instead they are saved to
            // temporary variables. This works because of asynchronous waiting.
            temp_img_name = response.headers.get("Image-Name");
            temp_similarity_measure = response.headers.get("Similarity-Measure");

            return response.blob()
        }).then((blob) => {
            // return URL.createObjectURL(blob);
            let imageUrl = URL.createObjectURL(blob);
            let divImage = document.createElement("div");
            let imageElement = document.createElement("img");
            imageElement.setAttribute(
                "class",
                "output-image"
            );
            let textLabel = document.createElement("p");
            let imageName = temp_img_name;
            let shortName = imageName.split("/");
            textLabel.innerHTML =
                "Rank: " +
                (rank_index + 1) +
                "<br>" +
                shortName[shortName.length - 1] +
                "<br>Confidence: <span class='confidence'>" +
                (
                    100 * temp_similarity_measure
                ).toFixed(2) +
                "%</span>";
            imageElement.src = imageUrl;
            divImage.appendChild(imageElement);
            divImage.appendChild(textLabel);
            imagesDiv.appendChild(divImage);

            rank_index += 1;
        })
    .catch((error) => {
        console.error("Error:", error);
    });
}
