// Get the two screens - the loading screen and the screen that displays the images
const display = document.getElementById("app");
const load = document.getElementById("load_screen");
const original = document.getElementById("original_image");

const get_image_path = "http://localhost:5000/get_denoised_image";
const submit_image_path = "http://localhost:5000/submit_denoised_image";
const load_image_path = "http://localhost:5000/load_denoised_images";
const get_raw_image_path = "http://localhost:5000/get_raw_image";

// While the images are being loaded, only the load screen is displayed not the
// display screen
display.style.display = "none";

// Waits for all the desired images to be loaded
load_images()
    .then(async () => {
        // After the images are loaded get all the loaded images from the backend
        promises = [];
        for (let i = 0; i < 4; i++) {
            promises.push(get_denoised_image(i));
        }

        for (p in promises) {
            await p;
        }
    })
    .then(() => {
        fetch(get_raw_image_path).then((response) => {
            response
                .text()
                .then((data) => {
                    original.src = "data:image/png;base64," + data;
                })
                .then(() => {
                    // Once images are loaded remove the loading screen and enable the display screen
                    load.style.display = "none";
                    display.style.display = "";
                });
        });
    });

// Asynchronous function that gets a specific index of a denoised image from the
// backend
async function get_denoised_image(option) {
    console.log("Option ", option);
    // Sends which number of denoised image is desired to the backend
    return fetch(get_image_path, {
        method: "POST",
        body: option,
    })
        .then((response) => {
            // Here the image itself is handled
            response.text().then((data) => {
                let el = document.getElementById("image-option-" + option);
                // Takes the text of the response (which is the image encoded in base64)
                // and displays it correctly
                el.src = "data:image/jpg;base64," + data;
                // When the image is clicked, the chosen image is submitted to the backend
                // and the page is changed.
                el.onclick = async function () {
                    submit_image(option).then(() => {
                        window.location.href = "threshold";
                    });
                };
            });
        })
        .catch((error) => {
            console.error("Error:", error);
        })
        .finally(() => {
            console.log("Done ", option);
        });
}

// Submits the image option that has been chosen by the user to the backend
async function submit_image(option) {
    return fetch(submit_image_path, {
        method: "POST",
        body: option,
    })
        .then((response) => response.blob())
        .then((blob) => {})
        .catch((error) => {
            console.error("Error:", error);
        });
}

// Loads the images from the backend. The response isn't important because this
// is all done on the backend, and stored in a variable on the backend.
async function load_images() {
    return fetch(load_image_path)
        .then((response) => {})
        .catch((error) => {
            console.error("Error:", error);
        });
}
