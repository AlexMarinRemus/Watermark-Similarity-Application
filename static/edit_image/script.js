// Canvas drawing code
// Help from: https://img.ly/blog/how-to-draw-on-an-image-with-javascript/

// Get elements from the html, such as the canvas
// used to draw with, and the buttons.
const send_edited_image_url = "http://localhost:5000/send_edited_image";
const get_harmonised_image_url = "http://localhost:5000/get_harmonized_image";

// gets the value of the user-chosen number of output images (is a get request)
const send_number_of_output_path = "http://localhost:5000/send_val";

const canvas = document.getElementById("edit-image-canvas");
const context = canvas.getContext("2d");

const home_button = document.getElementById("home_button");
const info_button = document.getElementById("info");
const continue_button = document.getElementById("continue_button");

const paint_button = document.getElementById("paintbrush");
const erase_button = document.getElementById("eraser");
const undo_button = document.getElementById("undo");
const clear_button = document.getElementById("clear");

const send_image_path = "http://localhost:5000/send_edited_image";
const get_alpha_image_path = "http://localhost:5000/get_alpha_image";
const get_raw_image_path = "http://localhost:5000/get_raw_image";
const get_harmonized_image_path = "http://localhost:5000/get_harmonized_image";

// The various types of images used for the canvas.
// Alpha image is the overlayed harmonized watermark
let alpha_image = new Image();
// Harmonized image is the binarized image that is sent to the next page
let harmonized_image = new Image();
// The raw image (background in the canvas)
let raw_image = new Image();
// The resized image, used for erasing
let resized_raw = new Image();

// Calls the asynchronous get image functions that sends a get request to
// receive the various images.
// get_harmonized_image().then(() => {
//     get_alpha_image().then(() => {
//         get_raw_image().then(() => {
//             console.log("Images acquired");
//         });
//     });
// });

// Arrays that store the paths of each drawed line
points = [];
paths = [];

// Takes the value indicated by the size scroll bar and
// assigns it to a variable that determines the line width
// used for drawing
const size_element = document.querySelector("#size-input");
let input_size_brush = size_element.value;
size_element.oninput = (e) => {
    input_size_brush = e.target.value;
};

// Add functionality that when the quit button is pressed it
// redirects the user to the home page.
home_button.onclick = function () {
    location.href = "/input";
};

// Shows instructions if the info button is pressed.
// info_button.onclick = function () {
//     alert("This screen is used for editing the harmonized image to enhance the outline of the watermark even more. The harmonized " +
//     "watermark is in green, and the background is the input image. In order to add lines to the harmonized watermark that may be missing " +
//     "click paint and draw with your mouse. In order to remove lines, such as noise, that appear in the harmonized watermark, click erase " +
//     "and draw with your mouse over the lines to be erased. The size of the paint/erase brush can be changed by moving the dot in the size bar. " +
//     "To undo a line, click the undo button. To restart everything, click restart. Note, restarting cannot be undone. When finished, click continue " +
//     "to move to the next screen.")
// }

// Only allow image drawing functionality once the images are loaded (since raw_image is loaded last
// only a listener for raw_image is used.
raw_image.addEventListener("load", () => {
    console.log("Image loaded");
    // Calculates the width and height of the image to be displayed on the
    // GUI, preserving the aspect ratio. This is done to ensure that the image
    // is not too small, and not so big that it covers parts of the UI.
    aspect_ratio = alpha_image.width / alpha_image.height;
    width = 0;
    height = 0;

    if (alpha_image.width <= alpha_image.height) {
        height = 550;
        width = aspect_ratio * 550;
    } else {
        width = 500;
        height = (1 / aspect_ratio) * 500;
    }

    canvas.width = width;
    canvas.height = height;
    canvas.style.minWidth = width;
    canvas.style.minHeight = height;
    // This draws first the raw_image so that it can be resized (since this is apparently
    // the easiest way of doing this. The resized_raw image is used for the erasing brush.
    context.drawImage(raw_image, 0, 0, width, height);
    resized_raw.src = canvas.toDataURL();

    // Draws the image on the canvas
    context.globalAlpha = 1;
    context.drawImage(raw_image, 0, 0, width, height);
    context.drawImage(alpha_image, 0, 0, width, height);

    // When the paintbrush button is pressed, then the paintbrush button
    // becomes darker (to indicate it is selected) and the erase button
    // is changed back to the lighter color (to indicate) it is not selected).
    // Then the drawing functionality is enabled with a white brush, so that the
    // foreground can be added to.
    paint_button.addEventListener("click", () => {
        paint_button.style.backgroundColor = "#d6ccc1";
        erase_button.style.backgroundColor = "#fbeee0";
        draw_on_image("rgb(0, 255, 0)", true);
        context.closePath();
    });

    // When the erase button is pressed, the erase button becomes darker
    // (to indicate it is selected) and the paintbursh button is changed back
    // to the lighter color (to indicate it is not selected). Then the drawing
    // functionality is enabled with a black brush so that the background can be added
    // to (i.e. the foreground is removed).
    erase_button.addEventListener("click", () => {
        erase_button.style.backgroundColor = "#d6ccc1";
        paint_button.style.backgroundColor = "#fbeee0";
        // Takes the resized image and draws with the pattern, removing the green harmonized
        // overlay and showing the background. In effect, erasing. No repeat simply means
        // that the pattern is not repeated at its boundaries (which shouldn't be reached anyway)
        let pattern = context.createPattern(resized_raw, "no-repeat");
        draw_on_image(pattern, false);
        context.closePath();
    });

    // When the undo button is clicked, then the image is redrawn over the previous
    // image, essentially resetting everything. Then all of the paths are drawn except
    // for the most recently added path.
    undo_button.addEventListener("click", () => {
        context.globalAlpha = 1;
        context.drawImage(raw_image, 0, 0, width, height);
        context.drawImage(alpha_image, 0, 0, width, height);

        paths.pop();
        console.log(paths);
        draw_paths();
    });

    // When the restart button is clicked, the image is redrawn over the previous image
    // which resets everything.
    clear_button.addEventListener("click", () => {
        paths = [];
        points = [];

        context.globalAlpha = 1;
        context.drawImage(raw_image, 0, 0, width, height);
        context.drawImage(alpha_image, 0, 0, width, height);

        context.closePath();
    });

    continue_button.onclick = post_image;
});

// Takes the data url of the canvas image, then converts it to a blob, which is then
// converted to a file. This file is then sent through a post method to the python code.
// This should be used to integrate with the pipeline.

// Code for conversion from data url to blob:
// https://stackoverflow.com/questions/19032406/convert-html5-canvas-into-file-to-be-uploaded
async function post_image() {
    // Sets the canvas to hidden so that the harmonized image can be added to without the user seeing it.
    // This is necessary because we want to return a black and white harmonized image, not the background image
    // with green foreground. So we take the original harmonized image and draw all the paths onto it in the appropriate
    // colors (white for paint, and black for erase)
    canvas.setAttribute("hidden", "hidden");
    context.drawImage(harmonized_image, 0, 0, width, height);
    draw_paths_binarized();

    dataURL = canvas.toDataURL();
    // Convert data url to a blob, which can be sent
    let blobBin = window.atob(dataURL.split(",")[1]);
    let array = [];
    for (let i = 0; i < blobBin.length; i++) {
        array.push(blobBin.charCodeAt(i));
    }
    let blob = new Blob([new Uint8Array(array)], { type: "image/png" });
    // Turns the blob to a file
    let file = new File([blob], "harmonized_image.png");
    // Appends the file to a form
    let formData = new FormData();
    formData.append("image", file);
    let number_of_images = await get_number_of_images();

    // If no number of images has been declared we are in database mode
    // and thus should send the image and continue to the process_image endpoint
    if (isNaN(number_of_images)) {
        await fetch(send_edited_image_url, {
            method: "POST",
            body: formData,
        })
            .then((response) => {
                console.log("Image sent");
            })
            .catch((error) => {
                // Handle error
            });
        window.location.href = "process_image";
    } else {
        // Sends the form (with the image appended) to the python code
        // through the upload_image endpoint
        await fetch(send_edited_image_url, {
            method: "POST",
            body: formData,
        })
            .then((response) => {
                console.log("Image sent");
                window.location.href = "output";
            })
            .catch((error) => {
                // Handle error
            });
    }
}

// Asynchronous function that sends a get request to get_alpha_image. The image information
// is extracted from it, and the image's source is set to it.
async function get_alpha_image() {
    return await fetch(get_alpha_image_path)
        .then((response) => response.text())
        .then((data) => {
            alpha_image.src = "data:image/jpg;base64," + data;
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

// Asynchronous function for getting the raw image, used for the canvas background
async function get_raw_image() {
    return await fetch(get_raw_image_path)
        .then((response) => response.text())
        .then((data) => {
            raw_image.src = "data:image/jpg;base64," + data;
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

// Asynchronous function for getting the harmonized image, used for sending
// the image to the output page
async function get_harmonized_image() {
    return await fetch(get_harmonized_image_path)
        .then((response) => response.text())
        .then((data) => {
            harmonized_image.src = "data:image/jpg;base64," + data;
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

// This function redraws all of the paths that have been drawn by the user onto the canvas.
// It goes through the paths array, which contains several "points" arrays, where each points
// array is a line to be drawn (i.e. a path).
function draw_paths() {
    paths.forEach((path_info) => {
        path = path_info.path_points;
        if (path.length > 0) {
            context.beginPath();
            context.lineWidth = +path_info.size + +1;
            context.strokeStyle = path_info.color;
            context.moveTo(path[0].x, path[0].y);
            for (let i = 1; i < path.length; i++) {
                context.lineTo(path[i].x, path[i].y);
                context.stroke();
            }
        }
    });
    context.closePath();
}

// Similar to draw paths but draws them with binarized colors
// based on if paint was chosen or if erase was chosen.
function draw_paths_binarized() {
    paths.forEach((path_info) => {
        color = "white";
        if (!path_info.setDraw) {
            color = "black";
        }

        path = path_info.path_points;
        if (path.length > 0) {
            context.beginPath();
            context.lineWidth = +path_info.size + +1;
            context.strokeStyle = color;
            context.moveTo(path[0].x, path[0].y);
            for (let i = 1; i < path.length; i++) {
                context.lineTo(path[i].x, path[i].y);
                context.stroke();
            }
        }
    });
    context.closePath();
}

// Asynchronous function that asks for the number of images to generate
async function get_number_of_images() {
    return await fetch(send_number_of_output_path)
        .then((response) => response.text())
        .then((data) => {
            return parseInt(data);
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

// Enables drawing on image, with a certain color and size. The color is passed in
// depending on if the user wants to draw (white) or erase (black). The size is determined
// by the size slider.
function draw_on_image(color, setDraw) {
    let isDrawing;

    // When the mouse button is pressed over the canvas, a new line is
    // started, and its color and size are set.
    canvas.onmousedown = (e) => {
        isDrawing = true;
        // r gets the coordinates of the canvas
        let r = canvas.getBoundingClientRect();

        // Checks the size bar to see if its value has changed. Draws
        // with this new value.
        size_element.oninput = (e) => {
            input_size_brush = e.target.value;
        };

        context.beginPath();
        console.log(color);
        context.lineWidth = input_size_brush;
        context.strokeStyle = color;
        // Gets the position of the mouse with e, and draws on the canvas.
        // The values are subtracted by r.left and r.top to correct for the
        // fact that the canvas does not start at (0,0).
        context.moveTo(e.clientX - r.left, e.clientY - r.top);
        points.push({ x: e.clientX - r.left, y: e.clientY - r.top });
    };

    // Whenever the mouse is moved and held down, a line is drawn.
    // When the mouse is moved without it being held down, nothing is drawn.
    canvas.onmousemove = (e) => {
        if (isDrawing) {
            let r = canvas.getBoundingClientRect();
            context.lineTo(e.clientX - r.left, e.clientY - r.top);
            points.push({ x: e.clientX - r.left, y: e.clientY - r.top });
            context.stroke();
        }
    };

    // Once the mouse is no longer being held down, the drawing stops, and
    // the path that was drawn is pushed to the array of lines.
    canvas.onmouseup = function () {
        isDrawing = false;
        if (points.length > 0) {
            //Set draw is true if paint is enabled, false for erase. Used to reconstruct paths.
            paths.push({
                color: color,
                setDraw: setDraw,
                size: input_size_brush,
                path_points: points,
            });
            console.log(paths);
        }
        points = [];
        context.closePath();
    };
}

// Added also at the end of the file to accomodate to threading
// differences between different devices and make sure that the
// system won't face any synchronisation issues
get_harmonized_image().then(() => {
    get_alpha_image().then(() => {
        get_raw_image().then(() => {
            console.log("Images acquired");
        });
    });
});
