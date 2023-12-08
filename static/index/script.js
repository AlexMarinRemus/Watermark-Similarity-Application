const upload_image_path = "http://localhost:5000/upload_image";
//this sends an image from gui to python
function upload_image() {
    image = document.getElementById("image");
    // Verify that an image has been entered
    if (image.src === "") {
        alert("You forgot to add the image!");
        return;
    }

    // Finds the checkbox and stores if it's been clicked as a boolean, 
    // as well as the values for the database path and number of output images needed
    let is_traced = document.getElementById("traced_checkbox").checked;
    let number_images = document.getElementById("image_number_value").value;
    let db_path = document.getElementById("database").value;
    
    // Verify that a database path has been entered
    if (db_path === "") {
        alert("You forgot to add a database path!");
        return;
    }

    if (number_images < 1 || number_images > 100){
        alert("The number of images to output must be between 1 and 100!");
        return;
    }

    // Send the file, number of images, and if it's traced to the backend as a form
    let formData = new FormData();
    formData.append(
        "image",
        new File(document.getElementById("upload_image").files, "image.png")
    );
    formData.append("number-images", number_images);
    formData.append("is-traced", is_traced);
    formData.append("database", db_path);
    console.log(formData);
    fetch(upload_image_path, {
        method: "POST",
        body: formData,
    })
        .then(res => {
            // Once the backend has recieved the information change the screen
            if (res.ok) {
                window.location.href = "denoise";
            } else {
                alert("Invalid path!");
            }
        })
        .catch((error) => {
            // Handle error
        });
}

function update_image() {
    let fileInput = document.getElementById("upload_image");
    if (fileInput.files.length > 0) {
        document.getElementById("upload_image_container").style.display =
            "none";
        file = fileInput.files[0];
        image = fileInput.files[0];
        document.getElementById("image").src = new URL(
            window.URL.createObjectURL(file)
        );
    } else {
        document.getElementById("image").src = "static/upload.png";
    }
}

function set_number(n) {
    document.getElementById("image_number_value").value = n;
}

function set_slider(n) {
    document.getElementById("image_number_slider").value = n;
}
