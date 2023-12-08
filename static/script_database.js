const upload_database_image = "http://localhost:5000/upload_database_image"
const input_url = "./static/input.png"
//This sends an image from the processing to the denoising module
async function handleDBImage(){
    let traced_checkbox = document.getElementById("checkbox")
    is_traced = traced_checkbox.checked
  
    // Send the file and the boolean showing whether it's traced 
    // to the backend as a form so as to store the global variables corresponding to them
    // on the server
    let file = await fetch(input_url).then(response => response.blob());
    let formData = new FormData();
    formData.append("image", file);
    formData.append("is-traced", is_traced)
    fetch(upload_database_image, {
      method: "POST",
      body: formData
    })
      .then(response => {
        // Once the backend has received the information change the screen
        window.location.href = "denoise"
    })
    .catch(error => {
      // Handle error
    });
}

const proceedButton = document.getElementById("proceed");
window.onload = function() {
    if (proceedButton) {
        proceedButton.addEventListener("click", handleDBImage);
    }
}

