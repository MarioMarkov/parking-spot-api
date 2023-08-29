var predictButton = document.getElementById("predict-button");
var fileInput = document.getElementById("image-input");
var xmlInput = document.getElementById("anno-input");
var resultContainer = document.getElementById("result-image");
const progress = document.getElementById("progress-bar");

predictButton.addEventListener("click", function () {
  const image = fileInput.files[0];
  const annotations = xmlInput.files[0];
  if (!annotations || !image) {
    alert("Please select an image file and annotation.");
    return;
  }

  const formData = new FormData();
  formData.append("image", image);
  formData.append("annotations", annotations);

  // const req = new XMLHttpRequest();
  // req.open("POST", "http://127.0.0.1:8000/prediction/");

  // progress.setAttribute("value", 50);

  // progress.nextElementSibling.nextElementSibling.innerText = 50 + "%";

  // req.addEventListener("load", function () {
  //   console.log(req.status);

  //   progress.setAttribute("value", 100);
  //   progress.nextElementSibling.nextElementSibling.innerText = 100 + "%";

  //   var imageEl = `<img id='result' src="data:image/jpeg;base64, ${req.response.encoded_img}" >`;
  //   resultContainer.innerHTML = imageEl;
  // });

  // req.send(formData);
  //http://127.0.0.1:8000
  //https://parking-spot-api-2.onrender.com
  //const inference_url = "http://127.0.0.1:8000";
  const inference_url = "https://parking-spot-api-2.onrender.com";
  fetch(`${inference_url}/prediction/`, {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((result) => {
      // Handle the response from the server
      var imageEl = `<img id='result' src="data:image/jpeg;base64, ${result.encoded_img}" >`;
      resultContainer.innerHTML = imageEl;
    })
    .catch((error) => {
      // Handle any errors that occur during the request
      console.error("Error:", error);
    });
});

xmlInput.addEventListener("change", function () {
  var file = xmlInput.files[0];

  if (file) {
    var reader = new FileReader();

    reader.onload = function (e) {
      // Store the uploaded image in browser storage (localStorage)
      //localStorage.setItem('uploadedAnno', e.target.result);
    };
    reader.readAsDataURL(file);
  }
});

fileInput.addEventListener("change", function () {
  var file = fileInput.files[0];

  if (file) {
    var reader = new FileReader();

    reader.onload = function (e) {
      // Store the uploaded image in browser storage (localStorage)
      //localStorage.setItem('uploadedImage', e.target.result);

      // Display the uploaded image
      var imageContainer = document.getElementById("image-container");
      imageContainer.innerHTML =
        "<img id='image' src=\"" + e.target.result + '">';
    };

    reader.readAsDataURL(file);
  }
});
