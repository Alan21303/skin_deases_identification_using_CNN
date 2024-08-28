// Example JavaScript for handling form submission and displaying results

document.querySelector("form").addEventListener("submit", function (event) {
  event.preventDefault();

  let formData = new FormData(this);

  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.prediction) {
        document.getElementById("prediction").innerText = data.prediction;
      } else {
        document.getElementById("prediction").innerText =
          "Error during prediction";
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      document.getElementById("prediction").innerText =
        "Error during prediction";
    });
});
