const webcamButton = document.getElementById('webcam-button');
const predictButton = document.getElementById('predict-button');
const captureButton = document.getElementById('capture-button');
const webcamVideo = document.getElementById('webcam-video');
const webcamCanvas = document.getElementById('webcam-canvas');
const webcamPreview = document.getElementById('webcam-preview');
let imageData = null;
// Modal logic
var modal = document.getElementById("webcam-modal");
var span = document.getElementsByClassName("close")[0];
webcamButton.onclick = () => {
    modal.style.display = "block";
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { width: 256, height: 256 } })
            .then(stream => {
                webcamVideo.srcObject = stream;
                webcamVideo.onloadedmetadata = (e) => {
                    webcamVideo.play();
                    captureButton.style.display = 'block'; // Mostra il pulsante di scatto quando la webcam è attiva
                };
            });
    }
}
captureButton.onclick = () => {
    const context = webcamCanvas.getContext('2d');
    context.drawImage(webcamVideo, 0, 0, 256, 256);
    imageData = webcamCanvas.toDataURL('image/png');
    // Mostra un'anteprima dell'immagine
    webcamPreview.src = imageData;
    webcamPreview.style.display = 'block';
    predictButton.style.display = 'block';
  }
span.onclick = function() {
    modal.style.display = "none";
    if (webcamVideo.srcObject) {
        let stream = webcamVideo.srcObject;
        let tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        webcamVideo.srcObject = null;
    }
}
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
        if (webcamVideo.srcObject) {
            let stream = webcamVideo.srcObject;
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            webcamVideo.srcObject = null;
        }
    }
}
webcamVideo.onclick = () => {
    const context = webcamCanvas.getContext('2d');
    context.drawImage(webcamVideo, 0, 0, 256, 256);
    imageData = webcamCanvas.toDataURL('image/png');
    // Mostra un'anteprima dell'immagine e mostra il pulsante per fare una previsione
    webcamPreview.src = imageData;
    webcamPreview.style.display = 'block';
    predictButton.style.display = 'block';
}
predictButton.onclick = () => {
    const imageDataStr = imageData.split(',')[1]; // Rimuove "data:image/png;base64," dall'inizio dell'URL dell'immagine             
    const timestamp = Date.now(); // Ottieni il timestamp corrente
    const imageName = `webcam_image_${timestamp}.png`; // Genera un nome univoco per l'immagine
    fetch('/predict-webcam', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageDataStr }) // Invia l'immagine come JSON
    })
    .then(response => response.json())
    .then(data => {
        // Verifica se ci sono elementi di previsione nella pagina
        const predictionElements = document.getElementsByClassName('prediction-row');
        if (predictionElements.length > 0) {
            for (let i = 0; i < data.predictions.length; i++) {
                predictionElements[i].children[1].textContent = data.predictions[i];
            }
            // Aggiungi il parametro di timestamp
            document.querySelector('.prediction-image').src = data.image_url + '?t=' + Date.now(); 
        }else {
            // Creare nuovi elementi di previsione se non ne esistono
            const predictionContainer = document.createElement('div');
            predictionContainer.classList.add('model-prediction');
            
            const predictionTextContainer = document.createElement('div');
            predictionTextContainer.classList.add('prediction-text');
            predictionContainer.appendChild(predictionTextContainer);
            
            for (let i = 0; i < data.predictions.length; i++) {
                const predictionRow = document.createElement('div');
                predictionRow.classList.add('prediction-row');
                
                const predictionLabel = document.createElement('h3');
                predictionLabel.textContent = `Modello ${i+1}:`;
                predictionRow.appendChild(predictionLabel);
                
                const predictionValue = document.createElement('p');
                predictionValue.textContent = data.predictions[i];
                predictionRow.appendChild(predictionValue);
                
                predictionTextContainer.appendChild(predictionRow);
            }
            
            const predictionImage = document.createElement('img');
            predictionImage.classList.add('prediction-image');
            predictionImage.src = data.image_url;
            predictionImage.alt = 'Immagine caricata';
            predictionContainer.appendChild(predictionImage);
            document.body.appendChild(predictionContainer);
            
        }
        // Close the modal after prediction
        modal.style.display = "none";
        if (webcamVideo.srcObject) {
            let stream = webcamVideo.srcObject;
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            webcamVideo.srcObject = null;
        }
    });
}