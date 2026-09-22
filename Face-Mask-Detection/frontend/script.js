const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayCtx = overlayCanvas.getContext("2d");

const uploadedImage = document.getElementById("uploadedImage");

const facesElement = document.getElementById("faces");
const maskStatusElement = document.getElementById("maskStatus");
const confidenceElement = document.getElementById("confidence");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const uploadBtn = document.getElementById("uploadBtn");
const photoInput = document.getElementById("photoInput");

const ctx = canvas.getContext("2d");

let stream = null;
let detectionInterval = null;
let isDetecting = false;

/* =========================
START CAMERA
========================= */

async function startCamera() {


console.log("START BUTTON CLICKED");

if (stream) {
    console.log("Camera already running");
    return;
}

try {

    uploadedImage.style.display = "none";
    video.style.display = "block";

    stream =
        await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

    video.srcObject = stream;

    await video.play();

    console.log(
        "Camera started successfully"
    );

    maskStatusElement.innerText =
        "Detecting...";

    confidenceElement.innerText =
        "0%";

    detectionInterval =
        setInterval(
            detectMask,
            1000
        );

} catch (error) {

    console.error(
        "Camera Error:",
        error
    );

    maskStatusElement.innerText =
        "Camera Error";

    confidenceElement.innerText =
        "0%";
}


}

/* =========================
STOP CAMERA
========================= */

function stopCamera() {


console.log(
    "STOP BUTTON CLICKED"
);

if (stream) {

    stream
        .getTracks()
        .forEach(
            track => {
                track.stop();
            }
        );

    stream = null;
}

video.srcObject = null;

if (detectionInterval) {

    clearInterval(
        detectionInterval
    );

    detectionInterval = null;
}

overlayCtx.clearRect(
    0,
    0,
    overlayCanvas.width,
    overlayCanvas.height
);

facesElement.innerText =
    "0";

maskStatusElement.innerText =
    "Stopped";

confidenceElement.innerText =
    "0%";

console.log(
    "Camera stopped"
);


}

/* =========================
DRAW DETECTIONS
========================= */

function drawDetections(
results,
imageWidth,
imageHeight
) {


if (
    !results ||
    results.length === 0
) {

    overlayCtx.clearRect(
        0,
        0,
        overlayCanvas.width,
        overlayCanvas.height
    );

    return;
}

overlayCanvas.width =
    imageWidth;

overlayCanvas.height =
    imageHeight;

overlayCtx.clearRect(
    0,
    0,
    overlayCanvas.width,
    overlayCanvas.height
);

results.forEach(
    (face, index) => {

        if (!face.box) {
            return;
        }

        const x =
            face.box.x1;

        const y =
            face.box.y1;

        const width =
            face.box.x2 -
            face.box.x1;

        const height =
            face.box.y2 -
            face.box.y1;

        const confidence =
            (
                face.mask_probability *
                100
            ).toFixed(1);

        const isMasked =
            face.label ===
            "With Mask";

        const color =
            isMasked
                ? "#00ff66"
                : "#ff0000";


        /* Face Box */

        overlayCtx.strokeStyle =
            color;

        overlayCtx.lineWidth =
            5;

        overlayCtx.strokeRect(
            x,
            y,
            width,
            height
        );


        /* Label */

        const label =
            `Face ${index + 1} | ${face.label} | ${confidence}%`;

        overlayCtx.font =
            "bold 18px Arial";

        const textWidth =
            overlayCtx
                .measureText(label)
                .width;

        const labelWidth =
            textWidth + 16;

        const labelHeight =
            32;


        overlayCtx.fillStyle =
            color;

        overlayCtx.fillRect(
            x,
            Math.max(
                0,
                y - labelHeight
            ),
            labelWidth,
            labelHeight
        );


        overlayCtx.fillStyle =
            "#000000";

        overlayCtx.fillText(
            label,
            x + 8,
            Math.max(
                22,
                y - 9
            )
        );
    }
);


}

/* =========================
SHOW RESULTS
========================= */

function showResults(data) {


if (
    !data.results ||
    data.results.length === 0
) {

    facesElement.innerText =
        "0";

    maskStatusElement.innerText =
        "No Face";

    confidenceElement.innerText =
        "0%";

    overlayCtx.clearRect(
        0,
        0,
        overlayCanvas.width,
        overlayCanvas.height
    );

    return;
}


facesElement.innerText =
    data.faces;

maskStatusElement.innerHTML =
    "";

let totalConfidence =
    0;


data.results.forEach(
    (face, index) => {

        const probability =
            face.mask_probability;

        const percentage =
            (
                probability * 100
            ).toFixed(1);

        totalConfidence +=
            probability;


        const faceResult =
            document.createElement(
                "div"
            );

        faceResult.className =
            "face-result";


        if (
            face.label ===
            "With Mask"
        ) {

            faceResult.style.color =
                "#00ff66";

            faceResult.innerText =
                `🟢 Face ${index + 1}: With Mask (${percentage}%)`;

        } else {

            faceResult.style.color =
                "#ff0000";

            faceResult.innerText =
                `🔴 Face ${index + 1}: Without Mask (${percentage}%)`;
        }


        maskStatusElement
            .appendChild(
                faceResult
            );
    }
);


const averageConfidence =
    totalConfidence /
    data.results.length;


confidenceElement.innerText =
    `${(
        averageConfidence * 100
    ).toFixed(1)}%`;


}

/* =========================
CAMERA DETECTION
========================= */

async function detectMask() {


if (
    !stream ||
    !video.videoWidth ||
    !video.videoHeight ||
    isDetecting
) {
    return;
}

isDetecting = true;


canvas.width =
    video.videoWidth;

canvas.height =
    video.videoHeight;


ctx.drawImage(
    video,
    0,
    0,
    canvas.width,
    canvas.height
);


canvas.toBlob(
    async (blob) => {

        if (!blob) {

            isDetecting =
                false;

            return;
        }


        const formData =
            new FormData();


        formData.append(
            "file",
            blob,
            "frame.jpg"
        );


        try {

            const response =
                await fetch(
                    "http://127.0.0.1:8000/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            console.log(
                "FastAPI:",
                data
            );


            if (
                data.faces === 0 ||
                !data.results ||
                data.results.length === 0
            ) {

                facesElement.innerText =
                    "0";

                maskStatusElement.innerText =
                    "No Face";

                confidenceElement.innerText =
                    "0%";

                overlayCtx.clearRect(
                    0,
                    0,
                    overlayCanvas.width,
                    overlayCanvas.height
                );

                return;
            }


            showResults(data);


            drawDetections(
                data.results,
                video.videoWidth,
                video.videoHeight
            );


        } catch (error) {

            console.error(
                "API Error:",
                error
            );

            maskStatusElement.innerText =
                "API Error";

            confidenceElement.innerText =
                "0%";

        } finally {

            isDetecting =
                false;
        }

    },
    "image/jpeg",
    0.8
);


}

/* =========================
UPLOAD PHOTO
========================= */

uploadBtn.addEventListener(
"click",
() => {


    console.log(
        "UPLOAD BUTTON CLICKED"
    );

    photoInput.click();
}


);

/* =========================
PROCESS UPLOADED PHOTO
========================= */

photoInput.addEventListener(
"change",
async (event) => {


    const file =
        event.target.files[0];


    if (!file) {
        return;
    }


    console.log(
        "Uploaded photo:",
        file.name
    );


    /* Stop camera */

    if (stream) {

        stream
            .getTracks()
            .forEach(
                track => {
                    track.stop();
                }
            );

        stream = null;
    }


    if (detectionInterval) {

        clearInterval(
            detectionInterval
        );

        detectionInterval = null;
    }


    /* Show uploaded image */

    const imageURL =
        URL.createObjectURL(file);


    uploadedImage.src =
        imageURL;

    uploadedImage.style.display =
        "block";

    video.style.display =
        "none";


    maskStatusElement.innerText =
        "Analyzing...";

    confidenceElement.innerText =
        "0%";

    facesElement.innerText =
        "0";


    const formData =
        new FormData();


    formData.append(
        "file",
        file,
        file.name
    );


    try {

        const response =
            await fetch(
                "http://127.0.0.1:8000/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            throw new Error(
                `HTTP Error: ${response.status}`
            );
        }


        const data =
            await response.json();


        console.log(
            "Upload Result:",
            data
        );


        if (
            !data.results ||
            data.results.length === 0
        ) {

            showResults(data);

            return;
        }


        /* Wait until image loads */

        uploadedImage.onload =
            () => {

                drawDetections(
                    data.results,
                    uploadedImage.naturalWidth,
                    uploadedImage.naturalHeight
                );
            };


        showResults(data);


    } catch (error) {

        console.error(
            "Upload Error:",
            error
        );

        maskStatusElement.innerText =
            "API Error";

        confidenceElement.innerText =
            "0%";
    }


    photoInput.value = "";
}


);

/* =========================
BUTTON EVENTS
========================= */

startBtn.addEventListener(
"click",
startCamera
);

stopBtn.addEventListener(
"click",
stopCamera
);

console.log(
"Face REC JavaScript loaded successfully"
);
