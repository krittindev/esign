// import * as tf from '@tensorflow/tfjs';
// Our input frames will come from here.
const videoElement = document.getElementsByClassName("input_video")[0];
const canvasElement = document.getElementsByClassName("output_canvas")[0];
const controlsElement = document.getElementsByClassName("control-panel")[0];
const canvasCtx = canvasElement.getContext("2d");

// We'll add this to our control panel later, but we'll save it here so we can
// call tick() each time the graph runs.
const fpsControl = new FPS();

// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector(".loading");
spinner.ontransitionend = () => {
  spinner.style.display = "none";
};

// model
async function run() {
  const model = await tf.loadModel("model/model.json");
}

(async () => {
  await run();
})();

document.addEventListener("keydown", function (event) {
  if (event.code == "KeyC") {
    // clear
    document.getElementById("titleResult").innerHTML = "";
  }
  if (event.code == "KeyT") {
    // test
    var strlist = "ทดสอบ";
    if (
      document.getElementById("titleResult").innerText.length < strlist.length
    )
      document.getElementById("titleResult").innerHTML = strlist.substring(
        0,
        1 + document.getElementById("titleResult").innerText.length
      );
  }
});

function onResults(results) {
  // Hide the spinner.
  document.body.classList.add("loaded");

  // Update the frame rate.
  fpsControl.tick();

  // Draw the overlays.
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
    results.image,
    0,
    0,
    canvasElement.width,
    canvasElement.height
  );
  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let index = 0; index < results.multiHandLandmarks.length; index++) {
      const classification = results.multiHandedness[index];
      const isRightHand = classification.label === "Right";
      const landmarks = results.multiHandLandmarks[index];
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: isRightHand ? "#f98404" : "#289672",
      }),
        drawLandmarks(canvasCtx, landmarks, {
          color: isRightHand ? "#fc5404" : "#1e6f5c",
          fillColor: isRightHand ? "#f9b208" : "#29bb89",
          radius: (x) => {
            return lerp(x.from.z, -0.15, 0.1, 10, 1);
          },
        });
    }
    // const prediction = model.predict(results.multiHandLandmarks);
    // document.getElementById("titleResult").innerHTML = prediction;
  }
  canvasCtx.restore();
}

const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
  },
});
hands.onResults(onResults);

/**
 * Instantiate a camera. We'll feed each frame we receive into the solution.
 */
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 1280,
  height: 720,
});
camera.start();

// Present a control panel through which the user can manipulate the solution
// options.
new ControlPanel(controlsElement, {
  selfieMode: true,
  maxNumHands: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
})
  .add([
    new StaticText({ title: "MediaPipe Hands" }),
    fpsControl,
    new Toggle({ title: "Selfie Mode", field: "selfieMode" }),
    new Slider({
      title: "Max Number of Hands",
      field: "maxNumHands",
      range: [1, 4],
      step: 1,
    }),
    new Slider({
      title: "Min Detection Confidence",
      field: "minDetectionConfidence",
      range: [0, 1],
      step: 0.01,
    }),
    new Slider({
      title: "Min Tracking Confidence",
      field: "minTrackingConfidence",
      range: [0, 1],
      step: 0.01,
    }),
  ])
  .on((options) => {
    videoElement.classList.toggle("selfie", options.selfieMode);
    hands.setOptions(options);
  });
