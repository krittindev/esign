const data = [
  " ก",
  " ต",
  " ส",
  " พ",
  " ห",
  " ร",
  " ด",
  " ฟ",
  " ล",
  " ย",
  " ม",
  " น",
  " ง",
  " ฉ",
  " อ",
  " ี",
  " โ",
  " ไ",
  " ิ",
  " ใ",
  " 1",
  " 2",
  " 3",
  " 4",
  " 5",
];
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
let model;
async function run() {
  model = await tf.loadLayersModel(
    "https://tuliptgr.github.io/esign/model/model.json"
  );
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
  if (event.code == "KeyA") {
    // add
    document.getElementById("titleResult").innerText +=
      document.getElementById("charactorResult").innerText;
  }
});

function pre_process_landmark(landmark_list) {
  let input = [...landmark_list][0];

  console.log(input);
  // Convert to relative coordinates
  let temp = [];
  let base_x = 0,
    base_y = 0;
  for (let i = 0; i < input.length; i++) {
    if (i == 0) {
      base_x = input[i].x;
      base_y = input[i].y;
    }
    temp.push(input[i].x - base_x);
    temp.push(input[i].y - base_y);
  }

  console.log(temp);
  // Normalization
  let max_value = Math.max(...temp.map((a) => Math.abs(a)));
  let result = [...temp.map((a) => a / max_value)];

  console.log(result);
  return result;
}

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
    let prediction = model.predict(
      tf.tensor(pre_process_landmark(results.multiHandLandmarks), [1, 42])
    );

    let index = prediction.argMax(-1).dataSync()[0];
    document.getElementById("charactorResult").innerHTML = data[index];

    let acc = prediction.dataSync()[prediction.argMax(-1).dataSync()[0]];
    document.getElementById("accResult").innerHTML = acc;
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
    new StaticText({ title: "eSIGN" }),
    fpsControl,
    new Toggle({ title: "Selfie Mode", field: "selfieMode" }),
    new Slider({
      title: "จำนวนมือสูงสุด",
      field: "maxNumHands",
      range: [1, 4],
      step: 1,
    }),
    new Slider({
      title: "ค่า MDC",
      field: "minDetectionConfidence",
      range: [0, 1],
      step: 0.01,
    }),
    new Slider({
      title: "ค่า MTC",
      field: "minTrackingConfidence",
      range: [0, 1],
      step: 0.01,
    }),
  ])
  .on((options) => {
    videoElement.classList.toggle("selfie", options.selfieMode);
    hands.setOptions(options);
  });
