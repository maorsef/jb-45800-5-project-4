const fileInput = document.querySelector("#file");
const dropzone = document.querySelector("#dropzone");
const preview = document.querySelector("#preview");
const previewWrap = document.querySelector("#previewWrap");
const dropCopy = document.querySelector("#dropCopy");
const labelEl = document.querySelector("#label");
const confidenceEl = document.querySelector("#confidence");
const statusEl = document.querySelector("#status");
const bars = document.querySelector("#bars");
const barCat = document.querySelector("#barCat");
const barDog = document.querySelector("#barDog");
const pctCat = document.querySelector("#pctCat");
const pctDog = document.querySelector("#pctDog");

function showPreview(url) {
  preview.src = url;
  previewWrap.hidden = false;
  dropCopy.hidden = true;
}

function setBusy() {
  labelEl.textContent = "מזהה...";
  confidenceEl.textContent = "שולח את התמונה למודל";
  statusEl.textContent = "רגע, רשת הנוירונים עובדת.";
  bars.hidden = true;
}

function renderResult(data) {
  labelEl.textContent = data.label_he;
  confidenceEl.textContent = `ביטחון ${Math.round(data.confidence * 100)}%`;
  const cat = data.probs.cat ?? 0;
  const dog = data.probs.dog ?? 0;
  bars.hidden = false;
  barCat.style.width = `${Math.round(cat * 100)}%`;
  barDog.style.width = `${Math.round(dog * 100)}%`;
  pctCat.textContent = `${Math.round(cat * 100)}%`;
  pctDog.textContent = `${Math.round(dog * 100)}%`;
  statusEl.textContent =
    data.label === "cat" ? "נראה כמו חתול." : "נראה כמו כלב.";
}

function renderError(message) {
  labelEl.textContent = "שגיאה";
  confidenceEl.textContent = "—";
  bars.hidden = true;
  statusEl.textContent = message;
}

async function predict(file) {
  setBusy();
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch("/api/predict", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Prediction failed");
    }
    renderResult(await res.json());
  } catch (error) {
    renderError("לא הצלחנו לזהות את התמונה. נסו קובץ אחר.");
    console.error(error);
  }
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    renderError("צריך קובץ תמונה.");
    return;
  }
  showPreview(URL.createObjectURL(file));
  predict(file);
}

fileInput.addEventListener("change", (event) => {
  handleFile(event.target.files[0]);
});

["dragenter", "dragover"].forEach((type) => {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag");
  });
});

["dragleave", "drop"].forEach((type) => {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag");
  });
});

dropzone.addEventListener("drop", (event) => {
  handleFile(event.dataTransfer.files[0]);
});

document.querySelectorAll(".sample-grid button").forEach((button) => {
  button.addEventListener("click", async () => {
    const src = button.dataset.src;
    const res = await fetch(src);
    const blob = await res.blob();
    const file = new File([blob], src.split("/").pop(), { type: blob.type });
    showPreview(src);
    predict(file);
  });
});
