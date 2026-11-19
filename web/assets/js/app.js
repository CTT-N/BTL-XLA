const state = {
    fruit: "apple",
    steps: [],
    currentStep: 0,
    datasetImages: [],
};

const elements = {
    fruitSelect: document.querySelector("#fruitSelect"),
    datasetButton: document.querySelector("#datasetButton"),
    imageUpload: document.querySelector("#imageUpload"),
    uploadCard: document.querySelector("#uploadCard"),
    emptyState: document.querySelector("#emptyState"),
    analysisView: document.querySelector("#analysisView"),
    datasetStrip: document.querySelector("#datasetStrip"),
    datasetCount: document.querySelector("#datasetCount"),
    thumbnailRow: document.querySelector("#thumbnailRow"),
    stageKicker: document.querySelector("#stageKicker"),
    stageTitle: document.querySelector("#stageTitle"),
    stageModule: document.querySelector("#stageModule"),
    beforeImage: document.querySelector("#beforeImage"),
    afterImage: document.querySelector("#afterImage"),
    stepDescription: document.querySelector("#stepDescription"),
    metrics: document.querySelector("#metrics"),
    stepCarousel: document.querySelector("#stepCarousel"),
    prevStep: document.querySelector("#prevStep"),
    nextStep: document.querySelector("#nextStep"),
    loading: document.querySelector("#loading"),
};

function setLoading(value) {
    elements.loading.classList.toggle("hidden", !value);
}

function showAnalysis(payload) {
    if (!payload.ok) {
        alert(payload.error || "Không thể phân tích ảnh.");
        return;
    }
    state.steps = payload.steps;
    state.currentStep = 0;
    elements.emptyState.classList.add("hidden");
    elements.analysisView.classList.remove("hidden");
    renderCarousel();
    renderStep();
}

function renderCarousel() {
    elements.stepCarousel.innerHTML = "";
    state.steps.forEach((step, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "step-tab";
        button.dataset.step = index;
        button.innerHTML = `
            <span class="step-number">${step.number}</span>
            <span>
                <strong>${step.title}</strong>
                <small>${step.subtitle}</small>
            </span>
        `;
        button.addEventListener("click", () => {
            state.currentStep = index;
            renderStep();
        });
        elements.stepCarousel.appendChild(button);
    });
}

function renderStep() {
    const step = state.steps[state.currentStep];
    if (!step) return;

    elements.stageKicker.textContent = `Bước ${step.number}`;
    elements.stageTitle.textContent = step.title;
    elements.stageModule.textContent = step.subtitle;
    elements.beforeImage.src = step.before;
    elements.afterImage.src = step.after;
    elements.stepDescription.textContent = step.description;

    elements.metrics.innerHTML = step.metrics.map(metric => `
        <div class="metric">
            <span>${metric.label}</span>
            <strong title="${metric.value}">${metric.value}</strong>
        </div>
    `).join("");

    document.querySelectorAll(".step-tab").forEach((tab, index) => {
        tab.classList.toggle("active", index === state.currentStep);
    });
}

async function loadDataset() {
    state.fruit = elements.fruitSelect.value;
    setLoading(true);
    try {
        const response = await fetch(`/api/dataset?fruit=${encodeURIComponent(state.fruit)}`);
        const data = await response.json();
        state.datasetImages = data.images;
        elements.datasetStrip.classList.remove("hidden");
        elements.datasetCount.textContent = `${data.images.length} ảnh`;
        elements.thumbnailRow.innerHTML = "";

        data.images.forEach((image, index) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "thumbnail";
            button.title = image.name;
            button.innerHTML = `<img src="${image.url}" alt="${image.name}">`;
            button.addEventListener("click", () => analyzeDatasetImage(index));
            elements.thumbnailRow.appendChild(button);
        });

        if (data.images.length) {
            await analyzeDatasetImage(0);
        } else {
            alert("Không tìm thấy ảnh trong tập test.");
        }
    } catch (error) {
        alert("Không thể tải danh sách dataset.");
    } finally {
        setLoading(false);
    }
}

async function analyzeDatasetImage(index) {
    const image = state.datasetImages[index];
    if (!image) return;
    document.querySelectorAll(".thumbnail").forEach((thumb, i) => {
        thumb.classList.toggle("active", i === index);
    });
    setLoading(true);
    try {
        const response = await fetch("/api/analyze-dataset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fruit: state.fruit, path: image.path }),
        });
        showAnalysis(await response.json());
    } catch (error) {
        alert("Không thể phân tích ảnh dataset.");
    } finally {
        setLoading(false);
    }
}

async function uploadImage(file) {
    if (!file) return;
    state.fruit = elements.fruitSelect.value;
    const form = new FormData();
    form.append("image", file);
    form.append("fruit", state.fruit);
    setLoading(true);
    elements.datasetStrip.classList.add("hidden");
    try {
        const response = await fetch("/api/analyze-upload", { method: "POST", body: form });
        showAnalysis(await response.json());
    } catch (error) {
        alert("Không thể tải ảnh lên.");
    } finally {
        setLoading(false);
    }
}

elements.datasetButton.addEventListener("click", loadDataset);
elements.fruitSelect.addEventListener("change", () => {
    state.fruit = elements.fruitSelect.value;
});
elements.imageUpload.addEventListener("change", event => uploadImage(event.target.files[0]));

["dragenter", "dragover"].forEach(name => {
    elements.uploadCard.addEventListener(name, event => {
        event.preventDefault();
        elements.uploadCard.classList.add("dragging");
    });
});
["dragleave", "drop"].forEach(name => {
    elements.uploadCard.addEventListener(name, event => {
        event.preventDefault();
        elements.uploadCard.classList.remove("dragging");
    });
});
elements.uploadCard.addEventListener("drop", event => {
    event.preventDefault();
    uploadImage(event.dataTransfer.files[0]);
});

elements.prevStep.addEventListener("click", () => {
    state.currentStep = (state.currentStep - 1 + state.steps.length) % state.steps.length;
    renderStep();
});
elements.nextStep.addEventListener("click", () => {
    state.currentStep = (state.currentStep + 1) % state.steps.length;
    renderStep();
});
