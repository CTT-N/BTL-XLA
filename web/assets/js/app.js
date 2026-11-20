const PAGE_SIZE = 10;

const state = {
    fruit: "apple",
    rawSteps: [],
    slides: [],
    currentSlide: 0,
    dataset: {
        healthy: [],
        rotten: [],
        healthyPage: 0,
        rottenPage: 0,
        activePath: null,
    },
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
    healthyRow: document.querySelector("#healthyRow"),
    rottenRow: document.querySelector("#rottenRow"),
    healthyPageInfo: document.querySelector("#healthyPageInfo"),
    rottenPageInfo: document.querySelector("#rottenPageInfo"),
    healthyPrev: document.querySelector("#healthyPrev"),
    healthyNext: document.querySelector("#healthyNext"),
    rottenPrev: document.querySelector("#rottenPrev"),
    rottenNext: document.querySelector("#rottenNext"),
    stageKicker: document.querySelector("#stageKicker"),
    stageTitle: document.querySelector("#stageTitle"),
    stageModule: document.querySelector("#stageModule"),
    resultBadge: document.querySelector("#resultBadge"),
    beforeLabel: document.querySelector("#beforeLabel"),
    afterLabel: document.querySelector("#afterLabel"),
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

// Chuyển đổi cấu trúc steps + substep thành chuỗi slide Before -> After nối tiếp nhau
function buildSlides(steps) {
    const slides = [];
    let prevImg = null;
    let prevTitle = "Ảnh gốc";

    const pushSlide = (item, moduleIndex, step, isFirstOfAll) => {
        const currentImg = item.img || step.after || step.before;
        const beforeImg = item.before || prevImg || currentImg;
        const beforeText = isFirstOfAll ? "Đầu vào gốc" : `Trước (${prevTitle})`;

        slides.push({
            moduleIndex,
            kicker: `Bước ${step.number} · ${step.title}`,
            title: item.title,
            moduleBadge: step.subtitle,
            beforeLabel: beforeText,
            afterLabel: item.title,
            before: beforeImg,
            after: currentImg,
            description: item.description || step.description || "",
            metrics: item.metrics || step.metrics || [],
        });

        prevImg = currentImg;
        prevTitle = item.title;
    };

    steps.forEach((step, moduleIndex) => {
        if (Array.isArray(step.substep) && step.substep.length > 0) {
            step.substep.forEach((sub, subIndex) => {
                // Nếu có các bước con cấp 3 (2.3.1 -> 2.3.6):
                // Chạy hết các bước quá trình TRƯỚC, rồi mới hiển thị bước tổng kết 2.3 SAU CÙNG
                if (Array.isArray(sub.substep) && sub.substep.length > 0) {
                    sub.substep.forEach((nestedSub, nestedIdx) => {
                        pushSlide(nestedSub, moduleIndex, step, moduleIndex === 0 && subIndex === 0 && nestedIdx === 0);
                    });
                    pushSlide(sub, moduleIndex, step, false);
                } else {
                    pushSlide(sub, moduleIndex, step, moduleIndex === 0 && subIndex === 0);
                }
            });
        } else {
            const beforeImg = step.before || prevImg;
            const afterImg = step.after || step.img;
            slides.push({
                moduleIndex,
                kicker: `Bước ${step.number}`,
                title: step.title,
                moduleBadge: step.subtitle,
                beforeLabel: "Tổng hợp vùng lỗi",
                afterLabel: "Kết quả phân loại",
                before: beforeImg,
                after: afterImg,
                description: step.description || "",
                metrics: step.metrics || [],
            });
            prevImg = afterImg;
            prevTitle = step.title;
        }
    });

    return slides;
}

function showAnalysis(payload) {
    if (!payload.ok) {
        alert(payload.error || "Không thể phân tích ảnh.");
        return;
    }
    state.rawSteps = payload.steps || [];
    state.slides = buildSlides(state.rawSteps);
    state.currentSlide = 0;

    if (elements.resultBadge && payload.label) {
        elements.resultBadge.style.display = "inline-block";
        elements.resultBadge.textContent = `Kết quả: ${payload.label}`;
    }

    elements.emptyState.classList.add("hidden");
    document.getElementById("videoView")?.classList.add("hidden"); // Thêm dòng này
    document.getElementById("evalView")?.classList.add("hidden");  // Thêm dòng này
    elements.analysisView.classList.remove("hidden");
    renderCarousel();
    renderStep();
}

function renderCarousel() {
    elements.stepCarousel.innerHTML = "";
    state.rawSteps.forEach((step, moduleIndex) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "step-tab";
        button.dataset.module = moduleIndex;
        button.innerHTML = `
            <span class="step-number">${step.number}</span>
            <span>
                <strong>${step.title}</strong>
                <small>${step.subtitle}</small>
            </span>
        `;
        button.addEventListener("click", () => {
            const targetIdx = state.slides.findIndex(s => s.moduleIndex === moduleIndex);
            if (targetIdx !== -1) {
                state.currentSlide = targetIdx;
                renderStep();
            }
        });
        elements.stepCarousel.appendChild(button);
    });
}

function renderStep() {
    const slide = state.slides[state.currentSlide];
    if (!slide) return;

    elements.stageKicker.textContent = `${slide.kicker} (${state.currentSlide + 1}/${state.slides.length})`;
    elements.stageTitle.textContent = slide.title;
    elements.stageModule.textContent = slide.moduleBadge;

    if (elements.beforeLabel) elements.beforeLabel.textContent = slide.beforeLabel;
    if (elements.afterLabel) elements.afterLabel.textContent = slide.afterLabel;

    elements.beforeImage.src = slide.before;
    elements.afterImage.src = slide.after;
    elements.stepDescription.textContent = slide.description;

    elements.metrics.innerHTML = (slide.metrics || []).map(metric => `
        <div class="metric">
            <span>${metric.label}</span>
            <strong title="${metric.value}">${metric.value}</strong>
        </div>
    `).join("");

    document.querySelectorAll(".step-tab").forEach((tab, index) => {
        tab.classList.toggle("active", index === slide.moduleIndex);
    });
}

function renderDatasetRow(group) {
    const list = state.dataset[group];
    const pageKey = `${group}Page`;
    const rowEl = elements[`${group}Row`];
    const infoEl = elements[`${group}PageInfo`];

    const totalPages = Math.max(1, Math.ceil(list.length / PAGE_SIZE));
    state.dataset[pageKey] = Math.max(0, Math.min(state.dataset[pageKey], totalPages - 1));

    const start = state.dataset[pageKey] * PAGE_SIZE;
    const slice = list.slice(start, start + PAGE_SIZE);

    infoEl.textContent = list.length
        ? `${start + 1}–${start + slice.length} / ${list.length}`
        : "0 ảnh";

    rowEl.innerHTML = "";
    slice.forEach((image) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "thumbnail";
        if (image.path === state.dataset.activePath) {
            button.classList.add("active");
        }
        button.title = `${image.label.toUpperCase()} — ${image.name}`;
        button.innerHTML = `<img src="${image.url}" alt="${image.name}" loading="lazy">`;
        button.addEventListener("click", () => analyzeDatasetImage(image));
        rowEl.appendChild(button);
    });
}

async function loadDataset() {
    state.fruit = elements.fruitSelect.value;
    setLoading(true);
    try {
        const response = await fetch(`/api/dataset?fruit=${encodeURIComponent(state.fruit)}`);
        const data = await response.json();

        state.dataset.healthy = data.healthy || [];
        state.dataset.rotten = data.rotten || [];
        state.dataset.healthyPage = 0;
        state.dataset.rottenPage = 0;

        elements.datasetStrip.classList.remove("hidden");
        elements.datasetCount.textContent = `Tổng ${data.total} ảnh (${state.dataset.healthy.length} tươi · ${state.dataset.rotten.length} hỏng)`;

        renderDatasetRow("healthy");
        renderDatasetRow("rotten");

        const firstImage = state.dataset.healthy[0] || state.dataset.rotten[0];
        if (firstImage) {
            await analyzeDatasetImage(firstImage);
        } else {
            alert("Không tìm thấy ảnh trong tập test.");
        }
    } catch (error) {
        alert("Không thể tải danh sách dataset.");
    } finally {
        setLoading(false);
    }
}

async function analyzeDatasetImage(image) {
    if (!image) return;
    state.dataset.activePath = image.path;
    renderDatasetRow("healthy");
    renderDatasetRow("rotten");

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

// --- ĐĂNG KÝ SỰ KIỆN ---
elements.datasetButton.addEventListener("click", loadDataset);
elements.imageUpload.addEventListener("change", event => uploadImage(event.target.files[0]));

elements.fruitSelect.addEventListener("change", () => {
    state.fruit = elements.fruitSelect.value;
    if (!elements.datasetStrip.classList.contains("hidden")) {
        loadDataset();
    }
});

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
    if (!state.slides.length) return;
    state.currentSlide = (state.currentSlide - 1 + state.slides.length) % state.slides.length;
    renderStep();
});
elements.nextStep.addEventListener("click", () => {
    if (!state.slides.length) return;
    state.currentSlide = (state.currentSlide + 1) % state.slides.length;
    renderStep();
});

elements.healthyPrev.addEventListener("click", () => {
    state.dataset.healthyPage--;
    renderDatasetRow("healthy");
});
elements.healthyNext.addEventListener("click", () => {
    state.dataset.healthyPage++;
    renderDatasetRow("healthy");
});
elements.rottenPrev.addEventListener("click", () => {
    state.dataset.rottenPage--;
    renderDatasetRow("rotten");
});
elements.rottenNext.addEventListener("click", () => {
    state.dataset.rottenPage++;
    renderDatasetRow("rotten");
});


//video
(() => {
    const fruitSelect = document.getElementById("fruitSelect");
    const videoUpload = document.getElementById("videoUpload");
    const demoVideoButton = document.getElementById("demoVideoButton");
    const emptyState = document.getElementById("emptyState");
    const analysisView = document.getElementById("analysisView");
    const videoView = document.getElementById("videoView");
    const loading = document.getElementById("loading");
    const resultVideoPlayer = document.getElementById("resultVideoPlayer");
    const videoStatsBadge = document.getElementById("videoStatsBadge");
    const videoResultsSummary = document.getElementById("videoResultsSummary");

    function showVideoResult(data) {
        if (!data.ok) {
            alert(data.error || "Xử lý video thất bại.");
            return;
        }
        emptyState?.classList.add("hidden");
        analysisView?.classList.add("hidden");
        document.getElementById("evalView")?.classList.add("hidden");
        videoView?.classList.remove("hidden");

        resultVideoPlayer.src = data.video_url;
        videoStatsBadge.textContent = `${data.results.length} quả · ${data.fps} FPS (${data.n_frames} frames)`;

        const rowsHtml = data.results.map(r => `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);">
                <td style="padding: 8px;">#${r.track_id}</td>
                <td style="padding: 8px; font-weight: 700;">${r.label}</td>
                <td style="padding: 8px;">${r.defect_ratio}%</td>
                <td style="padding: 8px;">${r.n_spots}</td>
                <td style="padding: 8px;">${(r.ripe_ratio * 100).toFixed(1)}%</td>
                <td style="padding: 8px; opacity: 0.85;">${r.reason || "-"}</td>
            </tr>
        `).join("");

        videoResultsSummary.innerHTML = `
            <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px;">
                <thead>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.18); opacity: 0.75;">
                        <th style="padding: 8px;">Track ID</th>
                        <th style="padding: 8px;">Phân loại</th>
                        <th style="padding: 8px;">Tỉ lệ lỗi</th>
                        <th style="padding: 8px;">Số vết</th>
                        <th style="padding: 8px;">Màu đạt</th>
                        <th style="padding: 8px;">Lý do</th>
                    </tr>
                </thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        `;
    }


    // Nút tạo & chạy video băng chuyền mẫu tự động
    demoVideoButton?.addEventListener("click", async () => {
        loading?.classList.remove("hidden");
        try {
            const res = await fetch("/api/demo-video", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fruit: fruitSelect?.value || "apple" }),
            });
            const data = await res.json();
            showVideoResult(data);
        } catch (err) {
            alert("Lỗi kết nối khi tạo video mô phỏng.");
        } finally {
            loading?.classList.add("hidden");
        }
    });
})();

// Confusion Matrix (Calib + Test)
(() => {
    const evalAllBtn = document.getElementById("evalAllButton");
    const runEvalBtn = document.getElementById("runEvalBtn");
    const evalView = document.getElementById("evalView");
    const analysisView = document.getElementById("analysisView");
    const videoView = document.getElementById("videoView");
    const emptyState = document.getElementById("emptyState");

    async function runEvaluation() {
        const fruit = elements.fruitSelect.value;
        const splits = [];
        if (document.getElementById("chkCalib")?.checked) splits.push("calib");
        if (document.getElementById("chkTest")?.checked) splits.push("test");
        if (splits.length === 0) {
            alert("Vui lòng chọn ít nhất 1 thư mục (Calib hoặc Test).");
            return;
        }

        emptyState?.classList.add("hidden");
        analysisView?.classList.add("hidden");
        videoView?.classList.add("hidden");
        evalView?.classList.remove("hidden");
        setLoading(true);

        try {
            const res = await fetch("/api/evaluate-all", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fruit, splits }),
            });
            const data = await res.json();
            if (!data.ok) {
                alert(data.error || "Lỗi khi đánh giá dataset");
                return;
            }
            renderConfusionMatrix(data);
        } catch (err) {
            alert("Không thể kết nối máy chủ để chạy đánh giá.");
        } finally {
            setLoading(false);
        }
    }

    function renderConfusionMatrix(data) {
        const m = data.metrics || {
            accuracy: data.accuracy,
            precision: 0,
            recall: 0,
            f1_score: 0,
            tp: 0, tn: 0, fp: 0, fn: 0,
        };

        document.getElementById("evalTitle").textContent =
            `Confusion Matrix — ${data.fruit.toUpperCase()} (${data.splits.join(" + ")})`;
        document.getElementById("evalAccuracyBadge").textContent =
            `F1-Score: ${m.f1_score}% · Acc: ${m.accuracy}% (${data.valid_total}/${data.total} ảnh)`;

        const preds = data.pred_labels; // ["LOAI 1", "LOAI 2", "LOAI 3"]
        const trues = data.true_labels; // ["healthy", "rotten"]

        const cellStyle = (t, p) => {
            const isMatch = (t === "healthy" && (p === "LOAI 1" || p === "LOAI 2")) ||
                            (t === "rotten" && p === "LOAI 3");
            return isMatch
                ? "background: rgba(54, 188, 148, 0.18); border: 1px solid rgba(54, 188, 148, 0.5); color: #36bc94;"
                : "background: rgba(255, 72, 72, 0.14); border: 1px solid rgba(255, 72, 72, 0.4); color: #ff6b6b;";
        };

        // 4 thẻ chỉ số đánh giá tổng hợp
        const metricsCardsHtml = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 16px;">
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px;">
                    <div style="font-size: 12px; opacity: 0.75;">Accuracy (Độ chính xác)</div>
                    <div style="font-size: 22px; font-weight: 800; color: #36bc94; margin-top: 4px;">${m.accuracy}%</div>
                    <div style="font-size: 11px; opacity: 0.6; margin-top: 2px;">TP+TN = ${m.tp + m.tn} / ${data.valid_total}</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px;">
                    <div style="font-size: 12px; opacity: 0.75;">Precision (Độ chuẩn xác)</div>
                    <div style="font-size: 22px; font-weight: 800; color: #38bdf8; margin-top: 4px;">${m.precision}%</div>
                    <div style="font-size: 11px; opacity: 0.6; margin-top: 2px;">TP=${m.tp}, FP=${m.fp}</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px;">
                    <div style="font-size: 12px; opacity: 0.75;">Recall (Độ phủ bắt lỗi)</div>
                    <div style="font-size: 22px; font-weight: 800; color: #fbbf24; margin-top: 4px;">${m.recall}%</div>
                    <div style="font-size: 11px; opacity: 0.6; margin-top: 2px;">TP=${m.tp}, FN=${m.fn}</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px;">
                    <div style="font-size: 12px; opacity: 0.75;">F1-Score (Trung bình điều hòa)</div>
                    <div style="font-size: 22px; font-weight: 800; color: #a78bfa; margin-top: 4px;">${m.f1_score}%</div>
                    <div style="font-size: 11px; opacity: 0.6; margin-top: 2px;">Lớp mục tiêu: Rotten (Loại 3)</div>
                </div>
            </div>
        `;

        let html = metricsCardsHtml + `
            <table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 14px;">
                <thead>
                    <tr>
                        <th style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.12); text-align: left;">
                            Nhãn thực tế (Raw) \\ Dự đoán (M3)
                        </th>
                        ${preds.map(p => `<th style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.12);">${p}</th>`).join("")}
                        <th style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.12);">Tổng thực tế</th>
                    </tr>
                </thead>
                <tbody>
        `;

        for (const t of trues) {
            const rowSum = preds.reduce((acc, p) => acc + data.counts[t][p], 0);
            const labelVi = t === "healthy" ? "Healthy (Tươi/Lành)" : "Rotten (Hỏng/Thối)";
            const badgeColor = t === "healthy" ? "#36bc94" : "#ff4848";

            html += `<tr>
                <td style="padding: 12px; text-align: left; font-weight: 600; color: ${badgeColor}; border-bottom: 1px solid rgba(255,255,255,0.08);">
                    ${labelVi}
                </td>`;

            for (const p of preds) {
                const count = data.counts[t][p];
                const pct = rowSum > 0 ? ((count / rowSum) * 100).toFixed(1) : "0.0";
                html += `
                    <td style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.08);">
                        <button type="button" class="cm-cell-btn" data-true="${t}" data-pred="${p}"
                            style="width: 100%; padding: 12px 8px; border-radius: 8px; cursor: pointer; font-weight: 700; font-size: 16px; ${cellStyle(t, p)}">
                            ${count} <span style="font-size: 12px; font-weight: 400; opacity: 0.85;">(${pct}%)</span>
                        </button>
                    </td>`;
            }

            html += `<td style="padding: 12px; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.08);">${rowSum}</td></tr>`;
        }

        html += `</tbody></table>`;

        html += `
            <div style="display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap;">
                <button type="button" class="dataset-button cm-filter-btn" data-filter="ALL" style="padding: 6px 12px; font-size: 12px;">Tất cả (${data.valid_total})</button>
                ${preds.map(p => {
                    const sumPred = trues.reduce((acc, t) => acc + data.counts[t][p], 0);
                    return `<button type="button" class="dataset-button cm-filter-btn" data-filter="${p}" style="padding: 6px 12px; font-size: 12px;">${p} (${sumPred})</button>`;
                }).join("")}
            </div>
        `;

        if (data.failed && data.failed.length > 0) {
            html += `<p style="margin-top: 10px; font-size: 12px; color: #ffaa48;">
                ⚠ Có ${data.failed.length} ảnh không tách được nền ở M1.
            </p>`;
        }

        const container = document.getElementById("confusionMatrixContainer");
        container.innerHTML = html;

        container.querySelectorAll(".cm-cell-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const t = btn.dataset.true;
                const p = btn.dataset.pred;
                showCellImages(`Thực tế: ${t.toUpperCase()} → Dự đoán: ${p}`, data.items[t][p]);
            });
        });

        container.querySelectorAll(".cm-filter-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const filter = btn.dataset.filter;
                let merged = [];
                for (const t of trues) {
                    for (const p of preds) {
                        if (filter === "ALL" || filter === p) {
                            merged = merged.concat(data.items[t][p]);
                        }
                    }
                }
                const title = filter === "ALL" ? "Tất cả ảnh đánh giá (Loại 1, 2, 3)" : `Tất cả ảnh dự đoán: ${filter}`;
                showCellImages(title, merged);
            });
        });

        container.querySelector('.cm-filter-btn[data-filter="ALL"]')?.click();
    }

    function showCellImages(titleText, items) {
        const detailBox = document.getElementById("matrixCellDetail");
        const grid = document.getElementById("cellImageGrid");
        detailBox.classList.remove("hidden");

        document.getElementById("cellDetailTitle").textContent = titleText;
        document.getElementById("cellDetailCount").textContent = `${items.length} ảnh (Nhấn vào ảnh để xem pipeline chi tiết)`;

        if (!items.length) {
            grid.innerHTML = `<p style="opacity: 0.6; font-size: 13px;">Không có ảnh nào rơi vào mục này.</p>`;
            return;
        }

        const predColors = {
            "LOAI 1": "#36bc94",
            "LOAI 2": "#f59e0b",
            "LOAI 3": "#ff4848",
        };

        grid.innerHTML = items.map(item => {
            const pColor = predColors[item.pred_label] || "#fff";
            const tColor = item.true_label === "healthy" ? "#36bc94" : "#ff6b6b";
            return `
            <div class="cm-img-card" data-path="${item.path}"
                style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 8px; cursor: pointer;">
                <div style="position: relative;">
                    <img src="${item.url}" alt="${item.name}" loading="lazy"
                        style="width: 100%; height: 110px; object-fit: cover; border-radius: 6px; background: #141720;">
                    <span style="position: absolute; top: 4px; left: 4px; background: rgba(0,0,0,0.75); color: #fff; font-size: 10px; padding: 2px 6px; border-radius: 4px; text-transform: uppercase;">
                        ${item.split}
                    </span>
                    <span style="position: absolute; top: 4px; right: 4px; background: rgba(0,0,0,0.82); color: ${pColor}; border: 1px solid ${pColor}; font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 4px;">
                        ${item.pred_label}
                    </span>
                </div>
                <div style="margin-top: 6px; font-size: 12px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${item.name}">
                    ${item.name}
                </div>
                <div style="font-size: 11px; color: #7dd3fc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="Ảnh gốc: ${item.original}">
                    Raw: ${item.original || "-"}
                </div>
                <div style="font-size: 11px; margin-top: 2px;">
                    Thực tế: <strong style="color: ${tColor};">${item.true_label.toUpperCase()}</strong>
                </div>
                <div style="font-size: 11px; opacity: 0.75; margin-top: 2px;">
                    Lỗi: ${item.defect_ratio}% · Chín: ${item.ripe_ratio}%
                </div>
                <div style="font-size: 10px; opacity: 0.65; margin-top: 2px;" title="${item.reason}">
                    ${item.reason}
                </div>
            </div>
        `}).join("");

        grid.querySelectorAll(".cm-img-card").forEach(card => {
            card.addEventListener("click", () => {
                elements.datasetStrip.classList.add("hidden");
                analyzeDatasetImage({ path: card.dataset.path });
            });
        });
    }

    evalAllBtn?.addEventListener("click", runEvaluation);
    runEvalBtn?.addEventListener("click", runEvaluation);
})();