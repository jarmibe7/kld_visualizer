const xMin = -4;
const xMax = 4;
const gridPoints = 380;
const epsilon = 1e-8;
const initialQ = { mu: 0.05, logSigma: 0.2 };

const state = {
  isPlaying: true,
  learningRate: 0.025,
  componentCount: 2,
  components: [
    { mean: -1.2, variance: 0.35, weight: 0.6 },
    { mean: 1.3, variance: 0.4, weight: 0.4 },
  ],
  reverse: { ...initialQ },
  forward: { ...initialQ },
};

const elements = {};

function init() {
  elements.toggleButton = document.getElementById("toggleButton");
  elements.resetButton = document.getElementById("resetButton");
  elements.learningRate = document.getElementById("learningRate");
  elements.learningRateValue = document.getElementById("learningRateValue");
  elements.componentCount = document.getElementById("componentCount");
  elements.componentControls = document.getElementById("componentControls");
  elements.reversePlot = document.getElementById("reversePlot");
  elements.forwardPlot = document.getElementById("forwardPlot");
  elements.reverseLoss = document.getElementById("reverseLoss");
  elements.forwardLoss = document.getElementById("forwardLoss");
  elements.reverseMu = document.getElementById("reverseMu");
  elements.forwardMu = document.getElementById("forwardMu");
  elements.reverseSigma = document.getElementById("reverseSigma");
  elements.forwardSigma = document.getElementById("forwardSigma");

  elements.toggleButton.addEventListener("click", () => {
    state.isPlaying = !state.isPlaying;
    updateToggleLabel();
  });

  elements.resetButton.addEventListener("click", resetOptimization);
  elements.learningRate.addEventListener("input", () => {
    state.learningRate = parseFloat(elements.learningRate.value);
    elements.learningRateValue.textContent = state.learningRate.toFixed(3);
  });
  elements.componentCount.addEventListener("change", () => {
    state.componentCount = Number(elements.componentCount.value);
    ensureComponentCount();
    renderComponentControls();
    render();
  });

  ensureComponentCount();
  renderComponentControls();
  updateToggleLabel();
  elements.learningRateValue.textContent = state.learningRate.toFixed(3);
  requestAnimationFrame(step);
}

function updateToggleLabel() {
  elements.toggleButton.textContent = state.isPlaying ? "Pause" : "Play";
}

function ensureComponentCount() {
  if (state.components.length < state.componentCount) {
    while (state.components.length < state.componentCount) {
      const index = state.components.length;
      state.components.push({
        mean: index === 0 ? -1.8 : index === 1 ? 1.2 : (index - 1) * 1.1,
        variance: 0.35,
        weight: 1 / state.componentCount,
      });
    }
  } else if (state.components.length > state.componentCount) {
    state.components = state.components.slice(0, state.componentCount);
  }
  normalizeWeights();
}

function normalizeWeights() {
  const total = state.components.reduce((sum, component) => sum + component.weight, 0);
  if (total <= 0) {
    state.components.forEach((component) => {
      component.weight = 1 / state.components.length;
    });
    return;
  }
  state.components.forEach((component) => {
    component.weight = component.weight / total;
  });
}

function renderComponentControls() {
  elements.componentControls.innerHTML = "";
  state.components.forEach((component, index) => {
    const row = document.createElement("div");
    row.className = "component-row";
    row.innerHTML = `
      <label>Component ${index + 1}</label>
      <div class="component-grid">
        <label class="mini-control">
          Mean
          <input type="range" min="-3" max="3" step="0.05" value="${component.mean.toFixed(2)}" data-field="mean" data-index="${index}" />
          <span class="value">${component.mean.toFixed(2)}</span>
        </label>
        <label class="mini-control">
          Variance
          <input type="range" min="0.1" max="1.6" step="0.01" value="${component.variance.toFixed(2)}" data-field="variance" data-index="${index}" />
          <span class="value">${component.variance.toFixed(2)}</span>
        </label>
        <label class="mini-control">
          Weight
          <input type="range" min="0.05" max="0.95" step="0.01" value="${component.weight.toFixed(2)}" data-field="weight" data-index="${index}" />
          <span class="value">${component.weight.toFixed(2)}</span>
        </label>
      </div>
    `;

    row.querySelectorAll("input").forEach((input) => {
      input.addEventListener("input", (event) => {
        const indexValue = Number(event.target.dataset.index);
        const field = event.target.dataset.field;
        const value = parseFloat(event.target.value);
        state.components[indexValue][field] = value;

        if (field === "weight") {
          normalizeWeights();
        }

        const valueLabel = event.target.parentElement.querySelector(".value");
        if (valueLabel) {
          valueLabel.textContent = value.toFixed(2);
        }

        render();
      });
    });

    elements.componentControls.appendChild(row);
  });
}

function buildGrid() {
  const xs = [];
  const step = (xMax - xMin) / (gridPoints - 1);
  for (let i = 0; i < gridPoints; i += 1) {
    xs.push(xMin + i * step);
  }
  return xs;
}

function gaussianPdf(x, mu, sigma) {
  const variance = sigma * sigma;
  const diff = x - mu;
  return (1 / (Math.sqrt(2 * Math.PI * variance))) * Math.exp(-(diff * diff) / (2 * variance));
}

function mixturePdf(xs, components) {
  return xs.map((x) => {
    let total = 0;
    components.forEach((component) => {
      total += component.weight * gaussianPdf(x, component.mean, Math.sqrt(component.variance));
    });
    return Math.max(total, epsilon);
  });
}

function klLoss(type, pValues, qValues) {
  if (type === "reverse") {
    return qValues.reduce((sum, q, index) => sum + q * Math.log(q / pValues[index]), 0);
  }
  return pValues.reduce((sum, p, index) => sum + p * Math.log(p / qValues[index]), 0);
}

function numericalGradients(type, pValues, mu, logSigma) {
  const h = 1e-3;
  const lossBase = evaluateLoss(type, pValues, mu, logSigma);
  const lossMuUp = evaluateLoss(type, pValues, mu + h, logSigma);
  const lossMuDown = evaluateLoss(type, pValues, mu - h, logSigma);
  const lossSigmaUp = evaluateLoss(type, pValues, mu, logSigma + h);
  const lossSigmaDown = evaluateLoss(type, pValues, mu, logSigma - h);

  return {
    mu: (lossMuUp - lossMuDown) / (2 * h),
    logSigma: (lossSigmaUp - lossSigmaDown) / (2 * h),
  };
}

function evaluateLoss(type, pValues, mu, logSigma) {
  const sigma = Math.exp(logSigma);
  const xs = buildGrid();
  const qValues = xs.map((x) => Math.max(gaussianPdf(x, mu, sigma), epsilon));
  return klLoss(type, pValues, qValues);
}

function updatePanel(panelName, type) {
  const xs = buildGrid();
  const pValues = mixturePdf(xs, state.components);
  const current = state[panelName];
  const grads = numericalGradients(type, pValues, current.mu, current.logSigma);
  current.mu -= state.learningRate * grads.mu;
  current.logSigma -= state.learningRate * grads.logSigma;
  current.mu = clamp(current.mu, -3.8, 3.8);
  current.logSigma = clamp(current.logSigma, -2.4, 1.3);
  const sigma = Math.exp(current.logSigma);
  const qValues = xs.map((x) => Math.max(gaussianPdf(x, current.mu, sigma), epsilon));
  const loss = klLoss(type, pValues, qValues);
  current.loss = loss;
  return { xs, pValues, qValues, loss };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function buildPath(values, color) {
  const width = 560;
  const height = 320;
  const padding = 24;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;
  const maxVal = Math.max(...values, 0.01);
  const points = values.map((value, index) => {
    const x = padding + (index / (values.length - 1)) * innerWidth;
    const y = height - padding - (value / maxVal) * innerHeight;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  return `M ${points.join(" L ")}`;
}

function buildAreaPath(values) {
  const width = 560;
  const height = 320;
  const padding = 24;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;
  const maxVal = Math.max(...values, 0.01);
  const points = values.map((value, index) => {
    const x = padding + (index / (values.length - 1)) * innerWidth;
    const y = height - padding - (value / maxVal) * innerHeight;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const lastX = width - padding;
  const firstX = padding;
  return `M ${firstX},${height - padding} L ${points[0]} L ${points.join(" L ")} L ${lastX},${height - padding} Z`;
}

function renderPlot(svg, pValues, qValues) {
  const width = 560;
  const height = 320;
  const padding = 24;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;
  const maxVal = Math.max(...pValues, ...qValues, 0.01);

  const pPoints = pValues.map((value, index) => {
    const x = padding + (index / (pValues.length - 1)) * innerWidth;
    const y = height - padding - (value / maxVal) * innerHeight;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const qPoints = qValues.map((value, index) => {
    const x = padding + (index / (qValues.length - 1)) * innerWidth;
    const y = height - padding - (value / maxVal) * innerHeight;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });

  const baselineY = height - padding;
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="16" fill="transparent"></rect>
    <line x1="${padding}" y1="${baselineY}" x2="${width - padding}" y2="${baselineY}" stroke="rgba(255,255,255,0.25)" stroke-width="1"></line>
    <path d="M ${padding},${baselineY} L ${pPoints.join(" L ")} L ${width - padding},${baselineY} Z" fill="rgba(100,181,246,0.24)" stroke="none"></path>
    <polyline points="${pPoints.join(" ")}" fill="none" stroke="#64b5f6" stroke-width="3"></polyline>
    <polyline points="${qPoints.join(" ")}" fill="none" stroke="#ff8a65" stroke-width="3"></polyline>
  `;
}

function renderStats(panelName, loss, mu, sigma) {
  const lossEl = panelName === "reverse" ? elements.reverseLoss : elements.forwardLoss;
  const muEl = panelName === "reverse" ? elements.reverseMu : elements.forwardMu;
  const sigmaEl = panelName === "reverse" ? elements.reverseSigma : elements.forwardSigma;

  lossEl.textContent = loss.toFixed(3);
  muEl.textContent = mu.toFixed(3);
  sigmaEl.textContent = sigma.toFixed(3);
}

function render() {
  const xs = buildGrid();
  const pValues = mixturePdf(xs, state.components);
  const reverseData = state.reverse;
  const forwardData = state.forward;
  const reverseSigma = Math.exp(reverseData.logSigma);
  const forwardSigma = Math.exp(forwardData.logSigma);
  const reverseQ = xs.map((x) => Math.max(gaussianPdf(x, reverseData.mu, reverseSigma), epsilon));
  const forwardQ = xs.map((x) => Math.max(gaussianPdf(x, forwardData.mu, forwardSigma), epsilon));

  renderPlot(elements.reversePlot, pValues, reverseQ);
  renderPlot(elements.forwardPlot, pValues, forwardQ);
  renderStats("reverse", reverseData.loss ?? 0, reverseData.mu, reverseSigma);
  renderStats("forward", forwardData.loss ?? 0, forwardData.mu, forwardSigma);
}

function step() {
  if (state.isPlaying) {
    const reverse = updatePanel("reverse", "reverse");
    const forward = updatePanel("forward", "forward");
    state.reverse.loss = reverse.loss;
    state.forward.loss = forward.loss;
    render();
  }
  requestAnimationFrame(step);
}

function resetOptimization() {
  state.reverse = { ...initialQ };
  state.forward = { ...initialQ };
  state.isPlaying = true;
  updateToggleLabel();
  render();
}

document.addEventListener("DOMContentLoaded", init);
