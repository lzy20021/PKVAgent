(function () {
  const data = window.INTERFACE_DISPLAY_DATA;
  const svgNS = "http://www.w3.org/2000/svg";
  let currentView = "overlay";

  function createSvg(tag, attrs = {}) {
    const el = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    return el;
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function symbolElement(point, opacity, sizeScale = 1) {
    const size = point.symbolSize * sizeScale;
    const strokeWidth = Math.max(1.15, size * 0.14);
    const attrs = { fill: point.symbolColor, stroke: point.symbolStroke, "stroke-width": strokeWidth, opacity };
    if (point.symbol === "square") return createSvg("rect", { ...attrs, x: point.px - size * 0.56, y: point.py - size * 0.56, width: size * 1.12, height: size * 1.12, rx: size * 0.18 });
    if (point.symbol === "triangle") return createSvg("polygon", { ...attrs, points: [[point.px, point.py - size * 0.68], [point.px - size * 0.68, point.py + size * 0.58], [point.px + size * 0.68, point.py + size * 0.58]].map((p) => p.join(",")).join(" ") });
    if (point.symbol === "diamond") return createSvg("polygon", { ...attrs, points: [[point.px, point.py - size * 0.7], [point.px + size * 0.7, point.py], [point.px, point.py + size * 0.7], [point.px - size * 0.7, point.py]].map((p) => p.join(",")).join(" ") });
    if (point.symbol === "cross") return createSvg("polygon", { ...attrs, points: [[point.px - size * 0.22, point.py - size * 0.62], [point.px + size * 0.22, point.py - size * 0.62], [point.px + size * 0.22, point.py - size * 0.22], [point.px + size * 0.62, point.py - size * 0.22], [point.px + size * 0.62, point.py + size * 0.22], [point.px + size * 0.22, point.py + size * 0.22], [point.px + size * 0.22, point.py + size * 0.62], [point.px - size * 0.22, point.py + size * 0.62], [point.px - size * 0.22, point.py + size * 0.22], [point.px - size * 0.62, point.py + size * 0.22], [point.px - size * 0.62, point.py - size * 0.22], [point.px - size * 0.22, point.py - size * 0.22]].map((p) => p.join(",")).join(" ") });
    if (point.symbol === "star") {
      const pts = [];
      for (let i = 0; i < 10; i += 1) {
        const angle = ((-90 + i * 36) * Math.PI) / 180;
        const radius = i % 2 === 0 ? size * 0.76 : size * 0.34;
        pts.push([point.px + Math.cos(angle) * radius, point.py + Math.sin(angle) * radius]);
      }
      return createSvg("polygon", { ...attrs, points: pts.map((p) => p.join(",")).join(" ") });
    }
    if (point.symbol === "hex") {
      const pts = [];
      for (let i = 0; i < 6; i += 1) {
        const angle = ((30 + i * 60) * Math.PI) / 180;
        pts.push([point.px + Math.cos(angle) * size * 0.64, point.py + Math.sin(angle) * size * 0.64]);
      }
      return createSvg("polygon", { ...attrs, points: pts.map((p) => p.join(",")).join(" ") });
    }
    return createSvg("circle", { ...attrs, cx: point.px, cy: point.py, r: size * 0.5 });
  }

  function project(point) {
    return { ...point, px: 70 + point.x * 720, py: 72 + point.y * 420 };
  }

  function addBadge(svg, x, y, text, emphasized = false) {
    const width = Math.max(126, Math.min(220, 16 + text.length * 6.8));
    svg.appendChild(createSvg("rect", {
      x,
      y,
      width,
      height: 26,
      rx: 13,
      fill: emphasized ? "#E7F0FA" : "#F3F7FC",
      stroke: emphasized ? "#BED1E4" : "#DAE4EE",
    }));
    const label = createSvg("text", {
      x: x + 10,
      y: y + 17,
      fill: emphasized ? "#31567F" : "#5C6F85",
      "font-size": "11",
      "font-family": "Segoe UI, Microsoft YaHei, sans-serif",
    });
    label.textContent = text;
    svg.appendChild(label);
  }

  function renderMap() {
    const summary = data.primarySample.summary;
    const structureItem = data.selectedSamples.find((item) => item.city === data.primarySample.city);
    const svg = document.getElementById("mapSvg");
    svg.innerHTML = "";

    svg.appendChild(createSvg("rect", { x: 18, y: 16, width: 824, height: 528, rx: 20, fill: "#FBFDFF", stroke: "#D8E3EE" }));
    svg.appendChild(createSvg("rect", { x: 58, y: 62, width: 744, height: 436, rx: 18, fill: "#FCFEFF", stroke: "#D8E3EE" }));
    svg.appendChild(createSvg("rect", { x: 66, y: 70, width: 728, height: 420, rx: 14, fill: "none", stroke: "#E4EDF6", "stroke-dasharray": "4 4" }));
    svg.appendChild(createSvg("rect", { x: 186, y: 132, width: 470, height: 262, rx: 12, fill: "none", stroke: "#C8D8E7", "stroke-width": "1.4" }));

    addBadge(svg, 74, 78, `Sample city: ${data.primarySample.city}`, true);
    addBadge(svg, 226, 78, `Structure type: ${structureItem ? structureItem.structure_display : data.primarySample.structure}`);
    addBadge(svg, 74, 110, `Input points: ${summary.beforeCount}`);
    addBadge(svg, 226, 110, `Output symbols: ${summary.afterCount}`);
    addBadge(svg, 378, 110, `Retained categories: ${Object.keys(summary.categoryCountAfter).length}`);

    const modeLabel = createSvg("text", { x: 688, y: 88, fill: "#4E6680", "font-size": "11", "font-family": "Segoe UI, Microsoft YaHei, sans-serif" });
    modeLabel.textContent = `View: ${currentView.toUpperCase()}`;
    svg.appendChild(modeLabel);

    const explain = createSvg("text", { x: 74, y: 468, fill: "#5A7290", "font-size": "11.5", "font-family": "Segoe UI, Microsoft YaHei, sans-serif" });
    explain.textContent = "light symbols = input points; solid symbols = generalized output";
    svg.appendChild(explain);

    const before = data.primarySample.before.map(project);
    const after = data.primarySample.after.map(project);

    function updateSummary() {
      const sample = data.primarySample;
      const sum = sample.summary;
      sum.afterCount = sample.after.length;
      sum.reductionRate = ((1 - sum.afterCount / sum.beforeCount) * 100).toFixed(1);
      sum.categoryCountAfter = {};
      sample.after.forEach(p => {
        sum.categoryCountAfter[p.category] = (sum.categoryCountAfter[p.category] || 0) + 1;
      });
    }

    const draw = (points, opacity, scale, isBeforeData) => {
      points.forEach((point) => {
        const el = symbolElement(point, opacity, scale);
        el.style.cursor = "pointer";
        
        if (isBeforeData) {
          el.addEventListener("dblclick", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const clickedId = point.id;
            const orig = data.primarySample.before.find(p => p.id === clickedId);
            if (orig && !data.primarySample.after.some(p => p.id === clickedId)) {
              data.primarySample.after.push(JSON.parse(JSON.stringify(orig)));
              updateSummary();
              renderMap();
              populateNarrative();
            }
          });
        } else {
          el.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const clickedId = point.id;
            data.primarySample.after = data.primarySample.after.filter(p => p.id !== clickedId);
            updateSummary();
            renderMap();
            populateNarrative();
          });
        }
        
        svg.appendChild(el);
      });
    };

    if (currentView === "before") draw(before, 0.48, 0.76, true);
    else if (currentView === "after") draw(after, 0.98, 1.03, false);
    else {
      draw(before, 0.22, 0.72, true);
      draw(after, 0.98, 1.05, false);
    }

    let x = 78;
    data.symbolSystem.forEach((item) => {
      const node = { symbol: item.shape, symbolColor: item.color, symbolStroke: item.stroke, symbolSize: 8.2, px: x, py: 513 };
      svg.appendChild(symbolElement(node, 1, 1));
      const label = createSvg("text", { x: x + 14, y: 517, fill: "#5E7188", "font-size": "11.5", "font-family": "Segoe UI, Microsoft YaHei, sans-serif" });
      label.textContent = item.category;
      svg.appendChild(label);
      x += 122;
    });
  }

  function renderLegend() {
    const wrap = document.getElementById("legendList");
    wrap.innerHTML = "";
    data.symbolSystem.forEach((item) => {
      const row = document.createElement("div");
      row.className = "legend-item";
      const chip = document.createElement("div");
      chip.className = "symbol-chip";
      const preview = document.createElementNS(svgNS, "svg");
      preview.setAttribute("viewBox", "0 0 28 28");
      preview.setAttribute("width", "28");
      preview.setAttribute("height", "28");
      preview.appendChild(symbolElement({ symbol: item.shape, symbolColor: item.color, symbolStroke: item.stroke, symbolSize: 7, px: 14, py: 14 }, 1, 1));
      chip.appendChild(preview);
      const text = document.createElement("div");
      text.innerHTML = `<strong>${item.category}</strong><span>${item.shape} symbol</span>`;
      row.appendChild(chip);
      row.appendChild(text);
      wrap.appendChild(row);
    });
  }

  function renderSamples() {
    const wrap = document.getElementById("sampleList");
    wrap.innerHTML = "";
    data.selectedSamples.forEach((item) => {
      const node = document.createElement("div");
      node.className = `sample-item${item.city === data.primarySample.city ? " active" : ""}`;
      node.innerHTML = `<strong>${item.city}</strong><span class="sample-meta">${item.structure_display}</span><br><span class="sample-size">${item.point_count} points</span>`;
      wrap.appendChild(node);
    });
  }

  function renderCategoryBars() {
    const summary = data.primarySample.summary;
    const beforeCounts = summary.categoryCountBefore;
    const afterCounts = summary.categoryCountAfter;
    const maxCount = Math.max(...Object.values(beforeCounts), ...Object.values(afterCounts), 1);
    const wrap = document.getElementById("categoryBars");
    wrap.innerHTML = "";

    data.symbolSystem.forEach((item) => {
      const before = beforeCounts[item.category] || 0;
      const after = afterCounts[item.category] || 0;
      const row = document.createElement("div");
      row.className = "category-row";
      row.innerHTML = `
        <span class="name">${item.category}</span>
        <div class="bar-track"><div class="bar-fill input" style="width:${(before / maxCount) * 100}%"></div></div>
        <div class="bar-track"><div class="bar-fill output" style="width:${(after / maxCount) * 100}%"></div></div>
        <span class="ratio">${after}/${before}</span>
      `;
      wrap.appendChild(row);
    });
  }

  function populateNarrative() {
    const summary = data.primarySample.summary;
    const sampleMeta = data.selectedSamples.find((item) => item.city === data.primarySample.city);
    setText("primaryCity", data.primarySample.city);
    setText("primaryStructure", sampleMeta ? sampleMeta.structure_display : data.primarySample.structure);
    setText("primaryFlow", `${summary.beforeCount} → ${summary.afterCount}`);
    setText("metricReduction", `${summary.reductionRate}%`);
    setText("mapDescription", `${data.primarySample.city} focus extent with structure-aware POI symbols. Before / After / Overlay highlights semantic-preserving generalization.`);
    setText("reductionBadge", `${summary.reductionRate}% reduction`);
    setText("footerInputCount", summary.beforeCount);
    setText("footerOutputCount", summary.afterCount);
    setText("footerCategories", Object.keys(summary.categoryCountAfter).length);
    setText("dataInputText", `${data.primarySample.city} focus subset with ${summary.beforeCount} input POIs across ${Object.keys(summary.categoryCountBefore).length} retained categories.`);
    setText("requestText", "Reduce overlap while preserving category semantics and stable symbol readability for publication-ready cartographic comparison.");
    setText("understandingText", `${sampleMeta ? sampleMeta.structure_display : data.primarySample.structure} indicates dense single-core clustering, requiring both geometric thinning and symbol-level retention.`);
    setText("strategyText", "Apply category-priority selection, minimum-spacing constraints, and consistent 6-symbol grammar (shape, size, stroke, opacity).");
    setText("outputText", `${summary.afterCount} symbolized outputs keep all key categories with clear before/after interpretability for explainable workflows.`);
    const qWrap = document.getElementById("questionList");
    qWrap.innerHTML = "";
    [
      "What are the input points and structure context?",
      "How does the agent decide thinning with symbol constraints?",
      "What semantic categories are preserved after generalization?",
    ].forEach((question) => {
      const bubble = document.createElement("div");
      bubble.className = "question-bubble";
      bubble.textContent = question;
      qWrap.appendChild(bubble);
    });
    renderCategoryBars();
  }

  document.querySelectorAll(".segment-btn").forEach((button) => {
    button.addEventListener("click", () => {
      currentView = button.dataset.view;
      document.querySelectorAll(".segment-btn").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      renderMap();
    });
  });
  populateNarrative();
  renderSamples();
  renderLegend();
  renderMap();
})();
