const NODES = ["1", "2", "3", "4", "5", "mocap"];
const MODALITIES = ["acoustic", "depthcam", "heartrate", "imu", "ira", "mmwave", "mocapcsv", "seekthermal", "ToF", "uwb", "wifi"];
const SCENES = ["1", "2", "3"];
const USERS = Array.from({ length: 42 }, (_, i) => String(i + 1));
const ACTIVITIES = [
  "airdrum", "answerphone", "bicepcurl", "blownose", "bow", "bowling", "boxing", "brushhair", "brushteeth",
  "clap", "conversation", "cough", "dance", "dodge", "drawcircleclockwise", "drawcirclecounterclockwise",
  "drawtriangle", "drawzigzag", "drink", "eat", "falldown", "freestyle", "gym", "handshake", "handwave", "hug",
  "jog", "jump", "jumpingjack", "jumprope", "kicksomeone", "legraise", "liftupahand", "lunge", "makeoksign",
  "makevictorysign", "moppingfloor", "pickup", "playphone", "pullhandin", "punchsomeone", "pushhandaway",
  "pushsomeone", "pushup", "shakehead", "sit", "sleep", "slide", "sneeze", "spreadandpinch", "squat", "stagger",
  "stopsign", "strechoneself", "stretchoneself", "sweep", "tap", "thumbdown", "thumbup", "touchface", "turn",
  "type", "walk", "wipeface", "yawn"
];
const ALL = { node: NODES, modality: MODALITIES, scene: SCENES, user: USERS, activity: ACTIVITIES };
const PALETTE = ["#ffffff", "#a3a3a3", "#737373", "#e5e5e5", "#f97316", "#eab308", "#22c55e", "#fb923c", "#fb7185", "#d4d4d4", "#525252", "#f59e0b"];
const PIE_KIND = { node: "Node", modality: "Modality", scene: "Scene", user: "User", activity: "Activity" };

export function initOctonetSelector(root, dataUrl) {
  if (!root || root.dataset.octonetReady === "1") return;
  root.dataset.octonetReady = "1";

  const $ = (sel) => root.querySelector(sel);
  const $$ = (sel) => [...root.querySelectorAll(sel)];

  let fileRecords = [];
  const selected = {
    node: new Set(NODES),
    modality: new Set(MODALITIES),
    scene: new Set(SCENES),
    user: new Set(USERS),
    activity: new Set(ACTIVITIES)
  };

  function switchTab(name) {
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
  }

  function buildButtons(containerId, items, prefix, showIndex) {
    const grid = root.querySelector("#" + containerId);
    grid.innerHTML = items.map((v, i) => `
      <button type="button" class="opt-btn selected" data-group="${prefix}" data-value="${v}">
        ${showIndex ? (i + 1) + ". " : ""}${v}
      </button>
    `).join("");
  }

  function toggleOption(btn) {
    const group = btn.dataset.group;
    const value = btn.dataset.value;
    if (selected[group].has(value)) {
      selected[group].delete(value);
      btn.classList.remove("selected");
    } else {
      selected[group].add(value);
      btn.classList.add("selected");
    }
    onSelectionChange();
  }

  function syncButtons(prefix) {
    $$(`.opt-btn[data-group="${prefix}"]`).forEach((btn) => {
      btn.classList.toggle("selected", selected[prefix].has(btn.dataset.value));
    });
  }

  function toggleAll(prefix, checked) {
    if (checked) ALL[prefix].forEach((v) => selected[prefix].add(v));
    else selected[prefix].clear();
    syncButtons(prefix);
    onSelectionChange();
  }

  function readRange(kind) {
    const startEl = root.querySelector("#" + kind + "Start");
    const endEl = root.querySelector("#" + kind + "End");
    let start = parseInt(startEl.value, 10);
    let end = parseInt(endEl.value, 10);
    if (start > end) {
      if (document.activeElement === startEl) startEl.value = end;
      else endEl.value = start;
      start = parseInt(startEl.value, 10);
      end = parseInt(endEl.value, 10);
    }
    return { start, end, min: parseInt(startEl.min, 10), max: parseInt(startEl.max, 10) };
  }

  function updateRangeUI(kind) {
    const { start, end, min, max } = readRange(kind);
    const span = max - min;
    const left = ((start - min) / span) * 100;
    const right = ((end - min) / span) * 100;
    root.querySelector("#" + kind + "RangeFill").style.left = left + "%";
    root.querySelector("#" + kind + "RangeFill").style.width = (right - left) + "%";
    const extra = kind === "user" ? " / 42" : " / " + ACTIVITIES.length;
    root.querySelector("#" + kind + "RangeLabel").innerHTML = `${start} – ${end} <span>${extra}</span>`;
  }

  function onRangeInput(kind) {
    updateRangeUI(kind);
    const startEl = root.querySelector("#" + kind + "Start");
    const endEl = root.querySelector("#" + kind + "End");
    const { min, max, start } = readRange(kind);
    const mid = (min + max) / 2;
    startEl.style.zIndex = start > mid ? "5" : "4";
    endEl.style.zIndex = start > mid ? "3" : "5";
  }

  function applyUserRange(select) {
    const { start, end } = readRange("user");
    if (select && selected.user.size === USERS.length) selected.user.clear();
    USERS.forEach((v) => {
      const n = parseInt(v, 10);
      if (n >= start && n <= end) {
        if (select) selected.user.add(v);
        else selected.user.delete(v);
      }
    });
    syncButtons("user");
    onSelectionChange();
  }

  function applyActivityRange(select) {
    const { start, end } = readRange("activity");
    if (select && selected.activity.size === ACTIVITIES.length) selected.activity.clear();
    ACTIVITIES.forEach((v, idx) => {
      const num = idx + 1;
      if (num >= start && num <= end) {
        if (select) selected.activity.add(v);
        else selected.activity.delete(v);
      }
    });
    syncButtons("activity");
    onSelectionChange();
  }

  function getSelected(prefix, allValues) {
    const set = selected[prefix];
    if (set.size === 0) return [];
    if (set.size === allValues.length) return null;
    return allValues.filter((v) => set.has(v));
  }

  function buildConfig() {
    return {
      node_id: getSelected("node", NODES),
      modality_id: getSelected("modality", MODALITIES),
      scene_id: getSelected("scene", SCENES),
      user_id: getSelected("user", USERS),
      activity_id: getSelected("activity", ACTIVITIES)
    };
  }

  function matchesRecord(rec, cfg) {
    if (cfg.node_id !== null && !cfg.node_id.includes(rec.node)) return false;
    if (cfg.modality_id !== null && !cfg.modality_id.includes(rec.modality)) return false;
    if (cfg.scene_id !== null && !cfg.scene_id.includes(rec.scene)) return false;
    if (cfg.user_id !== null && !cfg.user_id.includes(rec.user)) return false;
    if (cfg.activity_id !== null && !cfg.activity_id.includes(rec.activity)) return false;
    return true;
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(2) + " " + units[i];
  }

  function updateCountBadges() {
    for (const key of Object.keys(ALL)) {
      const n = selected[key].size;
      const el = root.querySelector("#tabBadge-" + key);
      if (el) el.textContent = `${n}/${ALL[key].length}`;
    }
    root.querySelector("#userCountBadge").textContent = `${selected.user.size}/${USERS.length}`;
    root.querySelector("#activityCountBadge").textContent = `${selected.activity.size}/${ACTIVITIES.length}`;
  }

  function polar(cx, cy, r, a) {
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  }

  function donutSlice(cx, cy, rOut, rIn, a0, a1) {
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const [x0, y0] = polar(cx, cy, rOut, a0);
    const [x1, y1] = polar(cx, cy, rOut, a1);
    const [x2, y2] = polar(cx, cy, rIn, a1);
    const [x3, y3] = polar(cx, cy, rIn, a0);
    return `M ${x0} ${y0} A ${rOut} ${rOut} 0 ${large} 1 ${x1} ${y1} L ${x2} ${y2} A ${rIn} ${rIn} 0 ${large} 0 ${x3} ${y3} Z`;
  }

  function entriesFromCounts(countMap, maxSlices) {
    const items = [...countMap.entries()].map(([label, value]) => ({ label, value }))
      .sort((a, b) => b.value - a.value);
    if (items.length <= maxSlices) return items;
    const head = items.slice(0, maxSlices - 1);
    const rest = items.slice(maxSlices - 1).reduce((s, x) => s + x.value, 0);
    head.push({ label: "Other", value: rest });
    return head;
  }

  function pieTooltipEl() {
    return root.querySelector("#pieTooltip");
  }

  function showPieTooltip(event, kind, label, value) {
    const tip = pieTooltipEl();
    const noun = PIE_KIND[kind] || kind;
    tip.textContent = `${noun}: ${label}\n${value.toLocaleString()} files`;
    tip.style.display = "block";
    movePieTooltip(event);
  }

  function movePieTooltip(event) {
    const tip = pieTooltipEl();
    const x = event.clientX + 12;
    const y = event.clientY + 12;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  function hidePieTooltip() {
    pieTooltipEl().style.display = "none";
  }

  function bindSliceTooltip(el, kind, label, value) {
    el.addEventListener("mouseenter", (e) => showPieTooltip(e, kind, label, value));
    el.addEventListener("mousemove", movePieTooltip);
    el.addEventListener("mouseleave", hidePieTooltip);
  }

  function renderPie(kind, countMap) {
    const svg = root.querySelector("#pie-" + kind);
    const legend = root.querySelector("#legend-" + kind);
    const total = [...countMap.values()].reduce((s, v) => s + v, 0);
    svg.innerHTML = "";
    hidePieTooltip();
    if (!total) {
      legend.innerHTML = `<div class="empty-chart">No matches</div>`;
      svg.innerHTML = `<circle cx="50" cy="50" r="34" fill="none" stroke="#262626" stroke-width="12"/>`;
      return;
    }
    const slices = entriesFromCounts(countMap, 10);
    const cx = 50, cy = 50, rOut = 40, rIn = 24;
    let angle = -Math.PI / 2;
    if (slices.length === 1) {
      const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      ring.setAttribute("class", "slice");
      ring.setAttribute("cx", cx);
      ring.setAttribute("cy", cy);
      ring.setAttribute("r", String((rOut + rIn) / 2));
      ring.setAttribute("fill", "none");
      ring.setAttribute("stroke", PALETTE[0]);
      ring.setAttribute("stroke-width", String(rOut - rIn));
      svg.appendChild(ring);
      bindSliceTooltip(ring, kind, slices[0].label, slices[0].value);
    } else {
      slices.forEach((s, i) => {
        const sweep = (s.value / total) * 2 * Math.PI;
        const a1 = angle + sweep;
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", donutSlice(cx, cy, rOut, rIn, angle, a1));
        path.setAttribute("fill", PALETTE[i % PALETTE.length]);
        svg.appendChild(path);
        bindSliceTooltip(path, kind, s.label, s.value);
        angle = a1;
      });
    }
    legend.innerHTML = slices.map((s, i) => `
      <div class="legend-row">
        <i class="swatch" style="background:${PALETTE[i % PALETTE.length]}"></i>
        <span class="legend-text">
          <span class="legend-name">${s.label}</span>
          <span class="legend-leader"></span>
        </span>
        <span class="legend-count">${s.value.toLocaleString()}</span>
      </div>
    `).join("");
    legend.querySelectorAll(".legend-row").forEach((row, i) => {
      bindSliceTooltip(row, kind, slices[i].label, slices[i].value);
    });
  }

  function updateCharts(matched) {
    const keys = [
      ["node", "node"],
      ["modality", "modality"],
      ["scene", "scene"],
      ["user", "user"],
      ["activity", "activity"]
    ];
    for (const [kind, recKey] of keys) {
      const map = new Map();
      for (const rec of matched) {
        map.set(rec[recKey], (map.get(rec[recKey]) || 0) + 1);
      }
      renderPie(kind, map);
    }
  }

  function updatePreview(cfg) {
    if (!cfg) cfg = buildConfig();
    root.querySelector("#preview").textContent = JSON.stringify(cfg, null, 4);
  }

  function onSelectionChange() {
    updateCountBadges();
    const cfg = buildConfig();

    if (fileRecords.length === 0) {
      root.querySelector("#matchCount").textContent = "—";
      root.querySelector("#matchSize").textContent = "—";
      root.querySelector("#matchSizeBytes").textContent = "";
      updateCharts([]);
      updatePreview(cfg);
      return;
    }

    const matched = [];
    let totalSize = 0;
    for (const rec of fileRecords) {
      if (matchesRecord(rec, cfg)) {
        matched.push(rec);
        totalSize += rec.size;
      }
    }
    root.querySelector("#matchCount").textContent = matched.length.toLocaleString();
    root.querySelector("#matchSize").textContent = formatBytes(totalSize);
    root.querySelector("#matchSizeBytes").textContent = totalSize.toLocaleString() + " bytes";
    updateCharts(matched);
    updatePreview(cfg);
  }

  function generateAndDownload() {
    const cfg = buildConfig();
    const content = JSON.stringify(cfg, null, 4);
    const blob = new Blob([content], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "config.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function parseFilename(name) {
    const base = name.replace(/\.zip$/i, "");
    const parts = base.split("_");
    if (parts.length < 7) return null;
    return {
      node: parts[0],
      modality: parts[1],
      scene: parts[2],
      user: parts[3],
      activity: parts[4],
      trial: parts[5],
      timestamp: parts[6]
    };
  }

  async function autoLoadFile() {
    const status = root.querySelector("#loadStatus");
    try {
      const res = await fetch(dataUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const lines = text.split(/\r?\n/).filter((l) => l.trim());
      fileRecords = [];
      let bad = 0;
      for (const line of lines) {
        const spaceIdx = line.indexOf(" ");
        if (spaceIdx === -1) { bad++; continue; }
        const size = parseInt(line.slice(0, spaceIdx), 10);
        const fname = line.slice(spaceIdx + 1).trim();
        if (isNaN(size) || !fname) { bad++; continue; }
        const parsed = parseFilename(fname);
        if (!parsed) { bad++; continue; }
        fileRecords.push({ size, ...parsed, raw: fname });
      }
      root.querySelector("#totalRecords").textContent = fileRecords.length.toLocaleString();
      status.textContent = bad ? `records (skipped ${bad} lines)` : "records";
      status.className = "stat-sub status ok";
      onSelectionChange();
    } catch (err) {
      root.querySelector("#totalRecords").textContent = "—";
      status.textContent = "Failed to load: " + err.message;
      status.className = "stat-sub status err";
    }
  }

  root.addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (tab && root.contains(tab)) {
      switchTab(tab.dataset.tab);
      return;
    }
    const opt = e.target.closest(".opt-btn");
    if (opt && root.contains(opt)) {
      toggleOption(opt);
      return;
    }
    const actionBtn = e.target.closest("[data-action]");
    if (!actionBtn || !root.contains(actionBtn)) return;
    const { action, group } = actionBtn.dataset;
    if (action === "toggle-all") toggleAll(group, actionBtn.dataset.checked === "true");
    if (action === "apply-user-range") applyUserRange(actionBtn.dataset.select === "true");
    if (action === "apply-activity-range") applyActivityRange(actionBtn.dataset.select === "true");
    if (action === "download") generateAndDownload();
  });

  root.addEventListener("input", (e) => {
    if (e.target.id === "userStart" || e.target.id === "userEnd") onRangeInput("user");
    if (e.target.id === "activityStart" || e.target.id === "activityEnd") onRangeInput("activity");
  });

  buildButtons("nodeGrid", NODES, "node", false);
  buildButtons("modalityGrid", MODALITIES, "modality", false);
  buildButtons("sceneGrid", SCENES, "scene", false);
  buildButtons("userGrid", USERS, "user", false);
  buildButtons("activityGrid", ACTIVITIES, "activity", true);

  const actMax = ACTIVITIES.length;
  root.querySelector("#activityStart").max = actMax;
  root.querySelector("#activityEnd").max = actMax;
  root.querySelector("#activityEnd").value = Math.min(10, actMax);
  updateRangeUI("user");
  updateRangeUI("activity");
  updateCountBadges();
  updatePreview();
  updateCharts([]);
  autoLoadFile();
}
