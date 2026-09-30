import { initOctonetSelector } from "./selector.js";

const DATA_URL = new URL("./filename_filesize.txt", import.meta.url);

const TEMPLATE = `
<section id="octonet-selector">
  <div class="container">
    <div class="text-center">
      <span class="sel-kicker">Subset Builder</span>
      <h2 class="text-3xl font-bold mb-4">Generate Your config.json</h2>
      <p class="sel-lead">Generate a configuration json file for partial download from our <a href="https://huggingface.co/datasets/hku-aiot/OctoNet/tree/main">Hugging Face repository</a> using our download script on <a href="https://github.com/aiot-lab/OctoNet/blob/main/partial_download_from_hf.py">GitHub</a>.</p>
    </div>

    <div class="card">
      <h2>1. Selection</h2>
      <div class="selection-body">
        <div class="tabs" role="tablist">
          <button type="button" class="tab" data-tab="node">node_id <span class="badge" id="tabBadge-node"></span></button>
          <button type="button" class="tab" data-tab="modality">modality_id <span class="badge" id="tabBadge-modality"></span></button>
          <button type="button" class="tab" data-tab="scene">scene_id <span class="badge" id="tabBadge-scene"></span></button>
          <button type="button" class="tab" data-tab="user">user_id <span class="badge" id="tabBadge-user"></span></button>
          <button type="button" class="tab active" data-tab="activity">activity_id <span class="badge" id="tabBadge-activity"></span></button>
        </div>

        <div class="panel-stack">
          <div class="panel" id="panel-node">
            <div class="section-header">
              <button class="btn btn-secondary btn-sm" data-action="toggle-all" data-group="node" data-checked="true">Select all</button>
              <button class="btn btn-secondary btn-sm" data-action="toggle-all" data-group="node" data-checked="false">Deselect all</button>
            </div>
            <div class="opt-grid" id="nodeGrid"></div>
          </div>

          <div class="panel" id="panel-modality">
            <div class="section-header">
              <button class="btn btn-secondary btn-sm" data-action="toggle-all" data-group="modality" data-checked="true">Select all</button>
              <button class="btn btn-secondary btn-sm" data-action="toggle-all" data-group="modality" data-checked="false">Deselect all</button>
            </div>
            <div class="opt-grid" id="modalityGrid"></div>
          </div>

          <div class="panel" id="panel-scene">
            <div class="section-header">
              <button class="btn btn-secondary btn-sm" data-action="toggle-all" data-group="scene" data-checked="true">Select all</button>
              <button class="btn btn-secondary btn-sm" data-action="toggle-all" data-group="scene" data-checked="false">Deselect all</button>
            </div>
            <div class="opt-grid" id="sceneGrid"></div>
          </div>

          <div class="panel" id="panel-user">
            <div class="section-header">
              <button class="btn btn-secondary btn-sm" data-action="toggle-all" data-group="user" data-checked="true">Select all</button>
              <button class="btn btn-secondary btn-sm" data-action="toggle-all" data-group="user" data-checked="false">Deselect all</button>
              <span class="badge" id="userCountBadge"></span>
            </div>
            <div class="split-panel">
              <div class="range-pane">
                <h4>Range</h4>
                <div class="range-value" id="userRangeLabel">1 – 10 <span>/ 42</span></div>
                <div class="dual-range">
                  <div class="dual-range-rail"><div class="dual-range-fill" id="userRangeFill"></div></div>
                  <input type="range" id="userStart" min="1" max="42" value="1" aria-label="User range start">
                  <input type="range" id="userEnd" min="1" max="42" value="10" aria-label="User range end">
                </div>
                <div class="range-actions">
                  <button class="btn btn-secondary btn-sm" data-action="apply-user-range" data-select="true">Select range</button>
                  <button class="btn btn-secondary btn-sm" data-action="apply-user-range" data-select="false">Deselect range</button>
                </div>
              </div>
              <div class="options-pane">
                <div class="opt-grid user" id="userGrid"></div>
              </div>
            </div>
          </div>

          <div class="panel active" id="panel-activity">
            <div class="section-header">
              <button class="btn btn-secondary btn-sm" data-action="toggle-all" data-group="activity" data-checked="true">Select all</button>
              <button class="btn btn-secondary btn-sm" data-action="toggle-all" data-group="activity" data-checked="false">Deselect all</button>
              <span class="badge" id="activityCountBadge"></span>
            </div>
            <div class="split-panel">
              <div class="range-pane">
                <h4>Index range</h4>
                <div class="range-value" id="activityRangeLabel">1 – 10 <span></span></div>
                <div class="dual-range">
                  <div class="dual-range-rail"><div class="dual-range-fill" id="activityRangeFill"></div></div>
                  <input type="range" id="activityStart" min="1" value="1" aria-label="Activity index start">
                  <input type="range" id="activityEnd" min="1" value="10" aria-label="Activity index end">
                </div>
                <div class="range-actions">
                  <button class="btn btn-secondary btn-sm" data-action="apply-activity-range" data-select="true">Select range</button>
                  <button class="btn btn-secondary btn-sm" data-action="apply-activity-range" data-select="false">Deselect range</button>
                </div>
              </div>
              <div class="options-pane">
                <div class="opt-grid activity" id="activityGrid"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="section-divider"></div>

      <div class="stats-section">
        <h2>2. Match stats &amp; generate</h2>
        <div class="stats-layout">
          <div class="stats-col">
            <h3>Totals</h3>
            <div class="stat-block">
              <div class="stat-label">In total</div>
              <div class="stat-value" id="totalRecords">—</div>
              <div class="stat-sub status" id="loadStatus">Loading filename_filesize.txt…</div>
            </div>
            <div class="stat-block">
              <div class="stat-label">Matched files</div>
              <div class="stat-value" id="matchCount">—</div>
            </div>
            <div class="stat-block">
              <div class="stat-label">Total size</div>
              <div class="stat-value" id="matchSize">—</div>
              <div class="stat-sub" id="matchSizeBytes"></div>
            </div>
          </div>
          <div class="stats-col">
            <h3>Distribution</h3>
            <div class="charts" id="charts">
              <div class="chart-card">
                <h4>Nodes</h4>
                <svg class="chart-svg" id="pie-node" viewBox="0 0 100 100"></svg>
                <div class="legend" id="legend-node"></div>
              </div>
              <div class="chart-card">
                <h4>Modalities</h4>
                <svg class="chart-svg" id="pie-modality" viewBox="0 0 100 100"></svg>
                <div class="legend" id="legend-modality"></div>
              </div>
              <div class="chart-card">
                <h4>Scenes</h4>
                <svg class="chart-svg" id="pie-scene" viewBox="0 0 100 100"></svg>
                <div class="legend" id="legend-scene"></div>
              </div>
              <div class="chart-card">
                <h4>Users</h4>
                <svg class="chart-svg" id="pie-user" viewBox="0 0 100 100"></svg>
                <div class="legend" id="legend-user"></div>
              </div>
              <div class="chart-card">
                <h4>Activities</h4>
                <svg class="chart-svg" id="pie-activity" viewBox="0 0 100 100"></svg>
                <div class="legend" id="legend-activity"></div>
              </div>
            </div>
          </div>
          <div class="stats-col config-col">
            <h3>config.json</h3>
            <button class="btn btn-primary" data-action="download">Download config.json</button>
            <pre id="preview">Select filters and load the data file</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div id="pieTooltip" class="pie-tooltip"></div>
</section>
`;

function ensureHost() {
  let host = document.getElementById("octonet-selector-host");
  if (host) return host;

  const modalities = document.getElementById("modalities");
  if (!modalities) return null;
  const citation = [...modalities.querySelectorAll("h2")].find((h) => h.textContent.trim() === "Citation");
  if (!citation) return null;
  const citationBlock = citation.closest(".max-w-3xl") || citation.parentElement;
  host = document.createElement("div");
  host.id = "octonet-selector-host";
  host.className = "mb-32";
  citationBlock.parentElement.insertBefore(host, citationBlock);
  return host;
}

function addNavLink() {
  const nav = document.querySelector("nav .hidden.md\\:flex, nav div.hidden");
  const links = document.querySelectorAll('nav a[href="#modalities"]');
  if (!links.length) return;
  const modalitiesLink = links[0];
  if (modalitiesLink.parentElement.querySelector('a[href="#octonet-selector"]')) return;
  const a = document.createElement("a");
  a.href = "#octonet-selector";
  a.className = modalitiesLink.className;
  a.textContent = "Selector";
  modalitiesLink.insertAdjacentElement("afterend", a);
}

function mount() {
  const host = ensureHost();
  if (!host) return false;
  addNavLink();
  if (!host.querySelector("#octonet-selector")) {
    host.innerHTML = TEMPLATE;
    initOctonetSelector(host.querySelector("#octonet-selector"), DATA_URL);
  }
  return true;
}

function start() {
  mount();
  const obs = new MutationObserver(() => {
    mount();
  });
  obs.observe(document.getElementById("root") || document.body, { childList: true, subtree: true });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", start);
} else {
  start();
}
