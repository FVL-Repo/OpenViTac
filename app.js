"use strict";

const config = window.OPENVITAC_CONFIG || { resources: [], demos: [] };
const groups = {
  all: "All tasks",
  PP: "Property perception",
  FA: "Fragility-aware",
  CR: "Contact-rich",
  PR: "Precision",
};
const tasks = [
  {
    id: "WC",
    name: "Weight classification",
    group: "PP",
    image: "weight_classify.png",
    real: false,
    description:
      "Infer an object’s weight from physical interaction and place it according to its weight class.",
  },
  {
    id: "HC",
    name: "Hardness classification",
    group: "PP",
    image: "hardness_classify.png",
    real: false,
    description:
      "Distinguish visually ambiguous objects by hardness and make a property-conditioned placement.",
  },
  {
    id: "RC",
    name: "Roughness classification",
    group: "PP",
    image: "roughness_classify.png",
    real: true,
    description:
      "Feel the surface texture to classify an object and place it in the corresponding location.",
  },
  {
    id: "RR",
    name: "Roughness-guided regrasp",
    group: "PP",
    image: "roughness_regrasp.png",
    real: false,
    description:
      "Use tactile surface information to guide regrasping and property-aware manipulation.",
  },
  {
    id: "ECS",
    name: "Empty-can selection",
    group: "PP",
    image: "empty_can_select.png",
    real: true,
    description:
      "Distinguish an empty can from filled cans through physical interaction, then select and move it.",
  },
  {
    id: "GC",
    name: "Grasp chip",
    group: "FA",
    image: "grasp_chips.png",
    real: true,
    description:
      "Gently grasp and place a fragile chip while regulating contact to avoid damage.",
  },
  {
    id: "GA",
    name: "Gear assembly",
    group: "CR",
    image: "gear_assembly.png",
    real: true,
    description:
      "Assemble a gear by matching the teeth and maintaining effective engagement throughout the interaction.",
  },
  {
    id: "PD",
    name: "Pull drawer",
    group: "CR",
    image: "pull_drawer.png",
    real: true,
    description:
      "Establish and maintain a stable grasp while pulling an articulated drawer open.",
  },
  {
    id: "In-USB",
    name: "Insert USB",
    group: "PR",
    image: "insert_usb.png",
    extra: "insert_usb_sensors.png",
    extraLabel: "View the multi-sensor setup",
    real: true,
    description:
      "Use contact feedback to align the USB connector with the instructed slot and complete the insertion.",
  },
  {
    id: "In-B v1",
    name: "Insert block v1",
    group: "PR",
    image: "insert_block_v1.png",
    real: true,
    description:
      "Precisely align a held block with a target opening and insert it under tight geometric tolerances.",
  },
  {
    id: "In-B v2",
    name: "Insert block v2",
    group: "PR",
    image: "insert_block_v2_p1.png",
    extra: "insert_block_v2_p2.png",
    extraLabel: "View additional task variants",
    real: true,
    description:
      "Select the instructed object before performing a contact-guided, precise block insertion.",
  },
];

// Table values transcribed from sec/4_experiments.tex. Last entry is the reported average.
const policies = [
  {
    name: "OpenVTLA",
    family: "VTLA · Ours",
    sim: [89, 99, 100, 90, 73, 69, 91, 99, 27, 11, 8, 68.7],
    real: [95, 85, 95, 50, 60, 20, 15, 100 / 6, 54.6],
  },
  {
    name: "FTP-1",
    family: "VTLA",
    sim: [61, 100, 100, 78, 50, 60, 83, 94, 27, 12, 8.3, 61.2],
    real: [90, 70, 95, 45, 70, 15, 20, 10, 51.9],
  },
  {
    name: "InternVLA-A1.5",
    family: "VLA",
    sim: [53, 76, 65, 77, 47, 53, 79, 98, 3, 6, 9.3, 51.5],
    real: [50, 50, 85, 45, 65, 15, 15, 20, 43.1],
  },
  {
    name: "π₀.₅",
    family: "VLA",
    sim: [53, 60, 64, 52, 48, 39, 87, 99, 14, 9, 9.3, 48.6],
    real: [50, 50, 80, 35, 60, 15, 25, 40 / 3, 41],
  },
  {
    name: "Lingbot-VLA-2",
    family: "VLA",
    sim: [52, 58, 58, 53, 52, 41, 78, 92, 8, 5, 2.7, 45.4],
    real: [50, 50, 75, 15, 50, 5, 10, 100 / 6, 34],
  },
  {
    name: "Xiaomi-Robotics",
    family: "VLA",
    sim: [52, 46, 54, 62, 38, 24, 73, 84, 5, 2, 0, 40],
    real: [50, 50, 80, 25, 40, 5, 15, 20 / 3, 34],
  },
  {
    name: "XVLA",
    family: "VLA",
    sim: [28, 30, 56, 47, 45, 42, 84, 76, 2, 3, 0.7, 37.6],
  },
  {
    name: "FastWAM",
    family: "WAM",
    sim: [49, 84, 67, 7, 9, 50, 51, 88, 6, 0, 0, 37.4],
    real: [40, 45, 60, 25, 50, 0, 5, 20 / 3, 29],
  },
  {
    name: "Lingbot-VA",
    family: "WAM",
    sim: [50, 44, 39, 40, 44, 22, 59, 15, 2, 2, 6, 29.4],
  },
  {
    name: "StarVLA (Cosmos)",
    family: "WAM",
    sim: [51, 67, 94, 9, 13, 1, 33, 22, 3, 1, 3.3, 27],
  },
  {
    name: "StarVLA (Qwen3)",
    family: "VLA",
    sim: [44, 38, 42, 35, 32, 10, 26, 35, 4, 0, 0, 24.2],
  },
];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Configured links are restricted to local files and HTTP(S); no script URLs.
function safeUrl(value) {
  if (!value || !value.trim()) return "";
  try {
    const url = new URL(value, document.baseURI);
    return ["http:", "https:", "file:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}

const resourceContainer = document.querySelector("#resource-links");
config.resources.forEach((resource) => {
  const url = safeUrl(resource.url);
  const link = element(url ? "a" : "span", "resource-link");
  link.append(element("span", "", resource.label));
  if (url) {
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.title = resource.description;
  } else {
    link.setAttribute("aria-disabled", "true");
    link.append(element("small", "", "Coming soon"));
  }
  resourceContainer.append(link);
});

let selectedGroup = "all";
let selectedTask = "In-USB";
const filters = document.querySelector("#task-filters");
Object.entries(groups).forEach(([id, name]) => {
  const button = element(
    "button",
    "",
    `${name} (${id === "all" ? tasks.length : tasks.filter((task) => task.group === id).length})`,
  );
  button.type = "button";
  button.dataset.group = id;
  button.addEventListener("click", () => {
    selectedGroup = id;
    const visible = tasks.filter((task) => id === "all" || task.group === id);
    if (!visible.some((task) => task.id === selectedTask))
      selectedTask = visible[0].id;
    renderTasks();
  });
  filters.append(button);
});

function renderTasks() {
  filters
    .querySelectorAll("button")
    .forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.group === selectedGroup),
      ),
    );
  const list = document.querySelector("#task-list");
  list.replaceChildren();
  tasks
    .filter((task) => selectedGroup === "all" || task.group === selectedGroup)
    .forEach((task) => {
      const button = element("button", "", task.name);
      button.type = "button";
      button.dataset.task = task.id;
      button.setAttribute("aria-pressed", String(task.id === selectedTask));
      button.append(element("span", "", task.id));
      button.addEventListener("click", () => {
        selectedTask = task.id;
        list
          .querySelectorAll("button")
          .forEach((item) =>
            item.setAttribute(
              "aria-pressed",
              String(item.dataset.task === selectedTask),
            ),
          );
        renderTaskDetail();
      });
      list.append(button);
    });
  renderTaskDetail();
}

function renderTaskDetail() {
  const task = tasks.find((task) => task.id === selectedTask);
  const detail = document.querySelector("#task-detail");
  const header = element("div", "task-detail-header");
  header.append(
    element("p", "task-category", groups[task.group]),
    element("h3", "", task.name),
    element("p", "", task.description),
  );
  const link = element("a", "task-image");
  link.href = `assets/tasks/${task.image}`;
  link.target = "_blank";
  link.rel = "noopener";
  const img = element("img");
  img.loading = "lazy";
  img.src = link.getAttribute("href").replace(".png", ".webp");
  img.alt = `${task.name}: sequential robot, wrist-camera, and tactile observations`;
  img.width = 1200;
  link.append(img);
  const footer = element("div", "task-detail-footer");
  footer.append(
    element(
      "span",
      "",
      task.real
        ? "Simulation + paired real-world evaluation"
        : "Simulation evaluation",
    ),
    element("a", "", "Open full-resolution figure"),
  );
  const fullLink = footer.querySelector("a");
  fullLink.href = link.href;
  fullLink.target = "_blank";
  fullLink.rel = "noopener";
  detail.replaceChildren(header, link, footer);
  if (task.extra) {
    const extra = element("details", "task-extra");
    extra.append(element("summary", "", task.extraLabel));
    const extraLink = element("a");
    extraLink.href = `assets/tasks/${task.extra}`;
    extraLink.target = "_blank";
    extraLink.rel = "noopener";
    const extraImage = element("img");
    extraImage.src = extraLink.getAttribute("href").replace(".png", ".webp");
    extraImage.alt = `${task.name}: ${task.extraLabel.toLowerCase()}`;
    extraImage.loading = "lazy";
    extraLink.append(extraImage);
    extra.append(extraLink);
    detail.append(extra);
  }
}
renderTasks();

let domain = "sim";
let expanded = false;
function renderResults() {
  const domainTasks =
    domain === "sim" ? tasks : tasks.filter((task) => task.real);
  const available = policies
    .filter((policy) => policy[domain])
    .sort((a, b) => b[domain].at(-1) - a[domain].at(-1));
  const shown = expanded ? available : available.slice(0, 5);
  const table = document.querySelector("#results-table");
  const caption = element(
    "caption",
    "sr-only",
    `${domain === "sim" ? "Simulation" : "Real-world"} policy success rates in percent`,
  );
  const thead = element("thead");
  const headRow = element("tr");
  const modelHead = element("th", "", "Policy");
  modelHead.scope = "col";
  headRow.append(modelHead);
  domainTasks.forEach((task) => {
    const th = element("th");
    th.scope = "col";
    const abbr = element("abbr", "", task.id);
    abbr.title = task.name;
    th.append(abbr);
    headRow.append(th);
  });
  const avg = element("th", "", "Avg.");
  avg.scope = "col";
  headRow.append(avg);
  thead.append(headRow);
  const tbody = element("tbody");
  shown.forEach((policy) => {
    const row = element("tr", policy.name === "OpenVTLA" ? "ours" : "");
    const heading = element("th", "", policy.name);
    heading.scope = "row";
    heading.append(element("span", "policy-family", policy.family));
    row.append(heading);
    policy[domain].forEach((value, i) =>
      row.append(
        element(
          "td",
          "",
          i === policy[domain].length - 1
            ? value.toFixed(1)
            : Number.isInteger(value)
              ? String(value)
              : value.toFixed(1),
        ),
      ),
    );
    tbody.append(row);
  });
  table.replaceChildren(caption, thead, tbody);
  document.querySelector("#result-context").textContent =
    `${domain === "sim" ? "11 simulation tasks" : "8 paired real-world tasks"}. Showing ${shown.length} of ${available.length} evaluated policies, ranked by average success.`;
  const button = document.querySelector("#expand-results");
  button.textContent = expanded
    ? "Show top 5 policies"
    : `Show all ${available.length} policies`;
  button.setAttribute("aria-expanded", String(expanded));
  document
    .querySelectorAll("[data-domain]")
    .forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.domain === domain),
      ),
    );
}
document.querySelectorAll("[data-domain]").forEach((button) =>
  button.addEventListener("click", () => {
    domain = button.dataset.domain;
    renderResults();
  }),
);
document.querySelector("#expand-results").addEventListener("click", () => {
  expanded = !expanded;
  renderResults();
});
renderResults();

/* Demo rendering temporarily disabled with the section in index.html.
config.demos.forEach((demo) => {
  const article = element("article", "demo-item");
  const src = safeUrl(demo.src);
  if (src) {
    const video = element("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.src = src;
    video.setAttribute("aria-label", demo.title);
    if (safeUrl(demo.poster)) video.poster = safeUrl(demo.poster);
    if (safeUrl(demo.captions)) {
      const track = element("track");
      track.kind = "captions";
      track.label = "English";
      track.srclang = "en";
      track.src = safeUrl(demo.captions);
      video.append(track);
    }
    video.addEventListener("error", () => {
      if (!article.querySelector(".video-error"))
        article.append(
          element(
            "p",
            "video-error",
            "This video could not be loaded. Please try again later.",
          ),
        );
    });
    article.append(video);
  } else {
    const placeholder = element("div", "demo-media");
    const symbol = element("span", "placeholder-symbol", "[ video ]");
    symbol.setAttribute("aria-hidden", "true");
    placeholder.append(symbol, element("p", "", "Demo coming soon"));
    article.append(placeholder);
  }
  article.append(
    element("h3", "", demo.title),
    element("p", "", demo.description),
  );
  document.querySelector("#demo-grid").append(article);
});
*/

document
  .querySelector("#copy-citation")
  .addEventListener("click", async (event) => {
    const code = document.querySelector("#bibtex");
    const button = event.currentTarget;
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = "Copied";
      document.querySelector("#copy-status").textContent =
        "Citation copied to clipboard.";
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "Citation selected";
      document.querySelector("#copy-status").textContent =
        "Copy unavailable. Citation selected; press Control+C or Command+C to copy.";
    }
    setTimeout(() => {
      button.textContent = "Copy citation";
    }, 2500);
  });

const navLinks = document.querySelectorAll(".nav-links a");
const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        navLinks.forEach((link) => {
          if (link.hash === `#${entry.target.id}`)
            link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      }
    });
  },
  { rootMargin: "-15% 0px -60% 0px" },
);
navLinks.forEach((link) => {
  const section = document.querySelector(link.hash);
  if (section) sectionObserver.observe(section);
});
