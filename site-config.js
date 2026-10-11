// Fill in official release URLs here. Empty URLs display an honest Coming soon state.
// Relative URLs (e.g. assets/paper.pdf) work on GitHub Pages project sites too.
window.OPENVITAC_CONFIG = {
  // Condition IDs match assets/demos/<sim|real>/<folder>/<condition>/<outcome>/.
  // Instructions are verbatim from OpenViTac/envs TASK_INSTRUCTION or build_instruction().
  // Grasp chip instruction is confirmed by the user from the training figure.
  // An empty conditions list keeps the task's static details available.
  taskDemos: {
    WC: { folder: "weight_classify", instruction: "Classify the block weight by touch: place the light block on the blue plate and the heavy block on the yellow plate.", conditions: [["light", "Light"], ["heavy", "Heavy"]] },
    HC: { folder: "hardness_classify", instruction: "Classify the block hardness by touch: place the soft red block on the blue plate and the hard red block on the yellow plate.", conditions: [["soft", "Soft"], ["hard", "Hard"]] },
    RC: { folder: "roughness_classify", real: { instruction: "Classify the block surface by touch: place the smooth block on the right plate and the rough block on the left plate." }, instruction: "Classify the block surface by touch: place the smooth block on the blue plate and the rough block on the yellow plate.", conditions: [["smooth", "Smooth"], ["rough", "Rough"]] },
    RR: { folder: "roughness_regrasp", instruction: "Find the rough block by touch and place it on the yellow plate.", conditions: [["left", "Left"], ["right", "Right"]] },
    ECS: { folder: "empty_can_select", real: true, instruction: "Find the empty can by touch and place it in the basket.", conditions: [["empty_fanta", "Empty Fanta"], ["empty_pepsi", "Empty Pepsi"]] },
    GC: { folder: "grasp_chips", real: true, instruction: "Gently grasp the chip and place it on the plate.", conditions: [["standard", "Standard"]] },
    GA: { folder: "gear_assembly", real: true, instruction: "Pick up the red gear, assemble it onto the base, and rotate it to turn the gear pair.", conditions: [["standard", "Standard"]] },
    PD: { folder: "pull_drawer", real: true, instruction: "Grasp the upper drawer handle and pull the upper drawer open.", conditions: [["standard", "Standard"]] },
    "In-USB": { folder: "insert_usb", real: true, instruction: "Pick up the USB plug from the blue slot and insert it into the red USB slot.", conditions: [["standard", "Standard"]] },
    "In-B v1": { folder: "insert_block_v1", real: { conditions: [["standard", "Standard"]] }, instruction: "Insert the blue cube into the matching hole in the yellow box.", conditions: [["layout1", "Layout 1"], ["layout2", "Layout 2"]] },
    "In-B v2": { folder: "insert_block_v2", real: true, conditions: [["cube", "Cube", "Insert the blue cube into the matching hole in the yellow box."], ["half_cyl", "Half-cylinder", "Insert the blue half cylinder into the matching hole in the yellow box."], ["hex", "Hexagonal prism", "Insert the red hexagonal prism into the matching hole in the yellow box."]] },
  },
  resources: [
    { label: "Paper", url: "https://arxiv.org/pdf/2610.10384v1", description: "Read the paper" },
    {
      label: "GitHub",
      url: "https://github.com/FVL-Repo/OpenViTac",
      description: "Code and benchmark",
    },
    { label: "Hugging Face", url: "", description: "Models and datasets" },
    { label: "ModelScope", url: "", description: "Models and datasets" },
  ],
  // Use local MP4/WebM files or direct video URLs, not YouTube watch-page URLs.
  // A populated src automatically replaces its placeholder with a native player.
  demos: [
    {
      title: "Physical-property perception",
      description: "Distinguish an empty can through physical interaction.",
      src: "",
      poster: "",
      captions: "",
    },
    {
      title: "Fragility-aware manipulation",
      description: "Grasp and place a chip without damaging it.",
      src: "",
      poster: "",
      captions: "",
    },
    {
      title: "Contact-rich manipulation",
      description: "Maintain gear engagement throughout assembly.",
      src: "",
      poster: "",
      captions: "",
    },
    {
      title: "Precision manipulation",
      description: "Use contact feedback to guide USB insertion.",
      src: "",
      poster: "",
      captions: "",
    },
  ],
};
