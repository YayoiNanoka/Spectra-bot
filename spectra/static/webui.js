/**
 * Spectra-bot Schema-Driven Config Studio (zero-build ES module).
 *
 * Dynamically renders all configuration sections from AppConfig.model_json_schema()
 * so future sections appear automatically without frontend changes.
 */

const TOKEN_STORAGE_KEY = "spectra_webui_token";
const THEME_STORAGE_KEY = "spectra_webui_theme";

const SECRET_SCALAR_RE =
  /^(api_key|auth_token|access_token|secret|password)$|(_key|_token|_secret|_password)$/i;
const SECRET_LIST_RE = /(_keys|_tokens|_secrets|_passwords)$/i;
const SECRET_HEADER_KEY_RE = /(authorization|api[-_]?key|token|secret)/i;

const state = {
  meta: null,
  schema: null,
  savedConfig: null,
  draftConfig: null,
  configExists: true,
  corruptedRecovery: null, // { detail, rawText, errors } when disk config is invalid
  activeSection: null,
  viewMode: "visual", // "visual" | "raw"
  searchQuery: "",
  errorsByPath: new Map(), // "a.b.c" -> string[]
  widgetParseErrors: new Map(), // "a.b.c" -> string (live JSON object parse error)
  rawErrors: [],
  saving: false,
  restartRequired: false,
  dirty: false,
};

const els = {
  metaVersion: document.getElementById("meta-version"),
  metaStatus: document.getElementById("meta-status"),
  metaStatusText: document.getElementById("meta-status-text"),
  metaConfigPath: document.getElementById("meta-config-path"),
  searchInput: document.getElementById("search-input"),
  searchClear: document.getElementById("search-clear"),
  dirtyIndicator: document.getElementById("dirty-indicator"),
  modeVisual: document.getElementById("mode-visual"),
  modeRaw: document.getElementById("mode-raw"),
  btnDefaults: document.getElementById("btn-defaults"),
  btnReload: document.getElementById("btn-reload"),
  btnSave: document.getElementById("btn-save"),
  btnAuth: document.getElementById("btn-auth"),
  btnTheme: document.getElementById("btn-theme"),
  mobileNavBtn: document.getElementById("mobile-nav-btn"),
  sidebar: document.getElementById("sidebar"),
  sectionCount: document.getElementById("section-count"),
  sectionNav: document.getElementById("section-nav"),
  bannerStack: document.getElementById("banner-stack"),
  visualView: document.getElementById("visual-view"),
  rawView: document.getElementById("raw-view"),
  sectionHeader: document.getElementById("section-header"),
  formRoot: document.getElementById("form-root"),
  rawEditor: document.getElementById("raw-editor"),
  rawError: document.getElementById("raw-error"),
  btnRawFormat: document.getElementById("btn-raw-format"),
  btnRawValidate: document.getElementById("btn-raw-validate"),
  authModal: document.getElementById("auth-modal"),
  authForm: document.getElementById("auth-form"),
  authInput: document.getElementById("auth-token-input"),
  authToggle: document.getElementById("auth-token-toggle"),
  authError: document.getElementById("auth-error"),
  authClearBtn: document.getElementById("auth-clear-btn"),
  toastContainer: document.getElementById("toast-container"),
};

/* ---------- Theme ---------- */

function initTheme() {
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  const preferred =
    stored ||
    (window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark");
  applyTheme(preferred);
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem(THEME_STORAGE_KEY, theme);
  els.btnTheme.textContent = theme === "dark" ? "\u2600" : "\u263D";
  els.btnTheme.title =
    theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
}

/* ---------- Auth Token (sessionStorage only) ---------- */

function getAuthToken() {
  return sessionStorage.getItem(TOKEN_STORAGE_KEY) || "";
}

function setAuthToken(token) {
  if (token) {
    sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
  } else {
    sessionStorage.removeItem(TOKEN_STORAGE_KEY);
  }
}

function openAuthModal(errorMessage = "") {
  els.authInput.value = getAuthToken();
  els.authInput.type = "password";
  els.authToggle.textContent = "Show";
  els.authClearBtn.classList.toggle("hidden", !getAuthToken());
  if (errorMessage) {
    els.authError.textContent = errorMessage;
    els.authError.classList.remove("hidden");
  } else {
    els.authError.textContent = "";
    els.authError.classList.add("hidden");
  }
  els.authModal.classList.remove("hidden");
  els.authInput.focus();
}

function closeAuthModal() {
  els.authModal.classList.add("hidden");
}

/* ---------- API Client ---------- */

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (options.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    ...options,
    headers,
    cache: "no-store",
  });
  let data = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    data = await response.json();
  }
  return { ok: response.ok, status: response.status, data };
}

/* ---------- Toast ---------- */

function showToast(message, variant = "info", durationMs = 4200) {
  const toast = document.createElement("div");
  toast.className = `toast toast-${variant}`;
  toast.textContent = message;
  els.toastContainer.appendChild(toast);
  window.setTimeout(() => {
    toast.remove();
  }, durationMs);
}

/* ---------- JSON Schema Helpers ---------- */

function resolveRef(node, rootSchema) {
  if (!node || typeof node !== "object") {
    return {};
  }
  if (!node.$ref) {
    return node;
  }
  const ref = String(node.$ref);
  let target = null;
  if (ref.startsWith("#/$defs/")) {
    const defName = ref.slice("#/$defs/".length);
    target = rootSchema?.$defs?.[defName] || null;
  } else if (ref.startsWith("#/definitions/")) {
    const defName = ref.slice("#/definitions/".length);
    target = rootSchema?.definitions?.[defName] || null;
  }
  if (!target) {
    return node;
  }
  const resolved = resolveRef(target, rootSchema);
  const { $ref: _ignored, ...overlay } = node;
  return { ...resolved, ...overlay };
}

function unwrapSchema(rawNode, rootSchema) {
  let node = resolveRef(rawNode, rootSchema);
  let nullable = false;

  const variants = node.anyOf || node.oneOf;
  if (Array.isArray(variants) && variants.length > 0) {
    const nonNull = [];
    for (const branch of variants) {
      const resolvedBranch = resolveRef(branch, rootSchema);
      if (resolvedBranch.type === "null") {
        nullable = true;
      } else {
        nonNull.push(resolvedBranch);
      }
    }
    if (nonNull.length === 1) {
      const { anyOf: _a, oneOf: _o, ...meta } = node;
      node = { ...nonNull[0], ...meta };
    }
  }

  return { schema: node, nullable };
}

function isSecretField(fieldKey) {
  return SECRET_SCALAR_RE.test(fieldKey);
}

function isSecretListField(fieldKey) {
  return SECRET_LIST_RE.test(fieldKey);
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function getValueAtPath(obj, path) {
  let cur = obj;
  for (const key of path) {
    if (cur == null || typeof cur !== "object") {
      return undefined;
    }
    cur = cur[key];
  }
  return cur;
}

function setValueAtPath(obj, path, value) {
  let cur = obj;
  for (let i = 0; i < path.length - 1; i += 1) {
    const key = path[i];
    if (cur[key] == null || typeof cur[key] !== "object") {
      cur[key] = {};
    }
    cur = cur[key];
  }
  cur[path[path.length - 1]] = value;
}

function pathKey(path) {
  return path.map(String).join(".");
}

function buildConstraintHint(schema) {
  const parts = [];
  if (schema.minimum !== undefined && schema.maximum !== undefined) {
    parts.push(`${schema.minimum} .. ${schema.maximum}`);
  } else if (schema.minimum !== undefined) {
    parts.push(`>= ${schema.minimum}`);
  } else if (schema.exclusiveMinimum !== undefined) {
    parts.push(`> ${schema.exclusiveMinimum}`);
  }
  if (schema.maximum !== undefined && schema.minimum === undefined) {
    parts.push(`<= ${schema.maximum}`);
  } else if (schema.exclusiveMaximum !== undefined) {
    parts.push(`< ${schema.exclusiveMaximum}`);
  }
  if (schema.format === "path") {
    parts.push("path");
  }
  return parts.join(", ");
}

/* ---------- Validation Error Mapping ---------- */

function setValidationErrors(errors) {
  state.rawErrors = Array.isArray(errors) ? errors : [];
  state.errorsByPath.clear();
  for (const err of state.rawErrors) {
    const loc = Array.isArray(err.loc) ? err.loc : [];
    const key = pathKey(loc);
    const list = state.errorsByPath.get(key) || [];
    list.push(err.msg || "Invalid value");
    state.errorsByPath.set(key, list);
  }
}

function countSectionErrors(sectionKey) {
  let count = 0;
  for (const [key, messages] of state.errorsByPath.entries()) {
    if (key === sectionKey || key.startsWith(`${sectionKey}.`)) {
      count += messages.length;
    }
  }
  return count;
}

/* ---------- Dirty Tracking ---------- */

function recomputeDirty() {
  if (state.corruptedRecovery) {
    state.dirty = true;
  } else if (state.savedConfig && state.draftConfig) {
    state.dirty =
      JSON.stringify(state.savedConfig) !== JSON.stringify(state.draftConfig);
  } else {
    state.dirty = false;
  }
  els.dirtyIndicator.classList.toggle("hidden", !state.dirty);
}

/* ---------- Banners ---------- */

function renderBanners() {
  els.bannerStack.textContent = "";

  if (state.corruptedRecovery) {
    const banner = document.createElement("div");
    banner.className = "banner banner-danger";

    const msgWrap = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = "Configuration Recovery Mode: ";
    msgWrap.appendChild(strong);
    msgWrap.appendChild(
      document.createTextNode(
        state.corruptedRecovery.detail ||
          "Existing config file is invalid. Repair it in Raw JSON or load defaults."
      )
    );

    const actions = document.createElement("div");
    actions.className = "banner-actions";
    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "btn btn-secondary btn-sm";
    resetBtn.textContent = "Load Defaults";
    resetBtn.addEventListener("click", () => {
      void handleLoadDefaults();
    });
    actions.appendChild(resetBtn);

    banner.append(msgWrap, actions);
    els.bannerStack.appendChild(banner);
  } else if (!state.configExists) {
    const banner = document.createElement("div");
    banner.className = "banner banner-info";
    const text = document.createElement("div");
    text.textContent = `No config file exists yet at ${state.meta?.config_path || "config.json"}. Showing defaults — click Save to create it.`;
    banner.appendChild(text);
    els.bannerStack.appendChild(banner);
  }

  if (state.restartRequired) {
    const banner = document.createElement("div");
    banner.className = "banner banner-success";
    const text = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = "Saved to disk. ";
    text.appendChild(strong);
    text.appendChild(
      document.createTextNode(
        `Saved to ${state.meta?.config_path || "config.json"} — restart Spectra-bot to apply changes.`
      )
    );

    const dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.className = "btn btn-secondary btn-sm";
    dismiss.textContent = "Dismiss";
    dismiss.addEventListener("click", () => {
      state.restartRequired = false;
      renderBanners();
    });

    banner.append(text, dismiss);
    els.bannerStack.appendChild(banner);
  }

  const rootErrors = state.errorsByPath.get("") || [];
  if (rootErrors.length > 0 && !state.corruptedRecovery) {
    const banner = document.createElement("div");
    banner.className = "banner banner-danger";
    const text = document.createElement("div");
    text.textContent = `Validation error: ${rootErrors.join("; ")}`;
    banner.appendChild(text);
    els.bannerStack.appendChild(banner);
  }
}

/* ---------- Search Matching ---------- */

function fieldMatchesSearch(fieldKey, fieldSchema, sectionTitle, query) {
  if (!query) return true;
  const q = query.toLowerCase();
  const haystacks = [
    fieldKey,
    fieldSchema.title || "",
    fieldSchema.description || "",
    sectionTitle || "",
  ];
  if (haystacks.some((s) => s.toLowerCase().includes(q))) {
    return true;
  }
  if (fieldSchema.properties && typeof fieldSchema.properties === "object") {
    for (const [childKey, rawChild] of Object.entries(fieldSchema.properties)) {
      const { schema: childSchema } = unwrapSchema(rawChild, state.schema);
      if (fieldMatchesSearch(childKey, childSchema, sectionTitle, query)) {
        return true;
      }
    }
  }
  return false;
}

function countSectionMatches(sectionKey, query) {
  const rawProp = state.schema?.properties?.[sectionKey];
  if (!rawProp) return 0;
  const { schema: sectionSchema } = unwrapSchema(rawProp, state.schema);
  const props = sectionSchema.properties || {};
  const sectionTitle = sectionSchema.title || sectionKey;
  let count = 0;
  for (const [k, rawChild] of Object.entries(props)) {
    const { schema: childSchema } = unwrapSchema(rawChild, state.schema);
    if (fieldMatchesSearch(k, childSchema, sectionTitle, query)) {
      count += 1;
    }
  }
  return count;
}

/* ---------- Sidebar ---------- */

function renderSidebar() {
  els.sectionNav.textContent = "";
  const properties = state.schema?.properties || {};
  const sectionKeys = Object.keys(properties);
  els.sectionCount.textContent = String(sectionKeys.length);

  for (const key of sectionKeys) {
    const { schema: sectionSchema } = unwrapSchema(properties[key], state.schema);
    const title = sectionSchema.title || key;
    const matchCount = countSectionMatches(key, state.searchQuery);
    const errorCount = countSectionErrors(key);

    if (state.searchQuery && matchCount === 0) {
      continue;
    }

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "nav-item";
    if (!state.searchQuery && state.activeSection === key) {
      btn.classList.add("active");
    }
    btn.title = sectionSchema.description || title;

    const textWrap = document.createElement("div");
    textWrap.className = "nav-item-text";

    const titleEl = document.createElement("span");
    titleEl.className = "nav-item-title";
    titleEl.textContent = title;

    const keyEl = document.createElement("span");
    keyEl.className = "nav-item-key";
    keyEl.textContent = key;

    textWrap.append(titleEl, keyEl);

    const badges = document.createElement("div");
    badges.className = "nav-badges";

    if (errorCount > 0) {
      const errBadge = document.createElement("span");
      errBadge.className = "badge-error";
      errBadge.textContent = String(errorCount);
      errBadge.title = `${errorCount} validation error(s)`;
      badges.appendChild(errBadge);
    }

    const countBadge = document.createElement("span");
    countBadge.className = "badge-count";
    countBadge.textContent = String(matchCount);
    badges.appendChild(countBadge);

    btn.append(textWrap, badges);
    btn.addEventListener("click", () => {
      state.activeSection = key;
      if (state.searchQuery) {
        state.searchQuery = "";
        els.searchInput.value = "";
        els.searchClear.classList.add("hidden");
      }
      if (state.viewMode !== "visual") {
        switchViewMode("visual");
      }
      els.sidebar.classList.remove("open");
      renderSidebar();
      renderForm();
    });

    els.sectionNav.appendChild(btn);
  }
}

/* ---------- Field Widgets ---------- */

function onFieldMutated(path, value) {
  setValueAtPath(state.draftConfig, path, value);
  state.errorsByPath.delete(pathKey(path));
  // Clear section-level error (e.g. SessionConfig._validate_window_bounds) when any child changes
  if (path.length >= 2) {
    state.errorsByPath.delete(pathKey(path.slice(0, -1)));
  }
  recomputeDirty();
  renderSidebar();
}

function createErrorList(messages) {
  const div = document.createElement("div");
  div.className = "inline-error";
  div.setAttribute("role", "alert");
  div.textContent = messages.join(" • ");
  return div;
}

function renderStringListWidget(path, fieldKey, value, itemType) {
  const items = Array.isArray(value) ? [...value] : [];
  const maskItems = isSecretListField(fieldKey);
  const wrap = document.createElement("div");
  wrap.className = "list-widget";

  function rerender() {
    wrap.textContent = "";
    if (items.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-hint";
      empty.textContent = "No entries yet.";
      wrap.appendChild(empty);
    }

    items.forEach((item, idx) => {
      const row = document.createElement("div");
      row.className = "list-item";

      const input = document.createElement("input");
      input.className = "text-input";
      input.type = maskItems
        ? "password"
        : itemType === "integer" || itemType === "number"
          ? "number"
          : "text";
      input.value = item == null ? "" : String(item);
      input.addEventListener("input", () => {
        if (itemType === "integer") {
          items[idx] = input.value === "" ? 0 : Number.parseInt(input.value, 10);
        } else if (itemType === "number") {
          items[idx] = input.value === "" ? 0 : Number.parseFloat(input.value);
        } else {
          items[idx] = input.value;
        }
        onFieldMutated(path, [...items]);
      });
      row.appendChild(input);

      if (maskItems) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "icon-btn-inline";
        toggle.textContent = "Show";
        toggle.addEventListener("click", () => {
          const showing = input.type === "text";
          input.type = showing ? "password" : "text";
          toggle.textContent = showing ? "Show" : "Hide";
        });
        row.appendChild(toggle);
      }

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "btn btn-danger-ghost";
      removeBtn.textContent = "Remove";
      removeBtn.addEventListener("click", () => {
        items.splice(idx, 1);
        onFieldMutated(path, [...items]);
        rerender();
      });
      row.appendChild(removeBtn);

      wrap.appendChild(row);
    });

    const addRow = document.createElement("div");
    addRow.className = "list-add-row";

    const addInput = document.createElement("input");
    addInput.className = "text-input";
    addInput.type = maskItems ? "password" : "text";
    addInput.placeholder = maskItems ? "Add secret key..." : "Add new value...";

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "btn btn-secondary btn-sm";
    addBtn.textContent = "+ Add";

    const commitAdd = () => {
      const raw = addInput.value;
      if (!raw.trim()) return;
      if (itemType === "integer") {
        items.push(Number.parseInt(raw, 10));
      } else if (itemType === "number") {
        items.push(Number.parseFloat(raw));
      } else {
        items.push(raw.trim());
      }
      addInput.value = "";
      onFieldMutated(path, [...items]);
      rerender();
    };

    addBtn.addEventListener("click", commitAdd);
    addInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        commitAdd();
      }
    });

    addRow.append(addInput, addBtn);
    wrap.appendChild(addRow);
  }

  rerender();
  return wrap;
}

function renderKeyValueWidget(path, value) {
  const entries = Object.entries(
    value && typeof value === "object" && !Array.isArray(value) ? value : {}
  ).map(([k, v]) => ({ key: k, value: v == null ? "" : String(v) }));

  const wrap = document.createElement("div");
  wrap.className = "list-widget";

  function syncObject() {
    const out = {};
    for (const entry of entries) {
      const k = entry.key.trim();
      if (k) {
        out[k] = entry.value;
      }
    }
    onFieldMutated(path, out);
  }

  function rerender() {
    wrap.textContent = "";
    if (entries.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-hint";
      empty.textContent = "No entries configured.";
      wrap.appendChild(empty);
    }

    entries.forEach((entry, idx) => {
      const row = document.createElement("div");
      row.className = "kv-row";

      const keyInput = document.createElement("input");
      keyInput.className = "text-input mono";
      keyInput.type = "text";
      keyInput.placeholder = "Header / Key";
      keyInput.value = entry.key;

      const valWrap = document.createElement("div");
      valWrap.className = "secret-input-wrap";

      const valInput = document.createElement("input");
      valInput.className = "text-input";
      const looksSecret = SECRET_HEADER_KEY_RE.test(entry.key);
      valInput.type = looksSecret ? "password" : "text";
      valInput.placeholder = "Value";
      valInput.value = entry.value;

      valWrap.appendChild(valInput);

      let toggleBtn = null;
      if (looksSecret) {
        toggleBtn = document.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "icon-btn-inline";
        toggleBtn.textContent = "Show";
        toggleBtn.addEventListener("click", () => {
          const showing = valInput.type === "text";
          valInput.type = showing ? "password" : "text";
          toggleBtn.textContent = showing ? "Show" : "Hide";
        });
        valWrap.appendChild(toggleBtn);
      }

      keyInput.addEventListener("input", () => {
        entry.key = keyInput.value;
        syncObject();
      });
      valInput.addEventListener("input", () => {
        entry.value = valInput.value;
        syncObject();
      });

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "btn btn-danger-ghost";
      removeBtn.textContent = "Remove";
      removeBtn.addEventListener("click", () => {
        entries.splice(idx, 1);
        syncObject();
        rerender();
      });

      row.append(keyInput, valWrap, removeBtn);
      wrap.appendChild(row);
    });

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "btn btn-secondary btn-sm";
    addBtn.style.alignSelf = "flex-start";
    addBtn.textContent = "+ Add Entry";
    addBtn.addEventListener("click", () => {
      entries.push({ key: "", value: "" });
      rerender();
    });
    wrap.appendChild(addBtn);
  }

  rerender();
  return wrap;
}

function renderJsonObjectWidget(path, value) {
  const fieldKey = pathKey(path);
  const wrap = document.createElement("div");
  wrap.className = "list-widget";

  const textarea = document.createElement("textarea");
  textarea.className = "textarea-input mono";
  textarea.spellcheck = false;
  textarea.value = JSON.stringify(
    value && typeof value === "object" ? value : {},
    null,
    2
  );

  const parseErr = document.createElement("div");
  parseErr.className = "inline-error hidden";

  textarea.addEventListener("input", () => {
    const raw = textarea.value.trim();
    if (!raw) {
      state.widgetParseErrors.delete(fieldKey);
      parseErr.classList.add("hidden");
      textarea.classList.remove("input-error");
      onFieldMutated(path, {});
      return;
    }
    try {
      const parsed = JSON.parse(raw);
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Value must be a JSON object ({ ... }).");
      }
      state.widgetParseErrors.delete(fieldKey);
      parseErr.classList.add("hidden");
      textarea.classList.remove("input-error");
      onFieldMutated(path, parsed);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Invalid JSON";
      state.widgetParseErrors.set(fieldKey, msg);
      parseErr.textContent = msg;
      parseErr.classList.remove("hidden");
      textarea.classList.add("input-error");
    }
  });

  wrap.append(textarea, parseErr);
  return wrap;
}

function renderField(path, fieldKey, rawFieldSchema) {
  const { schema, nullable } = unwrapSchema(rawFieldSchema, state.schema);
  const fullPathKey = pathKey(path);
  const fieldErrors = state.errorsByPath.get(fullPathKey) || [];
  const currentValue = getValueAtPath(state.draftConfig, path);

  // Nested sub-model object with declared properties
  if (schema.type === "object" && schema.properties) {
    const card = document.createElement("div");
    card.className = "nested-card";
    if (fieldErrors.length > 0) {
      card.classList.add("has-error");
    }

    const header = document.createElement("div");
    header.className = "nested-header";

    const titleRow = document.createElement("div");
    titleRow.className = "field-title-row";
    const title = document.createElement("span");
    title.className = "field-label";
    title.textContent = schema.title || fieldKey;
    const keyBadge = document.createElement("span");
    keyBadge.className = "field-key";
    keyBadge.textContent = fullPathKey;
    titleRow.append(title, keyBadge);
    header.appendChild(titleRow);

    if (schema.description) {
      const desc = document.createElement("p");
      desc.className = "field-desc";
      desc.textContent = schema.description;
      header.appendChild(desc);
    }
    if (fieldErrors.length > 0) {
      header.appendChild(createErrorList(fieldErrors));
    }
    card.appendChild(header);

    const children = document.createElement("div");
    children.className = "nested-children";
    for (const [childKey, childSchema] of Object.entries(schema.properties)) {
      children.appendChild(renderField([...path, childKey], childKey, childSchema));
    }
    card.appendChild(children);
    return card;
  }

  const card = document.createElement("div");
  card.className = "field-card";
  card.dataset.path = fullPathKey;
  if (fieldErrors.length > 0) {
    card.classList.add("has-error");
  }

  const info = document.createElement("div");
  info.className = "field-info";

  const titleRow = document.createElement("div");
  titleRow.className = "field-title-row";

  const label = document.createElement("label");
  label.className = "field-label";
  label.textContent = schema.title || fieldKey;
  const inputId = `field-${fullPathKey.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
  label.htmlFor = inputId;

  const keyBadge = document.createElement("span");
  keyBadge.className = "field-key";
  keyBadge.textContent = fieldKey;

  titleRow.append(label, keyBadge);

  const hint = buildConstraintHint(schema);
  if (hint) {
    const constraint = document.createElement("span");
    constraint.className = "field-constraint";
    constraint.textContent = hint;
    titleRow.appendChild(constraint);
  }

  info.appendChild(titleRow);

  if (schema.description) {
    const desc = document.createElement("p");
    desc.className = "field-desc";
    desc.textContent = schema.description;
    info.appendChild(desc);
  }

  // Boolean -> inline switch on the right
  if (schema.type === "boolean" && !nullable) {
    const row = document.createElement("div");
    row.className = "field-card-row";

    const controlWrap = document.createElement("div");
    controlWrap.className = "field-control-inline";

    const switchLabel = document.createElement("label");
    switchLabel.className = "switch";

    const checkbox = document.createElement("input");
    checkbox.id = inputId;
    checkbox.type = "checkbox";
    checkbox.role = "switch";
    checkbox.checked = Boolean(currentValue);
    checkbox.addEventListener("change", () => {
      onFieldMutated(path, checkbox.checked);
    });

    const slider = document.createElement("span");
    slider.className = "switch-slider";

    switchLabel.append(checkbox, slider);
    controlWrap.appendChild(switchLabel);
    row.append(info, controlWrap);
    card.appendChild(row);
    if (fieldErrors.length > 0) {
      card.appendChild(createErrorList(fieldErrors));
    }
    return card;
  }

  card.appendChild(info);

  const control = document.createElement("div");
  control.className = "field-control";

  let valueContainer = control;
  if (nullable) {
    const nullRow = document.createElement("label");
    nullRow.className = "nullable-row";
    const nullCheckbox = document.createElement("input");
    nullCheckbox.type = "checkbox";
    nullCheckbox.checked = currentValue === null || currentValue === undefined;
    nullCheckbox.addEventListener("change", () => {
      if (nullCheckbox.checked) {
        onFieldMutated(path, null);
      } else {
        const fallback =
          schema.default !== undefined
            ? schema.default
            : schema.type === "integer" || schema.type === "number"
              ? 0
              : schema.type === "boolean"
                ? false
                : "";
        onFieldMutated(path, fallback);
      }
      renderForm();
    });
    nullRow.append(nullCheckbox, document.createTextNode("Use null (unset)"));
    control.appendChild(nullRow);

    if (currentValue === null || currentValue === undefined) {
      card.appendChild(control);
      if (fieldErrors.length > 0) {
        card.appendChild(createErrorList(fieldErrors));
      }
      return card;
    }
  }

  // Const (single literal)
  if (schema.const !== undefined) {
    const input = document.createElement("input");
    input.id = inputId;
    input.className = "text-input mono";
    input.type = "text";
    input.value = String(schema.const);
    input.readOnly = true;
    input.disabled = true;
    valueContainer.appendChild(input);
  }
  // Enum -> select
  else if (Array.isArray(schema.enum)) {
    const select = document.createElement("select");
    select.id = inputId;
    select.className = "select-input";
    if (fieldErrors.length > 0) select.classList.add("input-error");

    for (const optionVal of schema.enum) {
      const opt = document.createElement("option");
      opt.value = String(optionVal);
      opt.textContent = String(optionVal);
      if (String(currentValue) === String(optionVal)) {
        opt.selected = true;
      }
      select.appendChild(opt);
    }
    select.addEventListener("change", () => {
      const matched = schema.enum.find((v) => String(v) === select.value);
      onFieldMutated(path, matched !== undefined ? matched : select.value);
    });
    valueContainer.appendChild(select);
  }
  // Integer / Number
  else if (schema.type === "integer" || schema.type === "number") {
    const input = document.createElement("input");
    input.id = inputId;
    input.className = "text-input";
    if (fieldErrors.length > 0) input.classList.add("input-error");
    input.type = "number";
    input.step = schema.type === "integer" ? "1" : "any";
    if (schema.minimum !== undefined) input.min = String(schema.minimum);
    if (schema.maximum !== undefined) input.max = String(schema.maximum);
    input.value = currentValue == null ? "" : String(currentValue);

    input.addEventListener("input", () => {
      if (input.value === "") {
        onFieldMutated(path, null);
        return;
      }
      const parsed =
        schema.type === "integer"
          ? Number.parseInt(input.value, 10)
          : Number.parseFloat(input.value);
      onFieldMutated(path, Number.isNaN(parsed) ? input.value : parsed);
    });
    valueContainer.appendChild(input);
  }
  // Array
  else if (schema.type === "array") {
    const { schema: itemSchema } = unwrapSchema(schema.items || {}, state.schema);
    const itemType = itemSchema.type || "string";
    valueContainer.appendChild(
      renderStringListWidget(path, fieldKey, currentValue, itemType)
    );
  }
  // Object with additionalProperties (dict[str, str] vs dict[str, Any])
  else if (schema.type === "object") {
    const addProps = schema.additionalProperties;
    const isStringMap =
      addProps &&
      typeof addProps === "object" &&
      unwrapSchema(addProps, state.schema).schema.type === "string";
    if (isStringMap) {
      valueContainer.appendChild(renderKeyValueWidget(path, currentValue));
    } else {
      valueContainer.appendChild(renderJsonObjectWidget(path, currentValue));
    }
  }
  // String (default)
  else {
    const strVal = currentValue == null ? "" : String(currentValue);
    if (isSecretField(fieldKey)) {
      const secretWrap = document.createElement("div");
      secretWrap.className = "secret-input-wrap";

      const input = document.createElement("input");
      input.id = inputId;
      input.className = "text-input";
      if (fieldErrors.length > 0) input.classList.add("input-error");
      input.type = "password";
      input.autocomplete = "off";
      input.value = strVal;
      input.addEventListener("input", () => {
        onFieldMutated(path, input.value);
      });

      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "icon-btn-inline";
      toggle.textContent = "Show";
      toggle.addEventListener("click", () => {
        const showing = input.type === "text";
        input.type = showing ? "password" : "text";
        toggle.textContent = showing ? "Show" : "Hide";
      });

      secretWrap.append(input, toggle);
      valueContainer.appendChild(secretWrap);
    } else if (fieldKey === "prompt" || strVal.includes("\n") || strVal.length > 100) {
      const textarea = document.createElement("textarea");
      textarea.id = inputId;
      textarea.className = "textarea-input";
      if (fieldErrors.length > 0) textarea.classList.add("input-error");
      textarea.value = strVal;
      textarea.addEventListener("input", () => {
        onFieldMutated(path, textarea.value);
      });
      valueContainer.appendChild(textarea);
    } else {
      const input = document.createElement("input");
      input.id = inputId;
      input.className = "text-input";
      if (fieldErrors.length > 0) input.classList.add("input-error");
      input.type = "text";
      input.value = strVal;
      input.addEventListener("input", () => {
        onFieldMutated(path, input.value);
      });
      valueContainer.appendChild(input);
    }
  }

  card.appendChild(control);
  if (fieldErrors.length > 0) {
    card.appendChild(createErrorList(fieldErrors));
  }
  return card;
}

/* ---------- Form Renderer ---------- */

function renderForm() {
  if (!state.schema || !state.draftConfig) {
    return;
  }
  els.sectionHeader.textContent = "";
  els.formRoot.textContent = "";

  const properties = state.schema.properties || {};
  const sectionKeys = Object.keys(properties);

  if (state.searchQuery) {
    const heading = document.createElement("h2");
    heading.className = "section-heading";
    heading.textContent = `Search Results for "${state.searchQuery}"`;
    els.sectionHeader.appendChild(heading);

    let totalRendered = 0;
    for (const secKey of sectionKeys) {
      const { schema: secSchema } = unwrapSchema(properties[secKey], state.schema);
      const secTitle = secSchema.title || secKey;
      const fields = secSchema.properties || {};
      const matchingEntries = Object.entries(fields).filter(([fk, rawChild]) => {
        const { schema: childSchema } = unwrapSchema(rawChild, state.schema);
        return fieldMatchesSearch(fk, childSchema, secTitle, state.searchQuery);
      });
      if (matchingEntries.length === 0) continue;

      const group = document.createElement("div");
      group.className = "section-group";

      const groupTitle = document.createElement("h3");
      groupTitle.className = "section-group-title";
      groupTitle.textContent = secTitle;
      group.appendChild(groupTitle);

      for (const [fk, rawChild] of matchingEntries) {
        group.appendChild(renderField([secKey, fk], fk, rawChild));
        totalRendered += 1;
      }
      els.formRoot.appendChild(group);
    }

    if (totalRendered === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-search";
      empty.textContent = "No settings match your filter.";
      els.formRoot.appendChild(empty);
    }
    return;
  }

  const activeKey = state.activeSection || sectionKeys[0];
  if (!activeKey || !properties[activeKey]) {
    return;
  }
  state.activeSection = activeKey;

  const { schema: sectionSchema } = unwrapSchema(properties[activeKey], state.schema);
  const heading = document.createElement("h2");
  heading.className = "section-heading";
  heading.textContent = sectionSchema.title || activeKey;
  els.sectionHeader.appendChild(heading);

  if (sectionSchema.description) {
    const desc = document.createElement("p");
    desc.className = "section-desc";
    desc.textContent = sectionSchema.description;
    els.sectionHeader.appendChild(desc);
  }

  const sectionErrors = state.errorsByPath.get(activeKey) || [];
  if (sectionErrors.length > 0) {
    const banner = document.createElement("div");
    banner.className = "banner banner-danger";
    banner.style.marginTop = "10px";
    banner.textContent = sectionErrors.join(" • ");
    els.sectionHeader.appendChild(banner);
  }

  const fields = sectionSchema.properties || {};
  for (const [fieldKey, fieldSchema] of Object.entries(fields)) {
    els.formRoot.appendChild(renderField([activeKey, fieldKey], fieldKey, fieldSchema));
  }
}

/* ---------- Raw JSON Mode ---------- */

function setRawError(message) {
  if (message) {
    els.rawError.textContent = message;
    els.rawError.classList.remove("hidden");
  } else {
    els.rawError.textContent = "";
    els.rawError.classList.add("hidden");
  }
}

function parseRawEditorJson() {
  try {
    const parsed = JSON.parse(els.rawEditor.value);
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { ok: false, error: "Root configuration must be a JSON object ({ ... })." };
    }
    return { ok: true, value: parsed };
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? err.message : "Invalid JSON syntax.",
    };
  }
}

function switchViewMode(targetMode) {
  if (targetMode === state.viewMode) return;

  if (targetMode === "raw") {
    if (state.corruptedRecovery && state.corruptedRecovery.rawText !== undefined) {
      els.rawEditor.value = state.corruptedRecovery.rawText;
    } else if (state.draftConfig) {
      els.rawEditor.value = JSON.stringify(state.draftConfig, null, 2);
    }
    setRawError("");
  } else {
    const parsed = parseRawEditorJson();
    if (!parsed.ok) {
      setRawError(`Fix JSON syntax before switching to Form view: ${parsed.error}`);
      showToast("Invalid JSON — fix syntax before switching to Form view.", "danger");
      return;
    }
    state.draftConfig = parsed.value;
    setRawError("");
    recomputeDirty();
    renderForm();
  }

  state.viewMode = targetMode;
  els.modeVisual.classList.toggle("active", targetMode === "visual");
  els.modeVisual.setAttribute("aria-selected", String(targetMode === "visual"));
  els.modeRaw.classList.toggle("active", targetMode === "raw");
  els.modeRaw.setAttribute("aria-selected", String(targetMode === "raw"));
  els.visualView.classList.toggle("hidden", targetMode !== "visual");
  els.rawView.classList.toggle("hidden", targetMode !== "raw");
}

/* ---------- Actions ---------- */

async function loadMeta() {
  const res = await apiFetch("/api/meta");
  if (!res.ok || !res.data) {
    throw new Error("Failed to load server metadata.");
  }
  state.meta = res.data;
  els.metaVersion.textContent = `v${res.data.version}`;
  els.metaConfigPath.textContent = res.data.config_path;
  els.metaConfigPath.title = res.data.config_path;

  els.metaStatus.classList.remove("running", "standalone");
  if (res.data.bot_running) {
    els.metaStatus.classList.add("running");
    els.metaStatusText.textContent = "Bot Running";
  } else {
    els.metaStatus.classList.add("standalone");
    els.metaStatusText.textContent = "Standalone WebUI";
  }

  els.btnAuth.classList.toggle("hidden", !res.data.auth_required);
}

async function loadSchemaAndConfig() {
  const [schemaRes, configRes] = await Promise.all([
    apiFetch("/api/config/schema"),
    apiFetch("/api/config"),
  ]);

  if (schemaRes.status === 401 || configRes.status === 401) {
    openAuthModal(getAuthToken() ? "Invalid or expired token. Please try again." : "");
    return false;
  }
  if (!schemaRes.ok || !schemaRes.data) {
    showToast("Failed to load configuration schema.", "danger");
    return false;
  }

  state.schema = schemaRes.data;
  const sectionKeys = Object.keys(state.schema.properties || {});
  if (!state.activeSection || !sectionKeys.includes(state.activeSection)) {
    state.activeSection = sectionKeys[0] || null;
  }

  if (configRes.status === 422 && configRes.data) {
    // Existing file on disk is corrupted or fails schema validation.
    state.widgetParseErrors.clear();
    state.configExists = true;
    state.corruptedRecovery = {
      detail: configRes.data.detail || "Configuration file is invalid.",
      rawText: configRes.data.raw_text || "",
      errors: configRes.data.errors || [],
    };
    setValidationErrors(configRes.data.errors || []);
    if (configRes.data.raw_config && typeof configRes.data.raw_config === "object") {
      state.draftConfig = deepClone(configRes.data.raw_config);
      state.savedConfig = null;
    } else {
      const defRes = await apiFetch("/api/config/defaults");
      state.draftConfig = defRes.ok && defRes.data ? defRes.data : {};
      state.savedConfig = null;
    }
    els.rawEditor.value =
      state.corruptedRecovery.rawText || JSON.stringify(state.draftConfig, null, 2);
    state.viewMode = "visual";
    switchViewMode("raw");
    const errSummary = (configRes.data.errors || [])
      .map((e) => `${(e.loc || []).join(".") || "root"}: ${e.msg}`)
      .join("\n");
    setRawError(errSummary || state.corruptedRecovery.detail);
  } else if (configRes.ok && configRes.data) {
    state.corruptedRecovery = null;
    state.widgetParseErrors.clear();
    state.configExists = Boolean(configRes.data.exists);
    state.savedConfig = deepClone(configRes.data.config);
    state.draftConfig = deepClone(configRes.data.config);
    setValidationErrors([]);
    els.rawEditor.value = JSON.stringify(state.draftConfig, null, 2);
    setRawError("");
  } else {
    showToast(configRes.data?.detail || "Failed to load configuration.", "danger");
    return false;
  }

  recomputeDirty();
  renderBanners();
  renderSidebar();
  renderForm();
  return true;
}

async function handleLoadDefaults() {
  const res = await apiFetch("/api/config/defaults");
  if (res.status === 401) {
    openAuthModal("Authentication required.");
    return;
  }
  if (!res.ok || !res.data) {
    showToast("Failed to fetch default configuration.", "danger");
    return;
  }
  state.corruptedRecovery = null;
  state.widgetParseErrors.clear();
  state.draftConfig = deepClone(res.data);
  setValidationErrors([]);
  els.rawEditor.value = JSON.stringify(state.draftConfig, null, 2);
  setRawError("");
  recomputeDirty();
  renderBanners();
  renderSidebar();
  renderForm();
  showToast("Loaded default values into editor (not saved yet).", "warning");
}

function getPayloadForSubmit() {
  if (state.viewMode === "raw") {
    const parsed = parseRawEditorJson();
    if (!parsed.ok) {
      setRawError(`JSON syntax error: ${parsed.error}`);
      showToast("Fix JSON syntax errors before continuing.", "danger");
      return null;
    }
    setRawError("");
    return parsed.value;
  }
  if (state.widgetParseErrors.size > 0) {
    const [firstPath, firstMsg] = state.widgetParseErrors.entries().next().value;
    showToast(`Fix invalid JSON in "${firstPath}" before saving: ${firstMsg}`, "danger");
    return null;
  }
  return state.draftConfig;
}

async function handleValidateRaw() {
  const payload = getPayloadForSubmit();
  if (!payload) return;

  const res = await apiFetch("/api/config/validate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if (res.status === 401) {
    openAuthModal("Authentication required.");
    return;
  }
  if (res.ok && res.data?.valid) {
    setValidationErrors([]);
    setRawError("");
    state.draftConfig = deepClone(res.data.normalized);
    els.rawEditor.value = JSON.stringify(res.data.normalized, null, 2);
    renderSidebar();
    showToast("Configuration is valid.", "success");
    return;
  }
  const errors = res.data?.errors || [];
  setValidationErrors(errors);
  renderSidebar();
  const formatted = errors
    .map((e) => `${(e.loc || []).join(".") || "root"}: ${e.msg}`)
    .join("\n");
  setRawError(formatted || res.data?.detail || "Validation failed.");
  showToast(`Validation failed (${errors.length} error(s)).`, "danger");
}

async function handleSave() {
  if (state.saving) return;
  const payload = getPayloadForSubmit();
  if (!payload) return;

  state.saving = true;
  els.btnSave.disabled = true;
  els.btnSave.textContent = "Saving...";

  try {
    const res = await apiFetch("/api/config", {
      method: "PUT",
      body: JSON.stringify(payload),
    });

    if (res.status === 401) {
      openAuthModal("Authentication token rejected. Please re-enter.");
      return;
    }

    if (res.ok && res.data?.saved) {
      state.corruptedRecovery = null;
      state.configExists = true;
      state.savedConfig = deepClone(res.data.config);
      state.draftConfig = deepClone(res.data.config);
      state.restartRequired = Boolean(res.data.restart_required);
      setValidationErrors([]);
      setRawError("");
      els.rawEditor.value = JSON.stringify(state.draftConfig, null, 2);
      recomputeDirty();
      renderBanners();
      renderSidebar();
      renderForm();
      showToast(
        `Saved to ${state.meta?.config_path || "config.json"} — restart Spectra-bot to apply changes.`,
        "success",
        5500
      );
      return;
    }

    const errors = res.data?.errors || [];
    setValidationErrors(errors);
    renderBanners();
    renderSidebar();

    if (state.viewMode === "raw") {
      const formatted = errors
        .map((e) => `${(e.loc || []).join(".") || "root"}: ${e.msg}`)
        .join("\n");
      setRawError(formatted || res.data?.detail || "Failed to save configuration.");
    } else {
      // Jump to first section containing an error if current section has none
      if (errors.length > 0 && countSectionErrors(state.activeSection) === 0) {
        const firstSection = errors[0]?.loc?.[0];
        if (firstSection && state.schema?.properties?.[firstSection]) {
          state.activeSection = String(firstSection);
          renderSidebar();
        }
      }
      renderForm();
    }
    showToast(
      errors.length > 0
        ? `Validation failed with ${errors.length} error(s).`
        : res.data?.detail || "Failed to save configuration.",
      "danger"
    );
  } catch (err) {
    showToast(
      `Network error while saving: ${err instanceof Error ? err.message : String(err)}`,
      "danger"
    );
  } finally {
    state.saving = false;
    els.btnSave.disabled = false;
    els.btnSave.textContent = "Save";
  }
}

/* ---------- Event Listeners ---------- */

function bindEvents() {
  els.btnTheme.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    applyTheme(current === "dark" ? "light" : "dark");
  });

  els.mobileNavBtn.addEventListener("click", () => {
    els.sidebar.classList.toggle("open");
  });

  els.searchInput.addEventListener("input", () => {
    state.searchQuery = els.searchInput.value.trim();
    els.searchClear.classList.toggle("hidden", !state.searchQuery);
    if (state.viewMode !== "visual" && state.searchQuery) {
      switchViewMode("visual");
    }
    renderSidebar();
    renderForm();
  });

  els.searchClear.addEventListener("click", () => {
    state.searchQuery = "";
    els.searchInput.value = "";
    els.searchClear.classList.add("hidden");
    renderSidebar();
    renderForm();
  });

  els.modeVisual.addEventListener("click", () => switchViewMode("visual"));
  els.modeRaw.addEventListener("click", () => switchViewMode("raw"));

  els.btnDefaults.addEventListener("click", () => {
    void handleLoadDefaults();
  });

  els.btnReload.addEventListener("click", async () => {
    const ok = await loadSchemaAndConfig();
    if (ok) {
      showToast("Reloaded configuration from disk.", "info");
    }
  });

  els.btnSave.addEventListener("click", () => {
    void handleSave();
  });

  els.btnAuth.addEventListener("click", () => {
    openAuthModal();
  });

  els.btnRawFormat.addEventListener("click", () => {
    const parsed = parseRawEditorJson();
    if (!parsed.ok) {
      setRawError(`Cannot format invalid JSON: ${parsed.error}`);
      return;
    }
    setRawError("");
    els.rawEditor.value = JSON.stringify(parsed.value, null, 2);
  });

  els.btnRawValidate.addEventListener("click", () => {
    void handleValidateRaw();
  });

  els.rawEditor.addEventListener("input", () => {
    const parsed = parseRawEditorJson();
    if (!parsed.ok) {
      setRawError(`JSON syntax error: ${parsed.error}`);
    } else {
      setRawError("");
      state.draftConfig = parsed.value;
    }
    recomputeDirty();
  });

  els.authToggle.addEventListener("click", () => {
    const showing = els.authInput.type === "text";
    els.authInput.type = showing ? "password" : "text";
    els.authToggle.textContent = showing ? "Show" : "Hide";
  });

  els.authClearBtn.addEventListener("click", () => {
    setAuthToken("");
    els.authInput.value = "";
    els.authClearBtn.classList.add("hidden");
    showToast("Cleared stored auth token.", "info");
  });

  els.authForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const token = els.authInput.value.trim();
    if (!token) return;
    setAuthToken(token);
    els.authError.classList.add("hidden");
    const ok = await loadSchemaAndConfig();
    if (ok) {
      closeAuthModal();
      showToast("Authenticated successfully.", "success");
    }
  });

  document.addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "s") {
      ev.preventDefault();
      void handleSave();
    } else if (
      ev.key === "/" &&
      document.activeElement?.tagName !== "INPUT" &&
      document.activeElement?.tagName !== "TEXTAREA" &&
      document.activeElement?.tagName !== "SELECT"
    ) {
      ev.preventDefault();
      els.searchInput.focus();
    }
  });

  window.addEventListener("beforeunload", (ev) => {
    if (state.dirty) {
      ev.preventDefault();
      ev.returnValue = "";
    }
  });
}

/* ---------- Bootstrap ---------- */

async function bootstrap() {
  initTheme();
  bindEvents();
  try {
    await loadMeta();
    if (state.meta.auth_required && !getAuthToken()) {
      openAuthModal();
      return;
    }
    await loadSchemaAndConfig();
  } catch (err) {
    els.formRoot.textContent = "";
    const box = document.createElement("div");
    box.className = "banner banner-danger";
    box.textContent = `Unable to connect to Spectra-bot Config WebUI: ${
      err instanceof Error ? err.message : String(err)
    }`;
    els.formRoot.appendChild(box);
  }
}

void bootstrap();
