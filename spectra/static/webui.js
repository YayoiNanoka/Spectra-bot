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
const CJK_RE = /[\u4e00-\u9fff]/;

const ENUM_LABELS = {
  blacklist: "黑名单 (blacklist)",
  whitelist: "白名单 (whitelist)",
  on: "开启 (on)",
  off: "关闭 (off)",
  shadow: "影子模式 (shadow)",
  auto: "自动识别 (auto)",
  system_one: "System One 决策协议 (system_one)",
  openai_decision: "OpenAI 决策协议 (openai_decision)",
  random: "随机区间 (random)",
  log: "对数延迟 (log)",
  regex: "正则表达式 (regex)",
  words: "分隔符列表 (words)",
  possibility_reply: "随机概率回复 (possibility_reply)",
  low: "低 (low)",
  medium: "中 (medium)",
  high: "高 (high)",
  max: "最高 (max)",
  tavily: "Tavily 搜索 (tavily)",
  bocha: "博查搜索 (bocha)",
  brave: "Brave 搜索 (brave)",
  firecrawl: "Firecrawl 搜索与抓取 (firecrawl)",
  baidu_ai_search: "百度 AI 搜索 (baidu_ai_search)",
  exa: "Exa 语义搜索 (exa)",
  anysearch: "AnySearch 搜索 (anysearch)",
  deepseek: "DeepSeek 搜索 (deepseek)",
  serpapi: "SerpApi (Google) 搜索 (serpapi)",
};

/* ---------- Pure Exported Schema & Widget Helpers ---------- */

export function formatEnumLabel(value) {
  const key = String(value);
  return Object.prototype.hasOwnProperty.call(ENUM_LABELS, key)
    ? ENUM_LABELS[key]
    : key;
}

export function localizeValidationError(err) {
  const rawMsg = String(err?.msg ?? "").trim();
  const errType = String(err?.type ?? "");

  if (rawMsg && CJK_RE.test(rawMsg)) {
    return rawMsg;
  }

  if (errType === "extra_forbidden") {
    return "不允许包含未定义的配置项";
  }
  if (errType === "missing") {
    return "缺少必填配置项";
  }
  if (
    errType === "int_parsing" ||
    errType === "int_type" ||
    errType === "int_from_float"
  ) {
    return "请输入有效的整数";
  }
  if (errType === "float_parsing" || errType === "float_type") {
    return "请输入有效的数字";
  }
  if (errType === "finite_number") {
    return "必须是有限数字（不允许 NaN 或 Infinity）";
  }
  if (errType === "bool_parsing" || errType === "bool_type") {
    return "必须是布尔值（启用或关闭）";
  }
  if (errType === "string_type") {
    return "必须是字符串";
  }
  if (errType === "path_type") {
    return "请输入有效的文件路径";
  }
  if (errType === "list_type") {
    return "必须是列表";
  }
  if (errType === "dict_type" || errType === "model_type") {
    return "必须是 JSON 对象";
  }
  if (errType === "literal_error") {
    const match = rawMsg.match(/^Input should be (.+)$/);
    if (match) {
      const formatted = match[1]
        .replace(/,\s+/g, "、")
        .replace(/\s+or\s+/g, " 或 ");
      return `取值无效，可选值：${formatted}`;
    }
    return "取值不在允许的枚举范围内";
  }
  if (errType === "greater_than") {
    const match = rawMsg.match(/greater than ([^\s]+)/);
    return match ? `数值必须大于 ${match[1]}` : "数值过小";
  }
  if (errType === "greater_than_equal") {
    const match = rawMsg.match(/greater than or equal to ([^\s]+)/);
    return match ? `数值必须大于等于 ${match[1]}` : "数值过小";
  }
  if (errType === "less_than") {
    const match = rawMsg.match(/less than ([^\s]+)/);
    return match ? `数值必须小于 ${match[1]}` : "数值过大";
  }
  if (errType === "less_than_equal") {
    const match = rawMsg.match(/less than or equal to ([^\s]+)/);
    return match ? `数值必须小于等于 ${match[1]}` : "数值过大";
  }
  if (errType === "value_error") {
    if (rawMsg.includes("Invalid IANA timezone")) {
      return "无效的 IANA 时区名称（例如 Asia/Shanghai、UTC）";
    }
    if (rawMsg.includes("window_keep_messages must be strictly less than")) {
      return "滚动后保留消息数必须严格小于上下文窗口最大消息数";
    }
    const cleaned = rawMsg.replace(/^Value error,\s*/i, "");
    return cleaned ? `值校验失败：${cleaned}` : "值校验失败";
  }
  return rawMsg ? `校验失败：${rawMsg}` : "配置值无效";
}

export function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

export function pathKey(path) {
  return path.map(String).join(".");
}

export function getValueAtPath(obj, path) {
  let cur = obj;
  for (const key of path) {
    if (cur == null || typeof cur !== "object") {
      return undefined;
    }
    cur = cur[key];
  }
  return cur;
}

export function setValueAtPath(obj, path, value) {
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

export function resolveRef(node, rootSchema) {
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

export function unwrapSchema(rawNode, rootSchema) {
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

export function defaultValueForNonScalarNullable(schema, rootSchema) {
  if (schema?.default !== undefined && schema?.default !== null) {
    return deepClone(schema.default);
  }
  if (schema?.type === "array") {
    return [];
  }
  if (schema?.type === "object") {
    if (schema.properties && typeof schema.properties === "object") {
      const out = {};
      for (const [key, rawChild] of Object.entries(schema.properties)) {
        const { schema: childSchema } = unwrapSchema(rawChild, rootSchema);
        if (childSchema.default !== undefined) {
          out[key] = deepClone(childSchema.default);
        }
      }
      return out;
    }
    return {};
  }
  return {};
}

export function parseNumberInput(
  rawValue,
  schemaType,
  { nullable = false, badInput = false } = {}
) {
  if (badInput) {
    return { ok: false, error: "请输入有效的数字。" };
  }
  const trimmed = String(rawValue ?? "").trim();
  if (trimmed === "") {
    if (nullable) {
      return { ok: true, value: null };
    }
    return { ok: false, error: "此项为必填数字。" };
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    return { ok: false, error: "必须是有限数字。" };
  }
  if (schemaType === "integer" && !Number.isInteger(parsed)) {
    return { ok: false, error: "必须是整数。" };
  }
  return { ok: true, value: parsed };
}

export function parseListItemValue(rawValue, itemType) {
  if (itemType === "integer" || itemType === "number") {
    return parseNumberInput(rawValue, itemType, { nullable: false, badInput: false });
  }
  return { ok: true, value: String(rawValue ?? "") };
}

export function serializeStringInput(rawValue, nullable) {
  const str = String(rawValue ?? "");
  return nullable && str === "" ? null : str;
}

export function serializeNullableEnumSelect(selectedIndex, enumValues) {
  if (selectedIndex <= 0) {
    return null;
  }
  return enumValues[selectedIndex - 1];
}

export function serializeNullableBoolSelect(selectedValue) {
  if (selectedValue === "") {
    return null;
  }
  return selectedValue === "true";
}

export function validateKeyValueEntries(entries) {
  const out = {};
  for (const entry of entries) {
    const key = String(entry?.key ?? "").trim();
    const val = String(entry?.value ?? "");
    if (!key) {
      if (val !== "") {
        return {
          ok: false,
          error: "已填写键值时，键名不能为空。",
        };
      }
      continue;
    }
    if (Object.prototype.hasOwnProperty.call(out, key)) {
      return {
        ok: false,
        error: `键名「${key}」重复。`,
      };
    }
    out[key] = val;
  }
  return { ok: true, value: out };
}

export function getEffectiveValue(draftConfig, defaultsConfig, path, schema) {
  const draftVal = getValueAtPath(draftConfig, path);
  if (draftVal !== undefined) {
    return draftVal;
  }
  const defaultVal = getValueAtPath(defaultsConfig, path);
  if (defaultVal !== undefined) {
    return defaultVal;
  }
  if (schema && schema.default !== undefined) {
    return schema.default;
  }
  return undefined;
}

export function collectRootAndUnknownErrors(errorsByPath, rootSchema) {
  const knownSections = new Set(Object.keys(rootSchema?.properties || {}));
  const messages = [];
  for (const [key, list] of errorsByPath.entries()) {
    if (key === "") {
      messages.push(...list);
    } else {
      const top = key.split(".")[0];
      if (!knownSections.has(top)) {
        for (const msg of list) {
          messages.push(`${key}: ${msg}`);
        }
      }
    }
  }
  return messages;
}

function isRenderedFieldPath(restSegments, parentSchema, rootSchema) {
  if (restSegments.length === 0) {
    return false;
  }
  const [head, ...tail] = restSegments;
  const props = parentSchema?.properties;
  if (!props || !Object.prototype.hasOwnProperty.call(props, head)) {
    return false;
  }
  const { schema: childSchema } = unwrapSchema(props[head], rootSchema);
  if (childSchema.type === "object" && childSchema.properties) {
    if (tail.length === 0) {
      return true;
    }
    return isRenderedFieldPath(tail, childSchema, rootSchema);
  }
  // Leaf field (scalar, array, or additionalProperties object) displays itself and descendants
  return true;
}

export function collectUnmatchedSectionErrors(
  sectionKey,
  sectionSchema,
  rootSchema,
  errorsByPath
) {
  const messages = [];
  for (const [key, list] of errorsByPath.entries()) {
    if (key === sectionKey) {
      messages.push(...list);
    } else if (key.startsWith(`${sectionKey}.`)) {
      const rest = key.slice(sectionKey.length + 1).split(".");
      if (!isRenderedFieldPath(rest, sectionSchema, rootSchema)) {
        for (const msg of list) {
          messages.push(`${key}: ${msg}`);
        }
      }
    }
  }
  return messages;
}

export function collectLeafFieldErrors(
  fieldPathKey,
  errorsByPath,
  widgetParseErrors,
  includeDescendants = true
) {
  const messages = [];
  const prefix = `${fieldPathKey}.`;

  for (const [key, list] of errorsByPath.entries()) {
    if (key === fieldPathKey) {
      messages.push(...list);
    } else if (includeDescendants && key.startsWith(prefix)) {
      const sub = key.slice(prefix.length);
      for (const msg of list) {
        messages.push(`[${sub}] ${msg}`);
      }
    }
  }

  for (const [key, msg] of widgetParseErrors.entries()) {
    if (key === fieldPathKey) {
      messages.push(msg);
    } else if (includeDescendants && key.startsWith(prefix)) {
      const sub = key.slice(prefix.length);
      messages.push(`[${sub}] ${msg}`);
    }
  }

  return messages;
}

/* ---------- Browser Application State & UI ---------- */

const state = {
  meta: null,
  schema: null,
  defaultsConfig: null,
  savedConfig: null,
  draftConfig: null,
  configExists: true,
  corruptedRecovery: null, // { detail, errors } when disk config is invalid
  activeSection: null,
  viewMode: "visual", // "visual" | "raw"
  searchQuery: "",
  errorsByPath: new Map(), // "a.b.c" -> string[]
  widgetParseErrors: new Map(), // "a.b.c" -> string (live local widget error)
  widgetRawDrafts: new Map(), // "a.b.c" -> raw string / entries preserved across rerenders
  rawErrors: [],
  saving: false,
  restartRequired: false,
  dirty: false,
};

let els = {};

function initElements() {
  els = {
    topbar: document.querySelector(".topbar"),
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
}

function clearLocalWidgetState() {
  state.widgetParseErrors.clear();
  state.widgetRawDrafts.clear();
}

function clearErrorsForPrefix(prefixPath) {
  const dot = `${prefixPath}.`;
  for (const key of Array.from(state.errorsByPath.keys())) {
    if (key === prefixPath || key.startsWith(dot)) {
      state.errorsByPath.delete(key);
    }
  }
}

function clearWidgetStateForPrefix(prefixPath) {
  const dot = `${prefixPath}.`;
  for (const key of Array.from(state.widgetParseErrors.keys())) {
    if (key === prefixPath || key.startsWith(dot)) {
      state.widgetParseErrors.delete(key);
    }
  }
  for (const key of Array.from(state.widgetRawDrafts.keys())) {
    if (key === prefixPath || key.startsWith(dot)) {
      state.widgetRawDrafts.delete(key);
    }
  }
}

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
    theme === "dark" ? "切换为浅色主题" : "切换为深色主题";
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
  els.authToggle.textContent = "显示";
  els.authToggle.setAttribute("aria-label", "显示令牌");
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

/* ---------- Secret & Constraint Helpers ---------- */

function isSecretField(fieldKey) {
  return SECRET_SCALAR_RE.test(fieldKey);
}

function isSecretListField(fieldKey) {
  return SECRET_LIST_RE.test(fieldKey);
}

function buildConstraintHint(schema) {
  const parts = [];
  if (schema.minimum !== undefined && schema.maximum !== undefined) {
    parts.push(`${schema.minimum} ~ ${schema.maximum}`);
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
    parts.push("路径");
  }
  return parts.join("，");
}

/* ---------- Validation Error Mapping ---------- */

function setValidationErrors(errors) {
  state.rawErrors = Array.isArray(errors) ? errors : [];
  state.errorsByPath.clear();
  for (const err of state.rawErrors) {
    const loc = Array.isArray(err.loc) ? err.loc : [];
    const key = pathKey(loc);
    const list = state.errorsByPath.get(key) || [];
    list.push(localizeValidationError(err));
    state.errorsByPath.set(key, list);
  }
}

function formatRawErrorsSummary(errors) {
  return (Array.isArray(errors) ? errors : [])
    .map((e) => `${(e.loc || []).join(".") || "根配置"}：${localizeValidationError(e)}`)
    .join("\n");
}

function countSectionErrors(sectionKey) {
  let count = 0;
  for (const [key, messages] of state.errorsByPath.entries()) {
    if (key === sectionKey || key.startsWith(`${sectionKey}.`)) {
      count += messages.length;
    }
  }
  for (const key of state.widgetParseErrors.keys()) {
    if (key === sectionKey || key.startsWith(`${sectionKey}.`)) {
      count += 1;
    }
  }
  return count;
}

function focusFirstWidgetError() {
  if (state.widgetParseErrors.size === 0) {
    return null;
  }
  const [firstPath, firstMsg] = state.widgetParseErrors.entries().next().value;
  const topSection = firstPath.split(".")[0];
  if (topSection && state.schema?.properties?.[topSection]) {
    state.activeSection = topSection;
    if (state.searchQuery) {
      state.searchQuery = "";
      els.searchInput.value = "";
      els.searchClear.classList.add("hidden");
    }
    renderSidebar();
    renderForm();
  }
  return { path: firstPath, message: firstMsg };
}

/* ---------- Dirty Tracking ---------- */

function recomputeDirty() {
  if (state.corruptedRecovery || state.widgetParseErrors.size > 0) {
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
    strong.textContent = "配置恢复模式：";
    msgWrap.appendChild(strong);
    msgWrap.appendChild(
      document.createTextNode(
        state.corruptedRecovery.detail ||
          "现有配置文件无效，请在「原始 JSON」视图中修复或恢复默认配置。"
      )
    );

    const actions = document.createElement("div");
    actions.className = "banner-actions";
    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "btn btn-secondary btn-sm";
    resetBtn.textContent = "恢复默认";
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
    text.textContent = `配置文件 ${state.meta?.config_path || "config.json"} 尚不存在，当前显示默认配置，点击「保存」即可创建。`;
    banner.appendChild(text);
    els.bannerStack.appendChild(banner);
  }

  if (state.restartRequired) {
    const banner = document.createElement("div");
    banner.className = "banner banner-success";
    const text = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = "配置已保存。";
    text.appendChild(strong);
    text.appendChild(
      document.createTextNode(
        `配置已保存到 ${state.meta?.config_path || "config.json"}，重启 Spectra-bot 后生效。`
      )
    );

    const dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.className = "btn btn-secondary btn-sm";
    dismiss.textContent = "知道了";
    dismiss.addEventListener("click", () => {
      state.restartRequired = false;
      renderBanners();
    });

    banner.append(text, dismiss);
    els.bannerStack.appendChild(banner);
  }

  const rootErrors = collectRootAndUnknownErrors(state.errorsByPath, state.schema);
  if (rootErrors.length > 0 && !state.corruptedRecovery) {
    const banner = document.createElement("div");
    banner.className = "banner banner-danger";
    const text = document.createElement("div");
    text.textContent = `配置校验错误：${rootErrors.join(" · ")}`;
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
  if (Array.isArray(fieldSchema.enum)) {
    for (const optionVal of fieldSchema.enum) {
      haystacks.push(String(optionVal), formatEnumLabel(optionVal));
    }
  } else if (fieldSchema.const !== undefined) {
    haystacks.push(String(fieldSchema.const), formatEnumLabel(fieldSchema.const));
  }
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
      errBadge.title = `${errorCount} 处校验错误`;
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
  const fullPath = pathKey(path);
  clearErrorsForPrefix(fullPath);
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

function renderListWidget(path, fieldKey, value, itemType) {
  const fullPath = pathKey(path);
  const isNumeric = itemType === "integer" || itemType === "number";
  const maskItems = !isNumeric && isSecretListField(fieldKey);

  const preservedDraft = state.widgetRawDrafts.get(fullPath);
  const rawItems = Array.isArray(preservedDraft)
    ? [...preservedDraft]
    : (Array.isArray(value) ? value : []).map((v) => (v == null ? "" : String(v)));

  const wrap = document.createElement("div");
  wrap.className = "list-widget";

  const parseErr = document.createElement("div");
  parseErr.className = "inline-error hidden";
  parseErr.setAttribute("role", "alert");

  function syncList() {
    if (!isNumeric) {
      state.widgetParseErrors.delete(fullPath);
      state.widgetRawDrafts.delete(fullPath);
      parseErr.classList.add("hidden");
      onFieldMutated(path, [...rawItems]);
      return;
    }

    const parsedValues = [];
    for (let i = 0; i < rawItems.length; i += 1) {
      const res = parseListItemValue(rawItems[i], itemType);
      if (!res.ok) {
        const msg = `第 ${i + 1} 项：${res.error}`;
        state.widgetParseErrors.set(fullPath, msg);
        state.widgetRawDrafts.set(fullPath, [...rawItems]);
        parseErr.textContent = msg;
        parseErr.classList.remove("hidden");
        recomputeDirty();
        renderSidebar();
        return;
      }
      parsedValues.push(res.value);
    }

    state.widgetParseErrors.delete(fullPath);
    state.widgetRawDrafts.delete(fullPath);
    parseErr.classList.add("hidden");
    onFieldMutated(path, parsedValues);
  }

  function rerender() {
    wrap.textContent = "";
    if (rawItems.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-hint";
      empty.textContent = "暂无条目";
      wrap.appendChild(empty);
    }

    rawItems.forEach((item, idx) => {
      const row = document.createElement("div");
      row.className = "list-item";

      const input = document.createElement("input");
      input.className = "text-input";
      input.type = maskItems ? "password" : isNumeric ? "number" : "text";
      if (isNumeric) {
        input.step = itemType === "integer" ? "1" : "any";
      }
      input.value = item;
      input.addEventListener("input", () => {
        if (isNumeric && input.validity?.badInput) {
          const msg = `第 ${idx + 1} 项：请输入有效的数字。`;
          state.widgetParseErrors.set(fullPath, msg);
          state.widgetRawDrafts.set(fullPath, [...rawItems]);
          parseErr.textContent = msg;
          parseErr.classList.remove("hidden");
          recomputeDirty();
          renderSidebar();
          return;
        }
        rawItems[idx] = input.value;
        syncList();
      });
      row.appendChild(input);

      if (maskItems) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "icon-btn-inline";
        toggle.textContent = "显示";
        toggle.addEventListener("click", () => {
          const showing = input.type === "text";
          input.type = showing ? "password" : "text";
          toggle.textContent = showing ? "显示" : "隐藏";
        });
        row.appendChild(toggle);
      }

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "btn btn-danger-ghost";
      removeBtn.textContent = "删除";
      removeBtn.addEventListener("click", () => {
        rawItems.splice(idx, 1);
        syncList();
        rerender();
      });
      row.appendChild(removeBtn);

      wrap.appendChild(row);
    });

    const addRow = document.createElement("div");
    addRow.className = "list-add-row";

    const addInput = document.createElement("input");
    addInput.className = "text-input";
    addInput.type = maskItems ? "password" : isNumeric ? "number" : "text";
    if (isNumeric) {
      addInput.step = itemType === "integer" ? "1" : "any";
    }
    addInput.placeholder = maskItems
      ? "输入密钥..."
      : isNumeric
        ? itemType === "integer"
          ? "输入整数..."
          : "输入数字..."
        : "输入新项...";

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "btn btn-secondary btn-sm";
    addBtn.textContent = "+ 添加";

    const commitAdd = () => {
      if (isNumeric) {
        if (addInput.validity?.badInput) {
          parseErr.textContent = "请先输入有效的数字再添加。";
          parseErr.classList.remove("hidden");
          return;
        }
        const res = parseListItemValue(addInput.value, itemType);
        if (!res.ok) {
          parseErr.textContent = res.error;
          parseErr.classList.remove("hidden");
          return;
        }
        rawItems.push(String(res.value));
      } else {
        const trimmed = addInput.value.trim();
        if (!trimmed) return;
        rawItems.push(trimmed);
      }
      addInput.value = "";
      syncList();
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

    const existingErr = state.widgetParseErrors.get(fullPath);
    if (existingErr) {
      parseErr.textContent = existingErr;
      parseErr.classList.remove("hidden");
    } else {
      parseErr.classList.add("hidden");
    }
    wrap.appendChild(parseErr);
  }

  rerender();
  return wrap;
}

function renderKeyValueWidget(path, value) {
  const fullPath = pathKey(path);
  const preserved = state.widgetRawDrafts.get(fullPath);
  const entries = Array.isArray(preserved)
    ? preserved.map((e) => ({ key: e.key, value: e.value }))
    : Object.entries(
        value && typeof value === "object" && !Array.isArray(value) ? value : {}
      ).map(([k, v]) => ({ key: k, value: v == null ? "" : String(v) }));

  const wrap = document.createElement("div");
  wrap.className = "list-widget";

  const parseErr = document.createElement("div");
  parseErr.className = "inline-error hidden";
  parseErr.setAttribute("role", "alert");

  function syncObject() {
    const validated = validateKeyValueEntries(entries);
    if (!validated.ok) {
      state.widgetParseErrors.set(fullPath, validated.error);
      state.widgetRawDrafts.set(
        fullPath,
        entries.map((e) => ({ key: e.key, value: e.value }))
      );
      parseErr.textContent = validated.error;
      parseErr.classList.remove("hidden");
      recomputeDirty();
      renderSidebar();
      return;
    }
    state.widgetParseErrors.delete(fullPath);
    state.widgetRawDrafts.delete(fullPath);
    parseErr.classList.add("hidden");
    onFieldMutated(path, validated.value);
  }

  function rerender() {
    wrap.textContent = "";
    if (entries.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-hint";
      empty.textContent = "暂无条目";
      wrap.appendChild(empty);
    }

    entries.forEach((entry, idx) => {
      const row = document.createElement("div");
      row.className = "kv-row";

      const keyInput = document.createElement("input");
      keyInput.className = "text-input mono";
      keyInput.type = "text";
      keyInput.placeholder = "键名";
      keyInput.value = entry.key;

      const valWrap = document.createElement("div");
      valWrap.className = "secret-input-wrap";

      const valInput = document.createElement("input");
      valInput.className = "text-input";
      const looksSecret = SECRET_HEADER_KEY_RE.test(entry.key);
      valInput.type = looksSecret ? "password" : "text";
      valInput.placeholder = "键值";
      valInput.value = entry.value;

      valWrap.appendChild(valInput);

      if (looksSecret) {
        const toggleBtn = document.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "icon-btn-inline";
        toggleBtn.textContent = "显示";
        toggleBtn.addEventListener("click", () => {
          const showing = valInput.type === "text";
          valInput.type = showing ? "password" : "text";
          toggleBtn.textContent = showing ? "显示" : "隐藏";
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
      removeBtn.textContent = "删除";
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
    addBtn.textContent = "+ 添加键值";
    addBtn.addEventListener("click", () => {
      entries.push({ key: "", value: "" });
      rerender();
    });
    wrap.appendChild(addBtn);

    const existingErr = state.widgetParseErrors.get(fullPath);
    if (existingErr) {
      parseErr.textContent = existingErr;
      parseErr.classList.remove("hidden");
    } else {
      parseErr.classList.add("hidden");
    }
    wrap.appendChild(parseErr);
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

  const preservedText = state.widgetRawDrafts.get(fieldKey);
  textarea.value =
    typeof preservedText === "string"
      ? preservedText
      : JSON.stringify(value && typeof value === "object" ? value : {}, null, 2);

  const parseErr = document.createElement("div");
  parseErr.className = "inline-error hidden";
  parseErr.setAttribute("role", "alert");

  const existingErr = state.widgetParseErrors.get(fieldKey);
  if (existingErr) {
    parseErr.textContent = existingErr;
    parseErr.classList.remove("hidden");
    textarea.classList.add("input-error");
  }

  textarea.addEventListener("input", () => {
    const raw = textarea.value.trim();
    if (!raw) {
      state.widgetParseErrors.delete(fieldKey);
      state.widgetRawDrafts.delete(fieldKey);
      parseErr.classList.add("hidden");
      textarea.classList.remove("input-error");
      onFieldMutated(path, {});
      return;
    }
    try {
      const parsed = JSON.parse(raw);
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("必须是 JSON 对象（{ ... }）。");
      }
      state.widgetParseErrors.delete(fieldKey);
      state.widgetRawDrafts.delete(fieldKey);
      parseErr.classList.add("hidden");
      textarea.classList.remove("input-error");
      onFieldMutated(path, parsed);
    } catch (err) {
      const msg =
        err instanceof Error && CJK_RE.test(err.message)
          ? err.message
          : `JSON 格式无效：${err instanceof Error ? err.message : "无法解析"}`;
      state.widgetParseErrors.set(fieldKey, msg);
      state.widgetRawDrafts.set(fieldKey, textarea.value);
      parseErr.textContent = msg;
      parseErr.classList.remove("hidden");
      textarea.classList.add("input-error");
      recomputeDirty();
      renderSidebar();
    }
  });

  wrap.append(textarea, parseErr);
  return wrap;
}

function renderField(path, fieldKey, rawFieldSchema) {
  const { schema, nullable } = unwrapSchema(rawFieldSchema, state.schema);
  const fullPathKey = pathKey(path);
  const currentValue = getEffectiveValue(
    state.draftConfig,
    state.defaultsConfig,
    path,
    schema
  );

  // Nested sub-model object with declared properties
  if (schema.type === "object" && schema.properties) {
    const nestedErrors = collectLeafFieldErrors(
      fullPathKey,
      state.errorsByPath,
      state.widgetParseErrors,
      false
    );
    const unmatchedNested = collectUnmatchedSectionErrors(
      fullPathKey,
      schema,
      state.schema,
      state.errorsByPath
    );
    const headerErrors = Array.from(new Set([...nestedErrors, ...unmatchedNested]));

    const card = document.createElement("div");
    card.className = "nested-card";
    if (headerErrors.length > 0) {
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

    if (nullable) {
      const nullRow = document.createElement("label");
      nullRow.className = "nullable-row";
      const nullCheckbox = document.createElement("input");
      nullCheckbox.type = "checkbox";
      nullCheckbox.checked = currentValue === null || currentValue === undefined;
      nullCheckbox.addEventListener("change", () => {
        clearWidgetStateForPrefix(fullPathKey);
        if (nullCheckbox.checked) {
          onFieldMutated(path, null);
        } else {
          onFieldMutated(path, defaultValueForNonScalarNullable(schema, state.schema));
        }
        renderForm();
      });
      nullRow.append(nullCheckbox, document.createTextNode("设为未设置 (null)"));
      header.appendChild(nullRow);
    }

    if (headerErrors.length > 0) {
      header.appendChild(createErrorList(headerErrors));
    }
    card.appendChild(header);

    if (!nullable || (currentValue !== null && currentValue !== undefined)) {
      const children = document.createElement("div");
      children.className = "nested-children";
      for (const [childKey, childSchema] of Object.entries(schema.properties)) {
        children.appendChild(renderField([...path, childKey], childKey, childSchema));
      }
      card.appendChild(children);
    }
    return card;
  }

  const serverFieldErrors = collectLeafFieldErrors(
    fullPathKey,
    state.errorsByPath,
    new Map(),
    true
  );
  const hasAnyError =
    serverFieldErrors.length > 0 || state.widgetParseErrors.has(fullPathKey);

  const card = document.createElement("div");
  card.className = "field-card";
  card.dataset.path = fullPathKey;
  if (hasAnyError) {
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

  // Non-nullable boolean -> inline switch on the right
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
    if (serverFieldErrors.length > 0) {
      card.appendChild(createErrorList(serverFieldErrors));
    }
    return card;
  }

  card.appendChild(info);

  const control = document.createElement("div");
  control.className = "field-control";

  // Nullable non-scalar (array / additionalProperties object) -> safe unset toggle
  const isNonScalar = schema.type === "array" || schema.type === "object";
  if (nullable && isNonScalar) {
    const nullRow = document.createElement("label");
    nullRow.className = "nullable-row";
    const nullCheckbox = document.createElement("input");
    nullCheckbox.type = "checkbox";
    nullCheckbox.checked = currentValue === null || currentValue === undefined;
    nullCheckbox.addEventListener("change", () => {
      clearWidgetStateForPrefix(fullPathKey);
      if (nullCheckbox.checked) {
        onFieldMutated(path, null);
      } else {
        onFieldMutated(path, defaultValueForNonScalarNullable(schema, state.schema));
      }
      renderForm();
    });
    nullRow.append(nullCheckbox, document.createTextNode("设为未设置 (null)"));
    control.appendChild(nullRow);

    if (currentValue === null || currentValue === undefined) {
      card.appendChild(control);
      if (serverFieldErrors.length > 0) {
        card.appendChild(createErrorList(serverFieldErrors));
      }
      return card;
    }
  }

  // Nullable boolean -> select with 未设置 (null) / 启用 / 关闭
  if (schema.type === "boolean" && nullable) {
    const select = document.createElement("select");
    select.id = inputId;
    select.className = "select-input";
    if (hasAnyError) select.classList.add("input-error");

    const options = [
      { value: "", label: "— 未设置 (null) —" },
      { value: "true", label: "启用 (true)" },
      { value: "false", label: "关闭 (false)" },
    ];
    for (const item of options) {
      const opt = document.createElement("option");
      opt.value = item.value;
      opt.textContent = item.label;
      if (
        (item.value === "" && (currentValue === null || currentValue === undefined)) ||
        (item.value === "true" && currentValue === true) ||
        (item.value === "false" && currentValue === false)
      ) {
        opt.selected = true;
      }
      select.appendChild(opt);
    }
    select.addEventListener("change", () => {
      onFieldMutated(path, serializeNullableBoolSelect(select.value));
    });
    control.appendChild(select);
  }
  // Const (single literal)
  else if (schema.const !== undefined && !nullable) {
    const input = document.createElement("input");
    input.id = inputId;
    input.className = "text-input";
    input.type = "text";
    input.value = formatEnumLabel(schema.const);
    input.readOnly = true;
    input.disabled = true;
    control.appendChild(input);
  }
  // Enum (or nullable const) -> select
  else if (Array.isArray(schema.enum) || (schema.const !== undefined && nullable)) {
    const enumValues = Array.isArray(schema.enum) ? schema.enum : [schema.const];
    const select = document.createElement("select");
    select.id = inputId;
    select.className = "select-input";
    if (hasAnyError) select.classList.add("input-error");

    if (nullable) {
      const unsetOpt = document.createElement("option");
      unsetOpt.value = "";
      unsetOpt.textContent = "— 未设置 (null) —";
      if (currentValue === null || currentValue === undefined) {
        unsetOpt.selected = true;
      }
      select.appendChild(unsetOpt);
    }

    for (const optionVal of enumValues) {
      const opt = document.createElement("option");
      opt.value = String(optionVal);
      opt.textContent = formatEnumLabel(optionVal);
      if (
        currentValue !== null &&
        currentValue !== undefined &&
        String(currentValue) === String(optionVal)
      ) {
        opt.selected = true;
      }
      select.appendChild(opt);
    }

    select.addEventListener("change", () => {
      if (nullable) {
        onFieldMutated(
          path,
          serializeNullableEnumSelect(select.selectedIndex, enumValues)
        );
      } else {
        const matched = enumValues[select.selectedIndex];
        onFieldMutated(path, matched !== undefined ? matched : select.value);
      }
    });
    control.appendChild(select);
  }
  // Integer / Number
  else if (schema.type === "integer" || schema.type === "number") {
    const input = document.createElement("input");
    input.id = inputId;
    input.className = "text-input";
    if (hasAnyError) input.classList.add("input-error");
    input.type = "number";
    input.step = schema.type === "integer" ? "1" : "any";
    if (schema.minimum !== undefined) input.min = String(schema.minimum);
    if (schema.maximum !== undefined) input.max = String(schema.maximum);
    if (nullable) input.placeholder = "未设置 (null)";

    const preservedText = state.widgetRawDrafts.get(fullPathKey);
    input.value =
      typeof preservedText === "string"
        ? preservedText
        : currentValue == null
          ? ""
          : String(currentValue);

    const parseErr = document.createElement("div");
    parseErr.className = "inline-error hidden";
    parseErr.setAttribute("role", "alert");

    const existingErr = state.widgetParseErrors.get(fullPathKey);
    if (existingErr) {
      parseErr.textContent = existingErr;
      parseErr.classList.remove("hidden");
    }

    input.addEventListener("input", () => {
      const res = parseNumberInput(input.value, schema.type, {
        nullable,
        badInput: Boolean(input.validity?.badInput),
      });
      if (!res.ok) {
        state.widgetParseErrors.set(fullPathKey, res.error);
        state.widgetRawDrafts.set(fullPathKey, input.value);
        parseErr.textContent = res.error;
        parseErr.classList.remove("hidden");
        input.classList.add("input-error");
        card.classList.add("has-error");
        recomputeDirty();
        renderSidebar();
        return;
      }
      state.widgetParseErrors.delete(fullPathKey);
      state.widgetRawDrafts.delete(fullPathKey);
      parseErr.classList.add("hidden");
      input.classList.remove("input-error");
      if (serverFieldErrors.length === 0) {
        card.classList.remove("has-error");
      }
      onFieldMutated(path, res.value);
    });
    control.append(input, parseErr);
  }
  // Array
  else if (schema.type === "array") {
    const { schema: itemSchema } = unwrapSchema(schema.items || {}, state.schema);
    const itemType = itemSchema.type || "string";
    control.appendChild(renderListWidget(path, fieldKey, currentValue, itemType));
  }
  // Object with additionalProperties (dict[str, str] vs dict[str, Any])
  else if (schema.type === "object") {
    const addProps = schema.additionalProperties;
    const isStringMap =
      addProps &&
      typeof addProps === "object" &&
      unwrapSchema(addProps, state.schema).schema.type === "string";
    if (isStringMap) {
      control.appendChild(renderKeyValueWidget(path, currentValue));
    } else {
      control.appendChild(renderJsonObjectWidget(path, currentValue));
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
      if (hasAnyError) input.classList.add("input-error");
      input.type = "password";
      input.autocomplete = "off";
      if (nullable) input.placeholder = "未设置 (null)";
      input.value = strVal;
      input.addEventListener("input", () => {
        onFieldMutated(path, serializeStringInput(input.value, nullable));
      });

      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "icon-btn-inline";
      toggle.textContent = "显示";
      toggle.addEventListener("click", () => {
        const showing = input.type === "text";
        input.type = showing ? "password" : "text";
        toggle.textContent = showing ? "显示" : "隐藏";
      });

      secretWrap.append(input, toggle);
      control.appendChild(secretWrap);
    } else if (fieldKey === "prompt" || strVal.includes("\n") || strVal.length > 100) {
      const textarea = document.createElement("textarea");
      textarea.id = inputId;
      textarea.className = "textarea-input";
      if (hasAnyError) textarea.classList.add("input-error");
      if (nullable) textarea.placeholder = "未设置 (null)";
      textarea.value = strVal;
      textarea.addEventListener("input", () => {
        onFieldMutated(path, serializeStringInput(textarea.value, nullable));
      });
      control.appendChild(textarea);
    } else {
      const input = document.createElement("input");
      input.id = inputId;
      input.className = "text-input";
      if (hasAnyError) input.classList.add("input-error");
      input.type = "text";
      if (nullable) input.placeholder = "未设置 (null)";
      input.value = strVal;
      input.addEventListener("input", () => {
        onFieldMutated(path, serializeStringInput(input.value, nullable));
      });
      control.appendChild(input);
    }
  }

  card.appendChild(control);
  if (serverFieldErrors.length > 0) {
    card.appendChild(createErrorList(serverFieldErrors));
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
    heading.textContent = `搜索结果：「${state.searchQuery}」`;
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
      empty.textContent = "未找到匹配的配置项。";
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

  const sectionErrors = collectUnmatchedSectionErrors(
    activeKey,
    sectionSchema,
    state.schema,
    state.errorsByPath
  );
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
      return { ok: false, error: "根配置必须是 JSON 对象（{ ... }）。" };
    }
    return { ok: true, value: parsed };
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? err.message : "JSON 语法无效。",
    };
  }
}

function applyViewModeDom(targetMode) {
  state.viewMode = targetMode;
  els.modeVisual.classList.toggle("active", targetMode === "visual");
  els.modeVisual.setAttribute("aria-selected", String(targetMode === "visual"));
  els.modeRaw.classList.toggle("active", targetMode === "raw");
  els.modeRaw.setAttribute("aria-selected", String(targetMode === "raw"));
  els.visualView.classList.toggle("hidden", targetMode !== "visual");
  els.rawView.classList.toggle("hidden", targetMode !== "raw");
}

function switchViewMode(targetMode) {
  if (targetMode === state.viewMode) return;

  if (targetMode === "raw") {
    if (state.widgetParseErrors.size > 0) {
      const first = focusFirstWidgetError();
      showToast(
        `请先修正「${first.path}」中的无效输入再切换到原始 JSON：${first.message}`,
        "danger"
      );
      return;
    }
    if (state.draftConfig) {
      els.rawEditor.value = JSON.stringify(state.draftConfig, null, 2);
    }
    setRawError("");
  } else {
    const parsed = parseRawEditorJson();
    if (!parsed.ok) {
      setRawError(`请先修复 JSON 语法错误再切换回表单视图：${parsed.error}`);
      showToast("JSON 格式无效，请先修复语法错误再切换到表单视图。", "danger");
      return;
    }
    clearLocalWidgetState();
    state.draftConfig = parsed.value;
    setRawError("");
    recomputeDirty();
    renderSidebar();
    renderForm();
  }

  applyViewModeDom(targetMode);
}

/* ---------- Actions ---------- */

async function loadMeta() {
  const res = await apiFetch("/api/meta");
  if (!res.ok || !res.data) {
    throw new Error("无法加载服务元信息。");
  }
  state.meta = res.data;
  els.metaVersion.textContent = `v${res.data.version}`;
  els.metaConfigPath.textContent = res.data.config_path;
  els.metaConfigPath.title = res.data.config_path;

  els.metaStatus.classList.remove("running", "standalone");
  if (res.data.bot_running) {
    els.metaStatus.classList.add("running");
    els.metaStatusText.textContent = "随 Bot 运行";
  } else {
    els.metaStatus.classList.add("standalone");
    els.metaStatusText.textContent = "独立 WebUI";
  }

  els.btnAuth.classList.toggle("hidden", !res.data.auth_required);
}

async function loadSchemaAndConfig() {
  const [schemaRes, defaultsRes, configRes] = await Promise.all([
    apiFetch("/api/config/schema"),
    apiFetch("/api/config/defaults"),
    apiFetch("/api/config"),
  ]);

  if (
    schemaRes.status === 401 ||
    defaultsRes.status === 401 ||
    configRes.status === 401
  ) {
    openAuthModal(getAuthToken() ? "令牌无效或已过期，请重试。" : "");
    return false;
  }
  if (!schemaRes.ok || !schemaRes.data) {
    showToast("加载配置结构（Schema）失败。", "danger");
    return false;
  }

  state.schema = schemaRes.data;
  state.defaultsConfig =
    defaultsRes.ok && defaultsRes.data ? deepClone(defaultsRes.data) : null;
  const sectionKeys = Object.keys(state.schema.properties || {});
  if (!state.activeSection || !sectionKeys.includes(state.activeSection)) {
    state.activeSection = sectionKeys[0] || null;
  }

  if (configRes.status === 422 && configRes.data) {
    // Existing file on disk is corrupted or fails schema validation.
    clearLocalWidgetState();
    state.configExists = true;
    state.corruptedRecovery = {
      detail: configRes.data.detail || "配置文件无效。",
      errors: configRes.data.errors || [],
    };
    setValidationErrors(configRes.data.errors || []);
    if (configRes.data.raw_config && typeof configRes.data.raw_config === "object") {
      state.draftConfig = deepClone(configRes.data.raw_config);
      state.savedConfig = null;
    } else {
      state.draftConfig = state.defaultsConfig ? deepClone(state.defaultsConfig) : {};
      state.savedConfig = null;
    }
    els.rawEditor.value =
      typeof configRes.data.raw_text === "string" && configRes.data.raw_text !== ""
        ? configRes.data.raw_text
        : JSON.stringify(state.draftConfig, null, 2);
    applyViewModeDom("raw");
    const errSummary = formatRawErrorsSummary(configRes.data.errors || []);
    setRawError(errSummary || state.corruptedRecovery.detail);
  } else if (configRes.ok && configRes.data) {
    clearLocalWidgetState();
    state.corruptedRecovery = null;
    state.configExists = Boolean(configRes.data.exists);
    state.savedConfig = deepClone(configRes.data.config);
    state.draftConfig = deepClone(configRes.data.config);
    setValidationErrors([]);
    els.rawEditor.value = JSON.stringify(state.draftConfig, null, 2);
    setRawError("");
  } else {
    showToast(configRes.data?.detail || "加载配置文件失败。", "danger");
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
    openAuthModal("需要身份验证。");
    return;
  }
  if (!res.ok || !res.data) {
    showToast("获取默认配置失败。", "danger");
    return;
  }
  clearLocalWidgetState();
  state.corruptedRecovery = null;
  state.defaultsConfig = deepClone(res.data);
  state.draftConfig = deepClone(res.data);
  setValidationErrors([]);
  els.rawEditor.value = JSON.stringify(state.draftConfig, null, 2);
  setRawError("");
  recomputeDirty();
  renderBanners();
  renderSidebar();
  renderForm();
  showToast("已将默认配置载入编辑器（尚未保存）。", "warning");
}

function getPayloadForSubmit() {
  if (state.viewMode === "raw") {
    const parsed = parseRawEditorJson();
    if (!parsed.ok) {
      setRawError(`JSON 语法错误：${parsed.error}`);
      showToast("请先修复 JSON 语法错误后再继续。", "danger");
      return null;
    }
    setRawError("");
    return parsed.value;
  }
  if (state.widgetParseErrors.size > 0) {
    const first = focusFirstWidgetError();
    showToast(
      `请先修正「${first.path}」中的无效输入再保存：${first.message}`,
      "danger"
    );
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
    openAuthModal("需要身份验证。");
    return;
  }
  if (res.ok && res.data?.valid) {
    clearLocalWidgetState();
    setValidationErrors([]);
    setRawError("");
    state.draftConfig = deepClone(res.data.normalized);
    els.rawEditor.value = JSON.stringify(res.data.normalized, null, 2);
    renderBanners();
    renderSidebar();
    showToast("配置校验通过。", "success");
    return;
  }
  const errors = res.data?.errors || [];
  setValidationErrors(errors);
  renderBanners();
  renderSidebar();
  const formatted = formatRawErrorsSummary(errors);
  setRawError(formatted || res.data?.detail || "配置校验失败。");
  showToast(`配置校验失败（共 ${errors.length} 处错误）。`, "danger");
}

async function handleSave() {
  if (state.saving) return;
  const payload = getPayloadForSubmit();
  if (!payload) return;

  state.saving = true;
  els.btnSave.disabled = true;
  els.btnSave.textContent = "保存中...";

  try {
    const res = await apiFetch("/api/config", {
      method: "PUT",
      body: JSON.stringify(payload),
    });

    if (res.status === 401) {
      openAuthModal("访问令牌无效，请重新输入。");
      return;
    }

    if (res.ok && res.data?.saved) {
      clearLocalWidgetState();
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
        `配置已保存到 ${state.meta?.config_path || "config.json"}，重启 Spectra-bot 后生效。`,
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
      const formatted = formatRawErrorsSummary(errors);
      setRawError(formatted || res.data?.detail || "保存配置失败。");
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
        ? `配置校验失败（共 ${errors.length} 处错误）。`
        : res.data?.detail || "保存配置失败。",
      "danger"
    );
  } catch (err) {
    showToast(
      `保存时发生网络错误：${err instanceof Error ? err.message : String(err)}`,
      "danger"
    );
  } finally {
    state.saving = false;
    els.btnSave.disabled = false;
    els.btnSave.textContent = "保存";
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
      showToast("已从磁盘重新加载配置。", "info");
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
      setRawError(`JSON 格式无效，无法格式化：${parsed.error}`);
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
      setRawError(`JSON 语法错误：${parsed.error}`);
    } else {
      setRawError("");
      state.draftConfig = parsed.value;
    }
    recomputeDirty();
  });

  els.authToggle.addEventListener("click", () => {
    const showing = els.authInput.type === "text";
    els.authInput.type = showing ? "password" : "text";
    els.authToggle.textContent = showing ? "显示" : "隐藏";
    els.authToggle.setAttribute("aria-label", showing ? "显示令牌" : "隐藏令牌");
  });

  els.authClearBtn.addEventListener("click", () => {
    setAuthToken("");
    els.authInput.value = "";
    els.authClearBtn.classList.add("hidden");
    showToast("已清除本地保存的访问令牌。", "info");
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
      showToast("身份验证成功。", "success");
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

function syncTopbarHeight() {
  if (!els.topbar) return;
  const height = els.topbar.getBoundingClientRect().height;
  if (height > 0) {
    document.documentElement.style.setProperty("--topbar-height", `${height}px`);
  }
}

function observeTopbarHeight() {
  syncTopbarHeight();
  if (typeof ResizeObserver !== "undefined" && els.topbar) {
    const observer = new ResizeObserver(syncTopbarHeight);
    observer.observe(els.topbar);
  }
  window.addEventListener("resize", syncTopbarHeight);
}

async function bootstrap() {
  initElements();
  observeTopbarHeight();
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
    box.textContent = `无法连接到 Spectra-bot 配置台：${
      err instanceof Error ? err.message : String(err)
    }`;
    els.formRoot.appendChild(box);
  }
}

if (typeof document !== "undefined" && typeof window !== "undefined") {
  void bootstrap();
}
