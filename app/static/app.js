const video = document.getElementById('camera');
const canvas = document.getElementById('canvas');
const preview = document.getElementById('preview');
const cameraMessage = document.getElementById('cameraMessage');
const captureButton = document.getElementById('capture');
const switchCameraButton = document.getElementById('switchCamera');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const scanAgainButton = document.getElementById('scanAgain');
const loading = document.getElementById('loading');
const laserBeam = document.getElementById('laserBeam');
const resultSection = document.getElementById('resultSection');
const historyDrawer = document.getElementById('historyDrawer');
const historyList = document.getElementById('historyList');
const historySearch = document.getElementById('historySearch');
const historyBadge = document.getElementById('historyBadge');
const clearHistoryButton = document.getElementById('clearHistory');
const loadMoreHistoryButton = document.getElementById('loadMoreHistory');
const themeToggle = document.getElementById('themeToggle');
const toast = document.getElementById('toast');
const dropZone = document.getElementById('dropZone');
const catalogGrid = document.getElementById('catalogGrid');
const aiStatus = document.getElementById('aiStatus');
const aiStatusText = document.getElementById('aiStatusText');
const feedbackPanel = document.getElementById('feedbackPanel');
const feedbackCorrectButton = document.getElementById('feedbackCorrect');
const feedbackFixButton = document.getElementById('feedbackFix');
const feedbackCorrection = document.getElementById('feedbackCorrection');
const feedbackCategory = document.getElementById('feedbackCategory');
const feedbackSaveButton = document.getElementById('feedbackSave');
const feedbackStatus = document.getElementById('feedbackStatus');
const learningMemoryBadge = document.getElementById('learningMemoryBadge');
const learningExampleCount = document.getElementById('learningExampleCount');
const historyEditModal = document.getElementById('historyEditModal');
const historyEditImage = document.getElementById('historyEditImage');
const historyEditImagePlaceholder = document.getElementById('historyEditImagePlaceholder');
const historyEditObjectName = document.getElementById('historyEditObjectName');
const historyEditPredicted = document.getElementById('historyEditPredicted');
const historyEditConfidence = document.getElementById('historyEditConfidence');
const historyEditTime = document.getElementById('historyEditTime');
const historyEditCurrentLabel = document.getElementById('historyEditCurrentLabel');
const historyEditCategory = document.getElementById('historyEditCategory');
const historyEditStatus = document.getElementById('historyEditStatus');
const historyEditSaveButton = document.getElementById('historyEditSave');
const historyEditDeleteButton = document.getElementById('historyEditDelete');

let stream = null;
let facingMode = 'environment';
let selectedBlob = null;
let toastTimer = null;
let previewObjectUrl = null;
let isBusy = false;
let isHistoryBusy = false;
let activeRequestController = null;
let requestSequence = 0;
let cameraSequence = 0;
let cachedHistoryItems = [];
let historyCursor = null;
let historyHasMore = false;
let isHistoryLoading = false;
let historyRequestController = null;
let historyRequestSequence = 0;
let historySearchTimer = null;
let filePickerOpening = false;
let cameraRestartTimer = null;
let currentResult = null;
let catalogItems = [];
let feedbackSubmitting = false;
let historyEditItem = null;
const historyThumbnailUrls = new Map();

const HISTORY_PAGE_SIZE = 50;
const CAMERA_TARGET_WIDTH = 960;
const CAMERA_TARGET_HEIGHT = 720;
const MAX_PROCESSING_DIMENSION = 1024;
const UPLOAD_OPTIMIZE_THRESHOLD_BYTES = 1.2 * 1024 * 1024;
const JPEG_QUALITY = 0.82;

const CLIENT_ID_STORAGE_KEY = 'waste-scanner-client-id';
const CLIENT_ID_COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

function isValidClientId(value) {
  const normalized = value?.trim();
  return Boolean(
    normalized
    && normalized.length <= 128
    && !['anonymous', 'legacy'].includes(normalized.toLowerCase())
  );
}

function safeStorageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch (error) {
    console.warn(`Không thể đọc localStorage (${key}):`, error);
    return null;
  }
}

function safeStorageSet(key, value) {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch (error) {
    console.warn(`Không thể ghi localStorage (${key}):`, error);
    return false;
  }
}

function safeCookieGet(name) {
  try {
    const prefix = `${encodeURIComponent(name)}=`;
    const entry = document.cookie
      .split(';')
      .map(item => item.trim())
      .find(item => item.startsWith(prefix));
    return entry ? decodeURIComponent(entry.slice(prefix.length)) : null;
  } catch (error) {
    console.warn(`Không thể đọc cookie (${name}):`, error);
    return null;
  }
}

function safeCookieSet(name, value) {
  try {
    const secure = location.protocol === 'https:' ? '; Secure' : '';
    document.cookie = `${encodeURIComponent(name)}=${encodeURIComponent(value)}; Path=/; Max-Age=${CLIENT_ID_COOKIE_MAX_AGE}; SameSite=Lax${secure}`;
    return safeCookieGet(name) === value;
  } catch (error) {
    console.warn(`Không thể ghi cookie (${name}):`, error);
    return false;
  }
}

function getOrCreateClientId() {
  const storedId = safeStorageGet(CLIENT_ID_STORAGE_KEY)?.trim();
  if (isValidClientId(storedId)) {
    // localStorage is the primary source when available. Mirror it into the
    // cookie so history keeps the same owner if localStorage later becomes blocked.
    const cookieSynced = safeCookieSet(CLIENT_ID_STORAGE_KEY, storedId);
    return { id: storedId, persistent: true, fullySynced: cookieSynced };
  }

  const cookieId = safeCookieGet(CLIENT_ID_STORAGE_KEY)?.trim();
  if (isValidClientId(cookieId)) {
    const storageSynced = safeStorageSet(CLIENT_ID_STORAGE_KEY, cookieId);
    safeCookieSet(CLIENT_ID_STORAGE_KEY, cookieId);
    return { id: cookieId, persistent: true, fullySynced: storageSynced };
  }

  const generatedId = globalThis.crypto?.randomUUID?.()
    || `browser-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;

  const storagePersisted = safeStorageSet(CLIENT_ID_STORAGE_KEY, generatedId);
  const cookiePersisted = safeCookieSet(CLIENT_ID_STORAGE_KEY, generatedId);
  if (storagePersisted || cookiePersisted) {
    return {
      id: generatedId,
      persistent: true,
      fullySynced: storagePersisted && cookiePersisted
    };
  }

  // Last-resort in-memory ID: this identifier is metadata only. Shared history
  // and shared learning continue to work even if it changes after a reload.
  return { id: generatedId, persistent: false, fullySynced: false };
}

const clientIdentity = getOrCreateClientId();
const clientId = clientIdentity.id;
const deviceHeaders = { 'X-Client-ID': clientId };

// --- Theme Switcher ---
function initTheme() {
  const storedTheme = safeStorageGet('theme');
  const savedTheme = storedTheme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  themeToggle.textContent = savedTheme === 'dark' ? '🌙' : '☀️';
}

themeToggle.addEventListener('click', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', nextTheme);
  safeStorageSet('theme', nextTheme);
  themeToggle.textContent = nextTheme === 'dark' ? '🌙' : '☀️';
});

// --- Toast Notification ---
function showToast(message) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.hidden = false;
  toastTimer = setTimeout(() => { toast.hidden = true; }, 4000);
}

// --- Controls Lock State ---
function updateControlState() {
  const locked = isBusy || isHistoryBusy || feedbackSubmitting;
  captureButton.disabled = locked || !stream;
  switchCameraButton.disabled = locked;
  fileInput.disabled = locked;
  if (uploadButton) uploadButton.disabled = locked;
  if (scanAgainButton) scanAgainButton.disabled = locked;
  if (clearHistoryButton) clearHistoryButton.disabled = locked;
  if (loadMoreHistoryButton) loadMoreHistoryButton.disabled = locked || isHistoryLoading;
  const feedbackLocked = locked || feedbackSubmitting || !currentResult?.scan_id;
  if (feedbackCorrectButton) feedbackCorrectButton.disabled = feedbackLocked;
  if (feedbackFixButton) feedbackFixButton.disabled = feedbackLocked;
  if (feedbackSaveButton) feedbackSaveButton.disabled = feedbackLocked;
  if (feedbackCategory) feedbackCategory.disabled = feedbackLocked;
  const historyEditLocked = locked || feedbackSubmitting || !historyEditItem;
  if (historyEditSaveButton) historyEditSaveButton.disabled = historyEditLocked;
  if (historyEditDeleteButton) historyEditDeleteButton.disabled = historyEditLocked;
  if (historyEditCategory) historyEditCategory.disabled = historyEditLocked;
}

function setBusy(value) {
  // Keep the processing overlay fully controlled by an explicit user action.
  // The inline display value is intentional: it prevents stale/cached CSS from
  // making the overlay visible when the page first opens.
  isBusy = Boolean(value);
  if (loading) {
    loading.hidden = !isBusy;
    loading.setAttribute('aria-hidden', String(!isBusy));
    loading.style.display = isBusy ? 'flex' : 'none';
  }
  if (laserBeam) laserBeam.hidden = !isBusy;
  updateControlState();
}

function setHistoryBusy(value) {
  isHistoryBusy = value;
  updateControlState();
}

function clearPreviewObjectUrl() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
}

function showPreview(blob) {
  clearPreviewObjectUrl();
  previewObjectUrl = URL.createObjectURL(blob);
  preview.src = previewObjectUrl;
  preview.hidden = false;
  video.hidden = true;
  cameraMessage.hidden = true;
}

function clearResult() {
  currentResult = null;
  resultSection.hidden = true;
  if (feedbackCorrection) feedbackCorrection.hidden = true;
  if (feedbackStatus) feedbackStatus.textContent = '';
  if (learningMemoryBadge) learningMemoryBadge.hidden = true;
  updateControlState();
}

// --- AI Health Status ---
function renderHealthStatus(state, title, tooltip) {
  if (!aiStatus || !aiStatusText) return;
  aiStatus.dataset.state = state;
  aiStatusText.textContent = title;
  aiStatus.title = tooltip || title;
}

async function refreshHealth() {
  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    let payload = {};
    try { payload = await response.json(); } catch (_) { /* ignore */ }
    const state = payload.classifier?.state || (response.ok ? 'not_loaded' : 'error');

    if (state === 'ready') {
      renderHealthStatus('ready', 'AI sẵn sàng', `Mô hình: ${payload.classifier?.model || 'đã nạp'}`);
    } else if (state === 'loading') {
      renderHealthStatus('checking', 'AI đang tải', 'Mô hình AI đang được nạp nền để lần quét đầu nhanh hơn.');
    } else if (state === 'not_loaded') {
      renderHealthStatus('not_loaded', 'AI chưa tải', 'Mô hình sẽ được tải khi bạn phân loại ảnh lần đầu.');
    } else if (state === 'retry_available') {
      renderHealthStatus('not_loaded', 'AI có thể thử lại', 'Thời gian chờ sau lỗi đã hết; lần quét tiếp theo sẽ thử nạp mô hình lại.');
    } else {
      const retryIn = Number(payload.classifier?.retry_in_seconds || 0);
      const retryHint = retryIn > 0 ? ` Có thể thử lại sau khoảng ${Math.ceil(retryIn)} giây.` : '';
      renderHealthStatus('error', 'AI lỗi', `${payload.classifier?.error || 'Mô hình AI hiện không khả dụng.'}${retryHint}`);
    }
  } catch (error) {
    console.error('Lỗi kiểm tra trạng thái AI:', error);
    renderHealthStatus('offline', 'Mất kết nối', 'Không thể kết nối đến máy chủ.');
  }
}

// --- Camera Logic ---
function stopStreamTracks(targetStream) {
  if (targetStream) targetStream.getTracks().forEach(track => track.stop());
}

function cameraErrorMessage(error) {
  switch (error?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera bị chặn. Hãy cấp quyền camera cho trang này hoặc tải ảnh lên.';
    case 'NotFoundError':
      return 'Không tìm thấy camera phù hợp. Bạn vẫn có thể tải ảnh từ thiết bị.';
    case 'NotReadableError':
      return 'Camera đang được ứng dụng khác sử dụng hoặc chưa sẵn sàng. Hãy thử lại hoặc tải ảnh lên.';
    case 'OverconstrainedError':
      return 'Camera không hỗ trợ cấu hình yêu cầu. Hãy thử đổi camera hoặc tải ảnh lên.';
    default:
      return 'Không thể kết nối camera. Vui lòng cấp quyền hoặc tải ảnh lên.';
  }
}

async function requestCameraStream({ requireFacingMode = false } = {}) {
  const facingConstraint = requireFacingMode
    ? { exact: facingMode }
    : { ideal: facingMode };
  const attempts = [
    {
      audio: false,
      video: {
        facingMode: facingConstraint,
        width: { ideal: CAMERA_TARGET_WIDTH },
        height: { ideal: CAMERA_TARGET_HEIGHT },
        frameRate: { ideal: 24, max: 30 }
      }
    },
    { audio: false, video: { facingMode: facingConstraint } }
  ];

  // A generic camera fallback is useful during normal startup, but must not
  // be used while switching cameras: otherwise the browser may return the
  // current camera and the UI would incorrectly report a successful switch.
  if (!requireFacingMode) attempts.push({ audio: false, video: true });

  let lastError = null;
  for (const constraints of attempts) {
    try {
      const candidate = await navigator.mediaDevices.getUserMedia(constraints);
      if (requireFacingMode) {
        const actualFacingMode = candidate.getVideoTracks()[0]?.getSettings?.().facingMode;
        if (actualFacingMode && actualFacingMode !== facingMode) {
          stopStreamTracks(candidate);
          lastError = new DOMException('Camera trả về không đúng hướng yêu cầu.', 'OverconstrainedError');
          continue;
        }
      }
      return candidate;
    } catch (error) {
      lastError = error;
      if (['NotAllowedError', 'SecurityError'].includes(error?.name)) break;
    }
  }
  throw lastError || new Error('Không thể mở camera.');
}

async function startCamera({ requireFacingMode = false } = {}) {
  if (isBusy || isHistoryBusy || filePickerOpening || document.hidden) return false;
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraMessage.textContent = 'Trình duyệt không hỗ trợ camera. Hãy tải ảnh rác từ thiết bị.';
    captureButton.disabled = true;
    return false;
  }

  stopCamera();
  const cameraRequestId = cameraSequence;
  preview.hidden = true;
  video.hidden = false;
  cameraMessage.hidden = false;
  cameraMessage.textContent = 'Đang khởi động camera...';

  try {
    const newStream = await requestCameraStream({ requireFacingMode });

    if (cameraRequestId !== cameraSequence || document.hidden || isBusy || isHistoryBusy || filePickerOpening) {
      stopStreamTracks(newStream);
      return false;
    }

    stream = newStream;
    video.srcObject = newStream;
    await video.play();

    if (cameraRequestId !== cameraSequence || document.hidden || isBusy || isHistoryBusy || filePickerOpening || stream !== newStream) {
      stopStreamTracks(newStream);
      if (stream === newStream) stream = null;
      if (video.srcObject === newStream) video.srcObject = null;
      return false;
    }

    cameraMessage.hidden = true;
    updateControlState();
    return true;
  } catch (error) {
    if (cameraRequestId !== cameraSequence) return false;
    stopCamera();
    console.error(error);
    cameraMessage.hidden = false;
    cameraMessage.textContent = cameraErrorMessage(error);
    captureButton.disabled = true;
    return false;
  }
}

function stopCamera() {
  cameraSequence += 1;
  stopStreamTracks(stream);
  stream = null;
  video.srcObject = null;
  captureButton.disabled = true;
}

function canvasToBlob(canvasElement, type = 'image/jpeg', quality = JPEG_QUALITY) {
  return new Promise((resolve, reject) => {
    canvasElement.toBlob(blob => blob ? resolve(blob) : reject(new Error('Không thể tạo ảnh.')), type, quality);
  });
}

function scaledSize(width, height, maxDimension = MAX_PROCESSING_DIMENSION) {
  const longest = Math.max(width, height);
  if (!longest || longest <= maxDimension) return { width, height };
  const scale = maxDimension / longest;
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale))
  };
}

async function loadImageBitmap(file) {
  if (globalThis.createImageBitmap) {
    return createImageBitmap(file);
  }

  const objectUrl = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = 'async';
    image.src = objectUrl;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function optimizeUploadedImage(file) {
  const supportedDirectTypes = new Set(['image/jpeg', 'image/png', 'image/webp']);
  if (supportedDirectTypes.has(file.type) && file.size <= UPLOAD_OPTIMIZE_THRESHOLD_BYTES) {
    return file;
  }

  let source;
  try {
    source = await loadImageBitmap(file);
    const sourceWidth = source.width || source.naturalWidth;
    const sourceHeight = source.height || source.naturalHeight;
    if (!sourceWidth || !sourceHeight) throw new Error('Không đọc được kích thước ảnh.');

    const size = scaledSize(sourceWidth, sourceHeight);
    const workCanvas = document.createElement('canvas');
    workCanvas.width = size.width;
    workCanvas.height = size.height;
    const context = workCanvas.getContext('2d', { alpha: false });
    context.drawImage(source, 0, 0, size.width, size.height);
    return await canvasToBlob(workCanvas, 'image/jpeg', JPEG_QUALITY);
  } catch (error) {
    if (supportedDirectTypes.has(file.type)) return file;
    throw new Error('Định dạng ảnh này chưa được trình duyệt hỗ trợ. Hãy chọn JPEG, PNG hoặc WebP.');
  } finally {
    if (source?.close) source.close();
  }
}

// --- Classification API Call ---
async function requestClassification(blob) {
  const requestId = ++requestSequence;
  activeRequestController?.abort();
  const controller = new AbortController();
  activeRequestController = controller;

  const formData = new FormData();
  formData.append('image', blob, 'waste-scan.jpg');

  try {
    const response = await fetch('/api/classify', {
      method: 'POST',
      body: formData,
      headers: deviceHeaders,
      signal: controller.signal
    });
    let payload = {};
    try { payload = await response.json(); } catch (_) { /* ignore */ }
    if (!response.ok) throw new Error(payload.detail || `Máy chủ trả về lỗi ${response.status}.`);

    if (requestId !== requestSequence) return false;
    renderResult(payload);
    void loadHistory({ reset: true });
    void refreshHealth();
    return true;
  } catch (error) {
    if (error.name === 'AbortError') return false;
    console.error(error);
    showToast(error.message || 'Không thể phân tích ảnh.');
    void refreshHealth();
    return false;
  } finally {
    if (activeRequestController === controller) activeRequestController = null;
  }
}

async function recoverScannerAfterFailure() {
  selectedBlob = null;
  fileInput.value = '';
  clearPreviewObjectUrl();
  preview.removeAttribute('src');
  preview.hidden = true;
  clearResult();

  if (!document.hidden) {
    await startCamera();
  }
}

async function captureFrame() {
  if (isBusy || isHistoryBusy) return;
  if (!stream || !video.videoWidth) {
    showToast('Camera chưa sẵn sàng.');
    return;
  }

  clearResult();
  setBusy(true);
  let classificationSucceeded = false;
  try {
    const size = scaledSize(video.videoWidth, video.videoHeight);
    canvas.width = size.width;
    canvas.height = size.height;
    const context = canvas.getContext('2d', { alpha: false });
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    selectedBlob = await canvasToBlob(canvas, 'image/jpeg', JPEG_QUALITY);
    showPreview(selectedBlob);
    stopCamera();
    classificationSucceeded = await requestClassification(selectedBlob);
  } catch (error) {
    console.error(error);
    showToast(error.message || 'Không thể chụp hoặc phân tích ảnh.');
  } finally {
    setBusy(false);
    if (!classificationSucceeded) await recoverScannerAfterFailure();
  }
}

// --- Feedback learning ---
function populateCategorySelect(selectElement, selectedKey = '') {
  if (!selectElement) return;
  const items = Array.isArray(catalogItems) ? catalogItems : [];
  selectElement.replaceChildren(...items.map(item => {
    const option = document.createElement('option');
    option.value = item.key;
    option.textContent = `${item.icon || '♻️'} ${item.display_name}`;
    return option;
  }));
  if (selectedKey && items.some(item => item.key === selectedKey)) {
    selectElement.value = selectedKey;
  }
}

function populateFeedbackCategories(selectedKey = '') {
  populateCategorySelect(feedbackCategory, selectedKey);
}

function renderLearningMemory(result) {
  if (!learningMemoryBadge) return;
  const info = result.analysis?.learning_memory;
  if (!info?.applied) {
    learningMemoryBadge.hidden = true;
    learningMemoryBadge.textContent = '';
    return;
  }
  const similarity = Math.round(Number(info.best_similarity || 0) * 100);
  const matched = Number(info.matched_examples || 0);
  learningMemoryBadge.textContent = `🧠 Đã tham khảo ${matched} mẫu bạn từng xác nhận · tương đồng cao nhất ${similarity}%`;
  learningMemoryBadge.hidden = false;
}

async function refreshLearningStats() {
  if (!learningExampleCount) return;
  try {
    const response = await fetch('/api/learning/stats', { cache: 'no-store' });
    let payload = {};
    try { payload = await response.json(); } catch (_) { /* ignore */ }
    if (!response.ok) return;
    const total = Number(payload.learnable_examples || 0);
    const corrected = Number(payload.corrected || 0);
    learningExampleCount.textContent = `Bộ nhớ học: ${total} mẫu · ${corrected} lần bạn đã sửa AI`;
  } catch (error) {
    console.warn('Không thể tải thống kê học:', error);
  }
}

async function postFeedback(scanId, correctKey) {
  const response = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scan_id: scanId, correct_key: correctKey })
  });
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* ignore */ }
  if (!response.ok) throw new Error(payload.detail || 'Không thể lưu phản hồi.');
  return payload;
}

function applyFeedbackResult(payload) {
  if (!currentResult || !payload?.corrected_key) return;

  currentResult.corrected_key = payload.corrected_key;
  currentResult.is_correct = payload.is_correct;

  // Feedback returns the complete canonical waste rule. Replace the visible
  // identity/handling fields while preserving the model's original confidence,
  // alternatives and analysis for reference.
  for (const field of ['display_name', 'category', 'bin_name', 'instruction', 'icon']) {
    if (typeof payload[field] === 'string') currentResult[field] = payload[field];
  }

  const hasUserLabel = Boolean(currentResult.corrected_key);
  document.getElementById('resultIcon').textContent =
    currentResult.uncertain && !hasUserLabel ? '❓' : currentResult.icon;
  document.getElementById('resultCategory').textContent =
    currentResult.uncertain && !hasUserLabel ? 'AI chưa đủ chắc chắn' : currentResult.category;
  document.getElementById('resultName').textContent = currentResult.display_name;
  document.getElementById('resultBin').textContent = currentResult.bin_name;
  document.getElementById('resultInstruction').textContent = currentResult.instruction;
}

async function submitFeedback(correctKey) {
  if (!currentResult?.scan_id || feedbackSubmitting || !correctKey) return;
  const scanId = currentResult.scan_id;
  feedbackSubmitting = true;
  updateControlState();
  if (feedbackStatus) feedbackStatus.textContent = 'Đang lưu phản hồi...';
  try {
    const payload = await postFeedback(scanId, correctKey);
    // Controls are locked while feedback is in-flight, but keep this identity
    // check as a second guard against stale async responses.
    if (currentResult?.scan_id === scanId) {
      // Keep the effective feedback label in sync with the server. Without this,
      // pressing "Đúng" after a correction would submit the original AI key
      // again and silently undo the user's correction.
      applyFeedbackResult(payload);
      populateFeedbackCategories(payload.corrected_key);
      if (feedbackStatus) feedbackStatus.textContent = payload.message || 'Đã lưu phản hồi.';
      if (feedbackCorrection) feedbackCorrection.hidden = true;
    }
    showToast(payload.message || 'Đã lưu phản hồi.');
    void loadHistory({ reset: true });
    void refreshLearningStats();
  } catch (error) {
    console.error(error);
    if (feedbackStatus) feedbackStatus.textContent = error.message || 'Không thể lưu phản hồi.';
    showToast(error.message || 'Không thể lưu phản hồi.');
  } finally {
    feedbackSubmitting = false;
    updateControlState();
  }
}

// --- Render Result Card ---
function renderResult(result) {
  currentResult = result;
  if (feedbackStatus) {
    if (result.history_saved === false || !result.scan_id) {
      feedbackStatus.textContent = 'Kết quả này không được lưu vì lịch sử đã bị xóa trong lúc AI xử lý. Hãy quét lại để lưu và phản hồi.';
    } else if (result.learning?.enabled === false) {
      feedbackStatus.textContent = 'Bạn có thể xác nhận hoặc sửa kết quả; chức năng học từ phản hồi hiện đang tắt.';
    } else if (result.learning?.feedback_available === false) {
      feedbackStatus.textContent = 'Bạn vẫn có thể phản hồi, nhưng lần quét này không có vector đặc trưng để dùng làm mẫu học.';
    } else {
      feedbackStatus.textContent = 'Xác nhận hoặc sửa kết quả để AI học từ vật thể này.';
    }
  }
  if (feedbackCorrection) feedbackCorrection.hidden = true;
  populateFeedbackCategories(result.corrected_key || result.key);
  renderLearningMemory(result);

  const hasUserLabel = Boolean(result.corrected_key);
  document.getElementById('resultIcon').textContent =
    result.uncertain && !hasUserLabel ? '❓' : result.icon;
  document.getElementById('resultCategory').textContent =
    result.uncertain && !hasUserLabel ? 'AI chưa đủ chắc chắn' : result.category;
  document.getElementById('resultName').textContent = result.display_name;
  document.getElementById('resultBin').textContent = result.bin_name;
  document.getElementById('resultInstruction').textContent = result.instruction;
  document.getElementById('resultNotice').textContent = result.notice;

  const percentage = Math.round(result.confidence * 100);
  document.getElementById('resultConfidence').textContent = `${percentage}%`;
  
  // Radial Gauge Animation
  const gaugeProgress = document.getElementById('gaugeProgress');
  if (gaugeProgress) {
    const circumference = 264;
    const offset = circumference - (percentage / 100) * circumference;
    gaugeProgress.style.strokeDashoffset = offset;
  }

  const alternatives = document.getElementById('alternatives');
  alternatives.replaceChildren(...(result.alternatives || []).map(item => {
    const row = document.createElement('div');
    row.className = 'alternative-item';
    const name = document.createElement('span');
    name.textContent = item.display_name;
    const score = document.createElement('strong');
    score.textContent = `${Math.round(item.confidence * 100)}%`;
    row.append(name, score);
    return row;
  }));

  resultSection.hidden = false;
  updateControlState();
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// --- Drag and Drop File Handling ---
['dragenter', 'dragover'].forEach(eventName => {
  dropZone.addEventListener(eventName, e => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('drag-over');
  });
});

['dragleave', 'drop'].forEach(eventName => {
  dropZone.addEventListener(eventName, e => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');
  });
});

dropZone.addEventListener('drop', async e => {
  const dt = e.dataTransfer;
  const file = dt.files[0];
  if (!file) return;
  processSelectedFile(file);
});

async function processSelectedFile(file) {
  if (isBusy || isHistoryBusy || feedbackSubmitting) return;
  clearResult();
  setBusy(true);
  let classificationSucceeded = false;
  try {
    stopCamera();
    showPreview(file);
    selectedBlob = await optimizeUploadedImage(file);
    classificationSucceeded = await requestClassification(selectedBlob);
  } catch (error) {
    console.error(error);
    showToast(error.message || 'Không thể đọc hoặc phân tích ảnh.');
  } finally {
    setBusy(false);
    if (!classificationSucceeded) await recoverScannerAfterFailure();
  }
}

// --- Sample Quick Tests ---
const SAMPLES = {
  plastic: '/static/samples/plastic-bottle.jpg',
  paper: '/static/samples/cardboard-box.jpg',
  metal: '/static/samples/aluminum-can.jpg',
  organic: '/static/samples/fruit-peel.jpg'
};

document.querySelectorAll('.chip-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (isBusy || isHistoryBusy) return;
    const sampleUrl = SAMPLES[btn.dataset.sample];
    if (!sampleUrl) return;

    try {
      const response = await fetch(sampleUrl, { cache: 'force-cache' });
      if (!response.ok) throw new Error('Không thể tải ảnh minh họa mẫu.');
      const blob = await response.blob();
      await processSelectedFile(blob);
    } catch (error) {
      console.error(error);
      showToast(error.message || 'Không thể mở ảnh minh họa mẫu.');
    }
  });
});

// --- Waste Catalog Dynamic Loader ---
async function loadCatalog() {
  if (!catalogGrid) return;
  try {
    const res = await fetch('/api/categories');
    if (!res.ok) throw new Error(`Máy chủ trả về lỗi ${res.status}.`);
    const data = await res.json();
    catalogItems = Array.isArray(data) ? data : [];
    populateFeedbackCategories(currentResult?.corrected_key || currentResult?.key || '');
    // History can finish loading before the catalog request. Re-render once the
    // localized category names are available so corrected keys never remain as
    // raw values such as "paper" or "plastic" until the next refresh.
    if (cachedHistoryItems.length) renderHistoryItems(cachedHistoryItems);
    
    catalogGrid.replaceChildren(...catalogItems.map(item => {
      const card = document.createElement('article');
      card.className = 'catalog-card';
      
      const header = document.createElement('div');
      header.className = 'catalog-card-header';
      
      const icon = document.createElement('div');
      icon.className = 'catalog-icon';
      icon.textContent = item.icon || '♻️';
      
      const title = document.createElement('h3');
      title.textContent = item.display_name;
      
      header.append(icon, title);
      
      const binTag = document.createElement('span');
      binTag.className = 'catalog-bin-tag';
      binTag.textContent = item.bin_name;
      
      const desc = document.createElement('p');
      desc.textContent = item.instruction;
      
      card.append(header, binTag, desc);
      return card;
    }));
  } catch (err) {
    console.error('Lỗi tải danh mục:', err);
    const message = document.createElement('div');
    message.className = 'catalog-loading';
    message.textContent = 'Không thể tải danh mục phân loại. Vui lòng tải lại trang hoặc kiểm tra kết nối.';
    catalogGrid.replaceChildren(message);
  }
}

// --- History Drawer & Statistics ---
function formatDate(value) {
  const date = new Date(value);
  return new Intl.DateTimeFormat('vi-VN', {
    dateStyle: 'short', timeStyle: 'short'
  }).format(date);
}

function updateHistoryStats(statistics = {}, historyTotal = null) {
  const total = Number(statistics.total || 0);
  const recycledCount = Number(statistics.recycled || 0);
  const averageConfidence = Number(statistics.average_confidence || 0);
  const overallTotal = historyTotal === null ? total : Number(historyTotal || 0);

  if (historyBadge) {
    historyBadge.textContent = overallTotal;
    historyBadge.hidden = overallTotal === 0;
  }

  const statTotal = document.getElementById('statTotal');
  const statRecycled = document.getElementById('statRecycled');
  const statConfidence = document.getElementById('statConfidence');
  if (!statTotal || !statRecycled || !statConfidence) return;

  statTotal.textContent = total;
  statRecycled.textContent = recycledCount;
  statConfidence.textContent = `${Math.round(averageConfidence * 100)}%`;
}


function displayNameForKey(key) {
  if (!key) return 'Chưa xác nhận';
  return catalogItems.find(item => item.key === key)?.display_name || key;
}

async function getHistoryThumbnailUrl(item) {
  if (!item?.thumbnail_available) return null;
  if (historyThumbnailUrls.has(item.id)) return historyThumbnailUrls.get(item.id);
  try {
    const response = await fetch(`/api/history/${item.id}/thumbnail`, { cache: 'no-store' });
    if (!response.ok) return null;
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    // Two UI consumers can request the same thumbnail before either fetch
    // finishes. Keep the first cached URL and immediately release duplicates.
    const cachedUrl = historyThumbnailUrls.get(item.id);
    if (cachedUrl) {
      URL.revokeObjectURL(url);
      return cachedUrl;
    }
    historyThumbnailUrls.set(item.id, url);
    return url;
  } catch (error) {
    console.warn(`Không thể tải thumbnail scan ${item.id}:`, error);
    return null;
  }
}

function releaseHistoryThumbnailUrl(scanId) {
  const url = historyThumbnailUrls.get(scanId);
  if (!url) return;
  URL.revokeObjectURL(url);
  historyThumbnailUrls.delete(scanId);
}

function releaseHistoryThumbnailUrls() {
  for (const scanId of Array.from(historyThumbnailUrls.keys())) {
    releaseHistoryThumbnailUrl(scanId);
  }
}

function isHistoryThumbnailNeeded(scanId) {
  return cachedHistoryItems.some(item => item.id === scanId)
    || historyEditItem?.id === scanId;
}

function pruneHistoryThumbnailUrls() {
  for (const scanId of Array.from(historyThumbnailUrls.keys())) {
    if (!isHistoryThumbnailNeeded(scanId)) releaseHistoryThumbnailUrl(scanId);
  }
}

async function attachHistoryThumbnail(item, imageElement, placeholderElement) {
  if (!item.thumbnail_available) {
    imageElement.hidden = true;
    placeholderElement.hidden = false;
    return;
  }
  const url = await getHistoryThumbnailUrl(item);
  if (!imageElement.isConnected && imageElement !== historyEditImage) {
    // A reset/search can replace the row while its thumbnail request is still
    // in flight. Revoke that late Object URL when no current UI still needs it.
    if (!isHistoryThumbnailNeeded(item.id)) releaseHistoryThumbnailUrl(item.id);
    return;
  }
  if (imageElement === historyEditImage && historyEditItem?.id !== item.id) {
    if (!isHistoryThumbnailNeeded(item.id)) releaseHistoryThumbnailUrl(item.id);
    return;
  }
  if (url) {
    imageElement.src = url;
    imageElement.hidden = false;
    placeholderElement.hidden = true;
  } else {
    imageElement.hidden = true;
    placeholderElement.hidden = false;
    placeholderElement.textContent = 'Không tải được ảnh';
  }
}

async function openHistoryEditor(item) {
  if (!historyEditModal || !item) return;
  if (!catalogItems.length) await loadCatalog();
  historyEditItem = item;
  const effectiveKey = item.corrected_key || item.waste_key;
  populateCategorySelect(historyEditCategory, item.recovered && !item.corrected_key ? '' : effectiveKey);
  if (item.recovered && !item.corrected_key && historyEditCategory) {
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = '— Chọn nhãn đúng —';
    placeholder.disabled = true;
    placeholder.selected = true;
    historyEditCategory.prepend(placeholder);
  }

  if (historyEditObjectName) {
    historyEditObjectName.textContent = item.recovered ? `Ảnh khôi phục #${item.id}` : item.display_name;
  }
  if (historyEditPredicted) {
    historyEditPredicted.textContent = item.recovered
      ? 'Không khôi phục được nhãn AI ban đầu'
      : `${item.display_name} · ${item.category}${item.uncertain ? ' · AI chưa chắc chắn' : ''}`;
  }
  if (historyEditConfidence) {
    historyEditConfidence.textContent = item.recovered
      ? 'Không có'
      : `${item.uncertain ? '~' : ''}${Math.round(item.confidence * 100)}%`;
  }
  if (historyEditTime) historyEditTime.textContent = formatDate(item.created_at);
  if (historyEditCurrentLabel) {
    historyEditCurrentLabel.textContent = item.corrected_key
      ? displayNameForKey(item.corrected_key)
      : (item.recovered ? 'Chưa xác nhận' : 'Chưa xác nhận · đang dùng dự đoán AI');
  }
  if (historyEditStatus) {
    historyEditStatus.textContent = item.recovered
      ? 'Ảnh này được khôi phục từ data/scans sau khi DB bị thiếu phần đuôi lịch sử. Hãy xem ảnh và chọn nhãn đúng; bản ghi khôi phục không có embedding cũ để dùng làm mẫu học.'
      : (item.thumbnail_available
        ? 'Đối chiếu ảnh rồi chọn nhãn đúng. Lưu lại sẽ cập nhật ngay bộ nhớ học.'
        : 'Bản ghi này không còn thumbnail. Bạn vẫn có thể sửa nhãn nếu nhận ra lần quét từ thông tin bên cạnh.');
  }
  if (historyEditImage) {
    historyEditImage.removeAttribute('src');
    historyEditImage.hidden = true;
  }
  if (historyEditImagePlaceholder) {
    historyEditImagePlaceholder.textContent = item.thumbnail_available ? 'Đang tải ảnh...' : 'Ảnh cũ không có thumbnail';
    historyEditImagePlaceholder.hidden = false;
  }

  historyEditModal.hidden = false;
  historyEditModal.setAttribute('aria-hidden', 'false');
  updateControlState();
  if (item.thumbnail_available && historyEditImage && historyEditImagePlaceholder) {
    void attachHistoryThumbnail(item, historyEditImage, historyEditImagePlaceholder);
  }
  historyEditCategory?.focus();
}

function closeHistoryEditor() {
  if (!historyEditModal || feedbackSubmitting) return;
  historyEditModal.hidden = true;
  historyEditModal.setAttribute('aria-hidden', 'true');
  historyEditItem = null;
  pruneHistoryThumbnailUrls();
  updateControlState();
}

async function saveHistoryEdit() {
  const correctKey = historyEditCategory?.value;
  if (!historyEditItem?.id || !correctKey || feedbackSubmitting) return;
  feedbackSubmitting = true;
  updateControlState();
  if (historyEditStatus) historyEditStatus.textContent = 'Đang cập nhật nhãn...';
  try {
    const payload = await postFeedback(historyEditItem.id, correctKey);
    historyEditItem.corrected_key = payload.corrected_key;
    historyEditItem.is_correct = payload.is_correct;
    if (currentResult?.scan_id === historyEditItem.id) {
      applyFeedbackResult(payload);
      populateFeedbackCategories(payload.corrected_key);
    }
    const cached = cachedHistoryItems.find(item => item.id === historyEditItem.id);
    if (cached) {
      cached.corrected_key = payload.corrected_key;
      cached.is_correct = payload.is_correct;
    }
    if (historyEditCurrentLabel) historyEditCurrentLabel.textContent = displayNameForKey(payload.corrected_key);
    if (historyEditStatus) historyEditStatus.textContent = payload.message || 'Đã cập nhật nhãn.';
    await loadHistory({ reset: true });
    void refreshLearningStats();
    showToast(payload.message || `Đã cập nhật nhãn thành ${displayNameForKey(payload.corrected_key)}.`);
  } catch (error) {
    console.error(error);
    if (historyEditStatus) historyEditStatus.textContent = error.message || 'Không thể cập nhật bản ghi.';
    showToast(error.message || 'Không thể cập nhật bản ghi.');
  } finally {
    feedbackSubmitting = false;
    updateControlState();
  }
}

async function deleteHistoryItem(item) {
  if (!item?.id || isBusy || isHistoryBusy || feedbackSubmitting) return;

  const itemName = item.recovered ? `Ảnh khôi phục #${item.id}` : item.display_name;
  const confirmed = window.confirm(
    `Xóa “${itemName}” khỏi lịch sử? Ảnh, phản hồi và dữ liệu học của riêng lần quét này cũng sẽ bị xóa. Thao tác này không thể hoàn tác.`
  );
  if (!confirmed) return;

  setHistoryBusy(true);
  try {
    const response = await fetch(`/api/history/${item.id}`, { method: 'DELETE' });
    let payload = {};
    try { payload = await response.json(); } catch (_) { /* ignore */ }
    if (!response.ok) throw new Error(payload.detail || 'Không thể xóa lần quét này.');

    releaseHistoryThumbnailUrl(item.id);
    if (historyEditItem?.id === item.id) {
      historyEditModal.hidden = true;
      historyEditModal.setAttribute('aria-hidden', 'true');
      historyEditItem = null;
    }
    if (currentResult?.scan_id === item.id) {
      // Keep the visible classification card as a reference, but it no longer has
      // a backing DB row for feedback after this history item is deleted.
      currentResult = null;
      if (feedbackCorrection) feedbackCorrection.hidden = true;
      if (feedbackStatus) feedbackStatus.textContent = '';
    }

    cachedHistoryItems = cachedHistoryItems.filter(entry => entry.id !== item.id);
    await loadHistory({ reset: true });
    await refreshLearningStats();

    showToast(`Đã xóa lần quét #${item.id}. ID đã dùng sẽ không được tái sử dụng để tránh ghép nhầm dữ liệu.`);
  } catch (error) {
    console.error(error);
    showToast(error.message || 'Không thể xóa lần quét này.');
  } finally {
    setHistoryBusy(false);
  }
}

function renderHistoryItems(items) {
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-history';
    empty.textContent = (historySearch?.value || '').trim()
      ? 'Không tìm thấy lịch sử phù hợp.'
      : 'Chưa có lịch sử quét rác.';
    historyList.replaceChildren(empty);
    return;
  }

  historyList.replaceChildren(...items.map(item => {
    const row = document.createElement('article');
    row.className = 'history-item';

    const thumbButton = document.createElement('button');
    thumbButton.className = 'history-thumb-button';
    thumbButton.type = 'button';
    thumbButton.setAttribute('aria-label', `Xem ảnh và sửa nhãn cho ${item.display_name}`);
    thumbButton.addEventListener('click', () => { void openHistoryEditor(item); });

    const thumb = document.createElement('img');
    thumb.className = 'history-thumb';
    thumb.alt = '';
    thumb.hidden = true;
    const thumbPlaceholder = document.createElement('span');
    thumbPlaceholder.className = 'history-thumb-placeholder';
    thumbPlaceholder.textContent = item.thumbnail_available ? '…' : 'Ảnh cũ';
    thumbButton.append(thumb, thumbPlaceholder);
    void attachHistoryThumbnail(item, thumb, thumbPlaceholder);

    const details = document.createElement('div');
    details.className = 'history-item-info';

    const title = document.createElement('strong');
    title.textContent = item.recovered ? `Ảnh khôi phục #${item.id}` : item.display_name;

    const meta = document.createElement('small');
    const feedbackText = item.corrected_key
      ? (item.recovered
        ? ` · Đã gán: ${displayNameForKey(item.corrected_key)}`
        : (item.is_correct ? ' · Đã xác nhận đúng' : ` · Đã sửa: ${displayNameForKey(item.corrected_key)}`))
      : ' · Chưa xác nhận';
    meta.textContent = item.recovered
      ? `Khôi phục từ ảnh · Chưa có nhãn AI${feedbackText} · ${formatDate(item.created_at)}`
      : `${item.category}${item.uncertain ? ' · Chưa chắc chắn' : ''}${feedbackText} · ${formatDate(item.created_at)}`;

    const itemActions = document.createElement('div');
    itemActions.className = 'history-item-actions';

    const editButton = document.createElement('button');
    editButton.className = 'history-edit-button';
    editButton.type = 'button';
    editButton.textContent = item.corrected_key ? 'Xem / sửa lại' : 'Xem / sửa nhãn';
    editButton.addEventListener('click', () => { void openHistoryEditor(item); });

    const deleteButton = document.createElement('button');
    deleteButton.className = 'history-delete-button';
    deleteButton.type = 'button';
    deleteButton.textContent = 'Xóa';
    deleteButton.setAttribute('aria-label', `Xóa ${item.display_name} khỏi lịch sử`);
    deleteButton.addEventListener('click', () => { void deleteHistoryItem(item); });

    itemActions.append(editButton, deleteButton);
    details.append(title, meta, itemActions);

    const score = document.createElement('span');
    score.className = 'history-score';
    score.textContent = item.recovered
      ? '—'
      : `${item.uncertain ? '~' : ''}${Math.round(item.confidence * 100)}%`;

    row.append(thumbButton, details, score);
    return row;
  }));
}

async function loadHistory({ reset = true } = {}) {
  if (!reset && (!historyHasMore || isHistoryLoading)) return;

  if (reset) {
    historyRequestController?.abort();
    historyCursor = null;
    historyHasMore = false;
  }

  const requestId = ++historyRequestSequence;
  const controller = new AbortController();
  historyRequestController = controller;
  isHistoryLoading = true;
  updateControlState();

  const params = new URLSearchParams({
    limit: String(HISTORY_PAGE_SIZE)
  });
  if (!reset && Number.isInteger(historyCursor)) {
    params.set('before_id', String(historyCursor));
  }
  const query = (historySearch?.value || '').trim();
  if (query) params.set('q', query);

  try {
    const response = await fetch(`/api/history?${params}`, { signal: controller.signal });
    let payload = {};
    try { payload = await response.json(); } catch (_) { /* ignore */ }
    if (!response.ok) throw new Error(payload.detail || 'Không thể tải lịch sử.');
    if (requestId !== historyRequestSequence) return;

    const items = Array.isArray(payload.items) ? payload.items : [];
    if (reset) {
      cachedHistoryItems = items;
      // Search/reset replaces the visible result set. Drop Object URLs for
      // thumbnails that are no longer represented, except an item still open
      // in the history editor.
      pruneHistoryThumbnailUrls();
    } else {
      const existingIds = new Set(cachedHistoryItems.map(item => item.id));
      cachedHistoryItems = [
        ...cachedHistoryItems,
        ...items.filter(item => !existingIds.has(item.id))
      ];
    }
    historyCursor = Number.isInteger(payload.next_cursor) ? payload.next_cursor : null;
    historyHasMore = Boolean(payload.has_more) && historyCursor !== null;

    updateHistoryStats(payload.statistics, payload.history_total);
    renderHistoryItems(cachedHistoryItems);
  } catch (error) {
    if (error.name === 'AbortError' || requestId !== historyRequestSequence) return;
    console.error(error);

    if (reset) {
      cachedHistoryItems = [];
      pruneHistoryThumbnailUrls();
      historyCursor = null;
      historyHasMore = false;
      updateHistoryStats({});
      const empty = document.createElement('div');
      empty.className = 'empty-history';
      empty.textContent = error.message;
      historyList.replaceChildren(empty);
    } else {
      showToast(error.message || 'Không thể tải thêm lịch sử.');
    }
  } finally {
    if (requestId === historyRequestSequence) {
      isHistoryLoading = false;
      if (historyRequestController === controller) historyRequestController = null;
      if (loadMoreHistoryButton) loadMoreHistoryButton.hidden = !historyHasMore;
      updateControlState();
    }
  }
}

if (historySearch) {
  historySearch.addEventListener('input', () => {
    clearTimeout(historySearchTimer);
    historyRequestController?.abort();
    historyHasMore = false;
    if (loadMoreHistoryButton) loadMoreHistoryButton.hidden = true;
    historySearchTimer = setTimeout(() => loadHistory({ reset: true }), 250);
  });
}

if (loadMoreHistoryButton) {
  loadMoreHistoryButton.addEventListener('click', () => loadHistory({ reset: false }));
}

function openDrawer() {
  historyDrawer.classList.add('open');
  historyDrawer.setAttribute('aria-hidden', 'false');
  loadHistory({ reset: true });
}

function closeDrawer() {
  historyDrawer.classList.remove('open');
  historyDrawer.setAttribute('aria-hidden', 'true');
}

// --- Event Listeners ---
captureButton.addEventListener('click', captureFrame);
switchCameraButton.addEventListener('click', async () => {
  if (isBusy || isHistoryBusy) return;
  const previousFacingMode = facingMode;
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  const switched = await startCamera({ requireFacingMode: true });
  if (!switched) {
    facingMode = previousFacingMode;
    if (!document.hidden) {
      const restored = await startCamera();
      showToast(restored
        ? 'Không thể chuyển camera; đã quay lại camera trước.'
        : 'Không thể chuyển camera và cũng không thể khôi phục camera trước.');
    }
  }
});

uploadButton?.addEventListener('click', () => {
  if (isBusy || isHistoryBusy) return;
  filePickerOpening = true;
  clearTimeout(cameraRestartTimer);

  // Open the picker directly from the user's click. Do not stop the camera
  // first: on some mobile browsers that can interrupt/cancel the picker gesture.
  fileInput.value = '';
  fileInput.click();
});

fileInput.addEventListener('change', async event => {
  const file = event.target.files?.[0];
  event.target.value = '';
  filePickerOpening = false;
  if (!file) {
    if (!stream && !document.hidden && !selectedBlob && !isBusy && !isHistoryBusy) void startCamera();
    return;
  }
  void processSelectedFile(file);
});

window.addEventListener('focus', () => {
  if (!filePickerOpening) return;
  clearTimeout(cameraRestartTimer);
  cameraRestartTimer = setTimeout(() => {
    if (!filePickerOpening) return;
    filePickerOpening = false;
    if (!stream && !selectedBlob && !isBusy && !isHistoryBusy && !document.hidden) void startCamera();
  }, 350);
});

if (scanAgainButton) {
  scanAgainButton.addEventListener('click', async () => {
    if (isBusy || isHistoryBusy) return;
    activeRequestController?.abort();
    requestSequence += 1;
    selectedBlob = null;
    fileInput.value = '';
    clearPreviewObjectUrl();
    preview.removeAttribute('src');
    preview.hidden = true;
    clearResult();
    await startCamera();
    document.getElementById('scanner').scrollIntoView({ behavior: 'smooth' });
  });
}

feedbackCorrectButton?.addEventListener('click', () => {
  const effectiveKey = currentResult?.corrected_key || currentResult?.key;
  if (!effectiveKey) return;
  void submitFeedback(effectiveKey);
});

feedbackFixButton?.addEventListener('click', async () => {
  if (!feedbackCorrection || !currentResult) return;
  if (!catalogItems.length) await loadCatalog();
  const effectiveKey = currentResult.corrected_key || currentResult.key;
  populateFeedbackCategories(effectiveKey);
  feedbackCorrection.hidden = !feedbackCorrection.hidden;
  if (!feedbackCorrection.hidden) feedbackCategory?.focus();
});

feedbackSaveButton?.addEventListener('click', () => {
  const correctKey = feedbackCategory?.value;
  if (correctKey) void submitFeedback(correctKey);
});

historyEditSaveButton?.addEventListener('click', () => { void saveHistoryEdit(); });
historyEditDeleteButton?.addEventListener('click', () => {
  if (historyEditItem) void deleteHistoryItem(historyEditItem);
});
document.querySelectorAll('[data-close-history-edit]').forEach(element => element.addEventListener('click', closeHistoryEditor));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && historyEditModal && !historyEditModal.hidden) closeHistoryEditor();
});

document.getElementById('historyToggle').addEventListener('click', openDrawer);
document.querySelectorAll('[data-close-drawer]').forEach(element => element.addEventListener('click', closeDrawer));

clearHistoryButton.addEventListener('click', async () => {
  if (isBusy || isHistoryBusy || feedbackSubmitting) return;
  const confirmed = window.confirm(
    'Xóa toàn bộ lịch sử dùng chung? Thao tác này xóa tất cả thumbnail và bộ nhớ học. ID đã dùng sẽ không được tái sử dụng để tránh ghép nhầm dữ liệu. Không thể hoàn tác.'
  );
  if (!confirmed) return;
  setHistoryBusy(true);
  try {
    const response = await fetch('/api/history', { method: 'DELETE' });
    if (!response.ok) throw new Error('Không thể xóa lịch sử.');
    releaseHistoryThumbnailUrls();
    closeHistoryEditor();
    // The visible result may refer to a scan that was just deleted. Keep the
    // result card for reference, but invalidate its DB-backed feedback actions.
    currentResult = null;
    if (feedbackCorrection) feedbackCorrection.hidden = true;
    if (feedbackStatus) feedbackStatus.textContent = '';
    updateControlState();
    await loadHistory({ reset: true });
    await refreshLearningStats();
    showToast('Đã xóa sạch lịch sử. ID cũ được giữ làm mốc và sẽ không bị tái sử dụng.');
  } catch (error) {
    showToast(error.message);
  } finally {
    setHistoryBusy(false);
  }
});

window.addEventListener('beforeunload', () => {
  activeRequestController?.abort();
  historyRequestController?.abort();
  stopCamera();
  clearPreviewObjectUrl();
  releaseHistoryThumbnailUrls();
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopCamera();
  else if (!selectedBlob && !isBusy && !isHistoryBusy && !feedbackSubmitting && !filePickerOpening) void startCamera();
});

// --- Initial Startup ---
setBusy(false);
filePickerOpening = false;
initTheme();
if (!clientIdentity.persistent) {
  console.warn('Trình duyệt đang chặn bộ nhớ và cookie; mã thiết bị chỉ ổn định trong phiên hiện tại. Lịch sử dùng chung không bị ảnh hưởng.');
} else if (!clientIdentity.fullySynced) {
  console.warn('Client ID chỉ lưu được ở một cơ chế trình duyệt; đây chỉ là metadata thiết bị, lịch sử dùng chung không bị ảnh hưởng.');
}
renderHealthStatus('checking', 'Đang kiểm tra AI', 'Đang kiểm tra trạng thái hệ thống AI.');
void startCamera();

const loadSecondaryContent = () => {
  void refreshHealth();
  void loadCatalog();
  void loadHistory({ reset: true });
  void refreshLearningStats();
};

if ('requestIdleCallback' in window) {
  requestIdleCallback(loadSecondaryContent, { timeout: 1000 });
} else {
  setTimeout(loadSecondaryContent, 120);
}
setInterval(() => void refreshHealth(), 30000);

