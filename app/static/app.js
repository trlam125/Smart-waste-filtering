const video = document.getElementById('camera');
const cameraShell = document.querySelector('.camera-shell');
const canvas = document.getElementById('canvas');
const preview = document.getElementById('preview');
const detectionOverlay = document.getElementById('detectionOverlay');
const objectResults = document.getElementById('objectResults');
const objectSummary = document.getElementById('objectSummary');
const cameraMessage = document.getElementById('cameraMessage');
const captureButton = document.getElementById('capture');
const switchCameraButton = document.getElementById('switchCamera');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const scanAgainButton = document.getElementById('scanAgain');
const loading = document.getElementById('loading');
const laserBeam = document.getElementById('laserBeam');
const resultSection = document.getElementById('resultSection');
const resultCard = document.getElementById('resultCard');
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
let currentResponse = null;
let currentObjects = [];
let activeObjectIndex = 0;
let catalogItems = [];
let feedbackSubmitting = false;
let historyEditItem = null;
const historyThumbnailUrls = new Map();

const HISTORY_PAGE_SIZE = 50;
const CAMERA_TARGET_WIDTH = 1920;
const CAMERA_TARGET_HEIGHT = 1080;
const MAX_PROCESSING_DIMENSION = 1024;
const UPLOAD_OPTIMIZE_THRESHOLD_BYTES = 1.2 * 1024 * 1024;
const JPEG_QUALITY = 0.82;
const COLLECTION_MAX_DIMENSION = 1600;
const COLLECTION_JPEG_QUALITY = 0.92;
const COLLECTION_DIRECT_MAX_BYTES = 12 * 1024 * 1024;

function detectPhoneDevice() {
  if (typeof navigator.userAgentData?.mobile === 'boolean') {
    return navigator.userAgentData.mobile;
  }
  const userAgent = navigator.userAgent || '';
  return /iPhone|iPod|Windows Phone|IEMobile|Opera Mini|Android.*Mobile/i.test(userAgent);
}

const PHONE_CAMERA_MODE = detectPhoneDevice();
document.documentElement.classList.toggle('phone-camera-mode', PHONE_CAMERA_MODE);
let preferredWideRearDeviceId = null;

function syncPhoneCameraAspect(width, height) {
  if (!PHONE_CAMERA_MODE || !cameraShell) return;
  const mediaWidth = Number(width);
  const mediaHeight = Number(height);
  if (!Number.isFinite(mediaWidth) || !Number.isFinite(mediaHeight) || mediaWidth <= 0 || mediaHeight <= 0) return;

  // Match the shell to the actual mobile camera/photo aspect ratio. This keeps
  // object-fit: contain without the large black side bars seen with portrait
  // camera streams inside the old fixed 4:3 shell.
  cameraShell.style.setProperty('--phone-camera-aspect', `${mediaWidth} / ${mediaHeight}`);
}

function syncPhoneVideoAspect() {
  syncPhoneCameraAspect(video.videoWidth, video.videoHeight);
}

function visibleObjectName(item, fallback = 'Object') {
  if (!item) return fallback;
  return item.display_name || fallback;
}

if (video) {
  video.addEventListener('loadedmetadata', syncPhoneVideoAspect);
  video.addEventListener('resize', syncPhoneVideoAspect);
}

if (preview) {
  preview.addEventListener('load', () => {
    syncPhoneCameraAspect(preview.naturalWidth, preview.naturalHeight);
    renderDetectionOverlay();
  });
}

const CLIENT_ID_STORAGE_KEY = 'waste-scanner-client-id';
const CLIENT_ID_COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

function isValidClientId(value) {
  const normalized = value?.trim();
  return Boolean(
    normalized
    && normalized.length <= 128
    && normalized.toLowerCase() !== 'anonymous'
  );
}

function safeStorageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch (error) {
    console.warn(`Could not read localStorage (${key}):`, error);
    return null;
  }
}

function safeStorageSet(key, value) {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch (error) {
    console.warn(`Could not write localStorage (${key}):`, error);
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
    console.warn(`Could not read cookie (${name}):`, error);
    return null;
  }
}

function safeCookieSet(name, value) {
  try {
    const secure = location.protocol === 'https:' ? '; Secure' : '';
    document.cookie = `${encodeURIComponent(name)}=${encodeURIComponent(value)}; Path=/; Max-Age=${CLIENT_ID_COOKIE_MAX_AGE}; SameSite=Lax${secure}`;
    return safeCookieGet(name) === value;
  } catch (error) {
    console.warn(`Could not write cookie (${name}):`, error);
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
  currentResponse = null;
  currentObjects = [];
  activeObjectIndex = 0;
  resultSection.hidden = true;
  if (objectResults) {
    objectResults.replaceChildren();
    objectResults.hidden = true;
  }
  if (objectSummary) {
    objectSummary.textContent = '';
    objectSummary.hidden = true;
  }
  if (detectionOverlay) {
    detectionOverlay.replaceChildren();
    detectionOverlay.hidden = true;
  }
  if (resultCard) resultCard.hidden = false;
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

    const classifierState = payload.classifier?.state || 'not_loaded';
    const detectorState = payload.detector?.state || 'not_loaded';
    const states = [detectorState, classifierState];

    if (payload.ready === true) {
      const classifierName = payload.classifier?.architecture || 'classifier';
      renderHealthStatus(
        'ready',
        'Multi-object AI ready',
        `YOLO detector + ${classifierName} loaded. You can scan multiple objects in a single image.`
      );
    } else if (states.includes('loading')) {
      renderHealthStatus('checking', 'AI is loading', 'The detector and classifier are loading in the background.');
    } else if (states.every(state => state === 'not_loaded')) {
      renderHealthStatus('not_loaded', 'AI not loaded', 'Both models will load when the app starts or on the first scan.');
    } else if (states.includes('retry_available')) {
      renderHealthStatus('not_loaded', 'AI can retry', 'A model retry cooldown has expired after an error; the next scan will try loading it again.');
    } else {
      const errors = [payload.detector?.error, payload.classifier?.error].filter(Boolean);
      const retryIn = Math.max(
        Number(payload.detector?.retry_in_seconds || 0),
        Number(payload.classifier?.retry_in_seconds || 0)
      );
      const retryHint = retryIn > 0 ? ` Retry in about ${Math.ceil(retryIn)} seconds.` : '';
      renderHealthStatus(
        'error',
        'AI error',
        `${errors.join(' | ') || 'The detector or classifier is currently unavailable.'}${retryHint}`
      );
    }
  } catch (error) {
    console.error('Error checking AI status:', error);
    renderHealthStatus('offline', 'Disconnected', 'Could not connect to the server.');
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
      return 'Camera access is blocked. Allow camera permission for this site or upload an image.';
    case 'NotFoundError':
      return 'No suitable camera was found. You can still upload an image from your device.';
    case 'NotReadableError':
      return 'The camera is in use by another app or is not ready. Try again or upload an image.';
    case 'OverconstrainedError':
      return 'The camera does not support the requested configuration. Try switching cameras or upload an image.';
    default:
      return 'Could not access the camera. Please grant permission or upload an image.';
  }
}

function looksLikeFrontCamera(label) {
  // Keep legacy localized camera-label aliases via escapes without localizing the UI.
  return /front|selfie|user|\u0074\u0072\u01b0\u1edbc|\u0074\u0072\u0075\u006f\u0063/i.test(label || '');
}

function looksLikeUltraWideCamera(label) {
  // Keep legacy localized ultra-wide aliases for devices that expose translated lens labels.
  return /ultra[\s-]*wide|ultrawide|0[.,]5\s*x|wide[\s-]*angle|\bwide\b|\u0067\u00f3\u0063\u0020\u0072\u1ed9\u006e\u0067|\u0067\u006f\u0063\u0020\u0072\u006f\u006e\u0067/i.test(label || '');
}

async function applyAccuracySafePhoneZoom(targetStream) {
  if (!PHONE_CAMERA_MODE || facingMode !== 'environment') return;
  const track = targetStream?.getVideoTracks?.()[0];
  if (!track?.getCapabilities || !track?.applyConstraints) return;
  try {
    const capabilities = track.getCapabilities();
    const zoom = capabilities?.zoom;
    const minZoom = Number(zoom?.min);
    const maxZoom = Number(zoom?.max);
    if (Number.isFinite(minZoom) && Number.isFinite(maxZoom)) {
      // On phones, prefer the widest optical/logical rear-camera view exposed
      // by the browser. Desktop/laptop behavior is unchanged by PHONE_CAMERA_MODE.
      const wideZoom = minZoom;
      await track.applyConstraints({ advanced: [{ zoom: wideZoom }] });
    }
  } catch (error) {
    console.debug('Could not set ultra-wide zoom for the mobile camera:', error);
  }
}

async function openExactCameraDevice(deviceId) {
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      deviceId: { exact: deviceId },
      width: { ideal: CAMERA_TARGET_WIDTH },
      height: { ideal: CAMERA_TARGET_HEIGHT },
      frameRate: { ideal: 24, max: 30 }
    }
  });
}

async function preferPhoneWideRearCamera(baseStream) {
  if (!PHONE_CAMERA_MODE || facingMode !== 'environment' || !navigator.mediaDevices?.enumerateDevices) {
    return baseStream;
  }

  try {
    const currentTrack = baseStream.getVideoTracks()[0];
    const currentSettings = currentTrack?.getSettings?.() || {};
    if (currentSettings.facingMode === 'user') return baseStream;
    const currentDeviceId = currentSettings.deviceId || '';
    const devices = await navigator.mediaDevices.enumerateDevices();
    const rearWideCandidates = devices.filter(device =>
      device.kind === 'videoinput'
      && device.deviceId
      && !looksLikeFrontCamera(device.label)
      && looksLikeUltraWideCamera(device.label)
    );

    const widePriority = device => {
      if (device.deviceId === preferredWideRearDeviceId) return 0;
      if (/ultra[\s-]*wide|ultrawide|0[.,]5\s*x/i.test(device.label || '')) return 1;
      return 2;
    };
    rearWideCandidates.sort((left, right) =>
      widePriority(left) - widePriority(right) || left.label.localeCompare(right.label)
    );

    const currentWide = rearWideCandidates.find(device => device.deviceId === currentDeviceId);
    if (currentWide) {
      preferredWideRearDeviceId = currentWide.deviceId;
      await applyAccuracySafePhoneZoom(baseStream);
      return baseStream;
    }

    const alternateCandidates = rearWideCandidates.filter(device => device.deviceId !== currentDeviceId);
    if (!alternateCandidates.length) {
      await applyAccuracySafePhoneZoom(baseStream);
      return baseStream;
    }

    // Some phones cannot open a second rear lens while the current track is
    // active. Release it before switching, and reopen the original device if
    // every ultra-wide candidate fails.
    stopStreamTracks(baseStream);
    let lastSwitchError = null;
    for (const device of alternateCandidates) {
      let wideStream = null;
      try {
        wideStream = await openExactCameraDevice(device.deviceId);
        const wideTrack = wideStream.getVideoTracks()[0];
        const actualFacingMode = wideTrack?.getSettings?.().facingMode;
        if (actualFacingMode === 'user') {
          stopStreamTracks(wideStream);
          continue;
        }
        await applyAccuracySafePhoneZoom(wideStream);
        preferredWideRearDeviceId = device.deviceId;
        return wideStream;
      } catch (error) {
        lastSwitchError = error;
        stopStreamTracks(wideStream);
        console.debug(`Could not open ultra-wide lens ${device.label || device.deviceId}:`, error);
      }
    }

    if (currentDeviceId) {
      try {
        const restoredStream = await openExactCameraDevice(currentDeviceId);
        await applyAccuracySafePhoneZoom(restoredStream);
        return restoredStream;
      } catch (restoreError) {
        console.debug('Could not reopen the camera after trying the ultra-wide lens:', restoreError);
        throw restoreError;
      }
    }
    throw lastSwitchError || new Error('Could not switch to the ultra-wide camera.');
  } catch (error) {
    // If baseStream is still live, keep it. If it was released during a lens
    // switch, bubble the error so requestCameraStream can retry its next safe
    // camera constraint instead of returning a dead MediaStream.
    if (baseStream?.getVideoTracks?.().some(track => track.readyState === 'live')) {
      console.debug('Could not detect an ultra-wide camera on mobile:', error);
      await applyAccuracySafePhoneZoom(baseStream);
      return baseStream;
    }
    throw error;
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
      let candidate = await navigator.mediaDevices.getUserMedia(constraints);
      if (requireFacingMode) {
        const actualFacingMode = candidate.getVideoTracks()[0]?.getSettings?.().facingMode;
        if (actualFacingMode && actualFacingMode !== facingMode) {
          stopStreamTracks(candidate);
          lastError = new DOMException('The camera did not match the requested facing mode.', 'OverconstrainedError');
          continue;
        }
      }
      if (PHONE_CAMERA_MODE && facingMode === 'environment') {
        candidate = await preferPhoneWideRearCamera(candidate);
      }
      return candidate;
    } catch (error) {
      lastError = error;
      if (['NotAllowedError', 'SecurityError'].includes(error?.name)) break;
    }
  }
  throw lastError || new Error('Could not open the camera.');
}

async function startCamera({ requireFacingMode = false } = {}) {
  if (isBusy || isHistoryBusy || filePickerOpening || document.hidden) return false;
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraMessage.textContent = 'This browser does not support camera access. Upload a waste image from your device.';
    captureButton.disabled = true;
    return false;
  }

  stopCamera();
  const cameraRequestId = cameraSequence;
  preview.hidden = true;
  video.hidden = false;
  cameraMessage.hidden = false;
  cameraMessage.textContent = 'Starting camera...';

  try {
    const newStream = await requestCameraStream({ requireFacingMode });

    if (cameraRequestId !== cameraSequence || document.hidden || isBusy || isHistoryBusy || filePickerOpening) {
      stopStreamTracks(newStream);
      return false;
    }

    stream = newStream;
    video.srcObject = newStream;
    await video.play();
    syncPhoneVideoAspect();

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
    canvasElement.toBlob(blob => blob ? resolve(blob) : reject(new Error('Could not create the image.')), type, quality);
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
    if (!sourceWidth || !sourceHeight) throw new Error('Could not read the image dimensions.');

    const size = scaledSize(sourceWidth, sourceHeight);
    const workCanvas = document.createElement('canvas');
    workCanvas.width = size.width;
    workCanvas.height = size.height;
    const context = workCanvas.getContext('2d', { alpha: false });
    context.drawImage(source, 0, 0, size.width, size.height);
    return await canvasToBlob(workCanvas, 'image/jpeg', JPEG_QUALITY);
  } catch (error) {
    if (supportedDirectTypes.has(file.type)) return file;
    throw new Error('This image format is not supported by the browser. Choose JPEG, PNG, or WebP.');
  } finally {
    if (source?.close) source.close();
  }
}

async function prepareHighQualityImage(file) {
  const supportedDirectTypes = new Set(['image/jpeg', 'image/png', 'image/webp']);
  let source;
  try {
    source = await loadImageBitmap(file);
    const sourceWidth = source.width || source.naturalWidth;
    const sourceHeight = source.height || source.naturalHeight;
    if (!sourceWidth || !sourceHeight) throw new Error('Could not read the image dimensions.');

    const size = scaledSize(sourceWidth, sourceHeight, COLLECTION_MAX_DIMENSION);
    const canKeepOriginal = supportedDirectTypes.has(file.type)
      && file.size <= COLLECTION_DIRECT_MAX_BYTES
      && size.width === sourceWidth
      && size.height === sourceHeight;
    if (canKeepOriginal) return file;

    const workCanvas = document.createElement('canvas');
    workCanvas.width = size.width;
    workCanvas.height = size.height;
    const context = workCanvas.getContext('2d', { alpha: false });
    context.drawImage(source, 0, 0, size.width, size.height);
    return await canvasToBlob(workCanvas, 'image/jpeg', COLLECTION_JPEG_QUALITY);
  } catch (error) {
    if (supportedDirectTypes.has(file.type) && file.size <= COLLECTION_DIRECT_MAX_BYTES) return file;
    throw new Error('Could not prepare a high-quality image for classification.');
  } finally {
    if (source?.close) source.close();
  }
}

// --- Classification API Call ---
async function requestClassification(blob, classifierBlob = null, collectionBlob = null, options = {}) {
  const requestId = ++requestSequence;
  activeRequestController?.abort();
  const controller = new AbortController();
  activeRequestController = controller;

  const formData = new FormData();
  formData.append('image', blob, 'waste-scan.jpg');
  formData.append('persist', options.persist === false ? 'false' : 'true');
  formData.append('source', options.source || 'user');
  if (classifierBlob && classifierBlob !== blob) {
    formData.append('classifier_image', classifierBlob, 'waste-scan-classifier-hq.jpg');
  }
  if (collectionBlob) {
    formData.append('collection_image', collectionBlob, 'waste-scan-collection-hq.jpg');
  }

  try {
    const response = await fetch('/api/classify', {
      method: 'POST',
      body: formData,
      headers: deviceHeaders,
      signal: controller.signal
    });
    let payload = {};
    try { payload = await response.json(); } catch (_) { /* ignore */ }
    if (!response.ok) throw new Error(payload.detail || `Server returned error ${response.status}.`);

    if (requestId !== requestSequence) return false;
    renderResult(payload);
    if (payload.history_saved) void loadHistory({ reset: true });
    void refreshHealth();
    return true;
  } catch (error) {
    if (error.name === 'AbortError') return false;
    console.error(error);
    showToast(error.message || 'Could not analyze the image.');
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
    showToast('Camera is not ready.');
    return;
  }

  clearResult();
  setBusy(true);
  let classificationSucceeded = false;
  try {
    const collectionSize = scaledSize(
      video.videoWidth,
      video.videoHeight,
      COLLECTION_MAX_DIMENSION
    );
    const collectionCanvas = document.createElement('canvas');
    collectionCanvas.width = collectionSize.width;
    collectionCanvas.height = collectionSize.height;
    const collectionContext = collectionCanvas.getContext('2d', { alpha: false });
    collectionContext.drawImage(video, 0, 0, collectionCanvas.width, collectionCanvas.height);
    const collectionBlob = await canvasToBlob(
      collectionCanvas,
      'image/jpeg',
      COLLECTION_JPEG_QUALITY
    );

    const size = scaledSize(collectionCanvas.width, collectionCanvas.height);
    canvas.width = size.width;
    canvas.height = size.height;
    const context = canvas.getContext('2d', { alpha: false });
    context.drawImage(collectionCanvas, 0, 0, canvas.width, canvas.height);
    selectedBlob = await canvasToBlob(canvas, 'image/jpeg', JPEG_QUALITY);
    showPreview(collectionBlob);
    stopCamera();
    classificationSucceeded = await requestClassification(
      selectedBlob,
      collectionBlob,
      collectionBlob
    );
  } catch (error) {
    console.error(error);
    showToast(error.message || 'Could not capture or analyze the image.');
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
  learningMemoryBadge.textContent = `🧠 Referenced ${matched} confirmed samples in shared memory · highest similarity ${similarity}%`;
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
    learningExampleCount.textContent = `Shared learning memory: ${total} samples · ${corrected} AI corrections`;
  } catch (error) {
    console.warn('Could not load learning statistics:', error);
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
  if (!response.ok) throw new Error(payload.detail || 'Could not save feedback.');
  return payload;
}

function updateResultConfidenceLabel(result) {
  const label = document.getElementById('resultConfidenceLabel');
  if (!label) return;
  const correctedPrediction = Boolean(result?.corrected_key) && result?.is_correct === false;
  const memoryAdjusted = Boolean(
    result?.learning?.memory_applied || result?.effective_prediction?.memory_applied
  );
  if (correctedPrediction) {
    label.textContent = memoryAdjusted
      ? 'Match score before your correction'
      : 'Initial AI prediction confidence';
  } else {
    label.textContent = memoryAdjusted
      ? 'Match score after memory adjustment'
      : 'AI match score';
  }
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
  document.getElementById('resultIcon').textContent = currentResult.icon;
  const categoryEl = document.getElementById('resultCategory');
  if (currentResult.uncertain && !hasUserLabel) {
    categoryEl.textContent = `${currentResult.category} · Low confidence`;
  } else {
    categoryEl.textContent = currentResult.category;
  }
  document.getElementById('resultName').textContent = currentResult.display_name;
  document.getElementById('resultBin').textContent = currentResult.bin_name;
  document.getElementById('resultInstruction').textContent = currentResult.instruction;
  updateResultConfidenceLabel(currentResult);
  renderObjectSelector();
  renderDetectionOverlay();
}

async function submitFeedback(correctKey) {
  if (!currentResult?.scan_id || feedbackSubmitting || !correctKey) return;
  const scanId = currentResult.scan_id;
  feedbackSubmitting = true;
  updateControlState();
  if (feedbackStatus) feedbackStatus.textContent = 'Saving feedback...';
  try {
    const payload = await postFeedback(scanId, correctKey);
    // Controls are locked while feedback is in-flight, but keep this identity
    // check as a second guard against stale async responses.
    if (currentResult?.scan_id === scanId) {
      // Keep the effective feedback label in sync with the server. Without this,
      // pressing "Correct" after a correction would submit the original AI key
      // again and silently undo the user's correction.
      applyFeedbackResult(payload);
      populateFeedbackCategories(payload.corrected_key);
      if (feedbackStatus) feedbackStatus.textContent = payload.message || 'Feedback saved.';
      if (feedbackCorrection) feedbackCorrection.hidden = true;
    }
    showToast(payload.message || 'Feedback saved.');
    void loadHistory({ reset: true });
    void refreshLearningStats();
  } catch (error) {
    console.error(error);
    if (feedbackStatus) feedbackStatus.textContent = error.message || 'Could not save feedback.';
    showToast(error.message || 'Could not save feedback.');
  } finally {
    feedbackSubmitting = false;
    updateControlState();
  }
}

// --- Render Multi-object Result ---
function activeObject() {
  return currentObjects[activeObjectIndex] || null;
}

function renderObjectSelector() {
  if (!objectResults) return;
  if (!currentObjects.length) {
    objectResults.replaceChildren();
    objectResults.hidden = true;
    return;
  }

  const buttons = currentObjects.map((item, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `object-result-button${index === activeObjectIndex ? ' active' : ''}`;
    button.dataset.objectIndex = String(index);

    const title = document.createElement('div');
    title.className = 'object-result-title';
    const name = document.createElement('span');
    name.textContent = `${index + 1}. ${visibleObjectName(item)}`;
    const score = document.createElement('strong');
    score.textContent = `${Math.round(Number(item.confidence || 0) * 100)}%`;
    title.append(name, score);

    const meta = document.createElement('div');
    meta.className = 'object-result-meta';
    if (item.detected === false || item.fallback_full_frame) {
      const label = item.fallback_full_frame ? 'Full-frame fallback' : 'No detector bounding box';
      meta.classList.add('object-result-warning');
      meta.textContent = label;
    } else {
      const detectorScore = item.detector_confidence == null
        ? ''
        : ` · detection ${Math.round(Number(item.detector_confidence) * 100)}%`;
      const uncertainLabel = item.uncertain ? ' · Low confidence' : '';
      meta.textContent = `${item.category || 'Unknown'}${uncertainLabel}${detectorScore}`;
    }

    button.append(title, meta);
    button.addEventListener('click', () => selectObject(index));
    return button;
  });

  objectResults.replaceChildren(...buttons);
  objectResults.hidden = currentObjects.length <= 1;
}

function renderDetectionOverlay() {
  if (!detectionOverlay) return;
  detectionOverlay.replaceChildren();

  if (preview.hidden || !preview.src || !preview.naturalWidth || !preview.naturalHeight) {
    detectionOverlay.hidden = true;
    return;
  }

  const detectedObjects = currentObjects.filter(item => item.detected !== false && item.bbox);
  if (!detectedObjects.length) {
    detectionOverlay.hidden = true;
    return;
  }

  const shell = preview.parentElement;
  if (!shell) {
    detectionOverlay.hidden = true;
    return;
  }

  const containerWidth = shell.clientWidth;
  const containerHeight = shell.clientHeight;
  if (!containerWidth || !containerHeight) {
    detectionOverlay.hidden = true;
    return;
  }

  // Phones use object-fit: contain to keep the full wide frame visible.
  // Desktop/laptop keeps object-fit: cover. Match the active CSS geometry here.
  const scaleForWidth = containerWidth / preview.naturalWidth;
  const scaleForHeight = containerHeight / preview.naturalHeight;
  const scale = PHONE_CAMERA_MODE
    ? Math.min(scaleForWidth, scaleForHeight)
    : Math.max(scaleForWidth, scaleForHeight);
  const renderedWidth = preview.naturalWidth * scale;
  const renderedHeight = preview.naturalHeight * scale;
  const offsetX = (containerWidth - renderedWidth) / 2;
  const offsetY = (containerHeight - renderedHeight) / 2;

  currentObjects.forEach((item, index) => {
    if (item.detected === false || !item.bbox) return;
    const bbox = item.bbox;
    const rawLeft = offsetX + Number(bbox.x1 || 0) * renderedWidth;
    const rawTop = offsetY + Number(bbox.y1 || 0) * renderedHeight;
    const rawRight = offsetX + Number(bbox.x2 || 0) * renderedWidth;
    const rawBottom = offsetY + Number(bbox.y2 || 0) * renderedHeight;

    const left = Math.max(0, Math.min(containerWidth, rawLeft));
    const top = Math.max(0, Math.min(containerHeight, rawTop));
    const right = Math.max(0, Math.min(containerWidth, rawRight));
    const bottom = Math.max(0, Math.min(containerHeight, rawBottom));
    if (right - left < 2 || bottom - top < 2) return;

    const box = document.createElement('div');
    box.className = `detection-box${index === activeObjectIndex ? ' active' : ''}`;
    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
    box.style.width = `${right - left}px`;
    box.style.height = `${bottom - top}px`;
    box.dataset.objectIndex = String(index);
    box.setAttribute('role', 'button');
    box.setAttribute('aria-label', `Select object ${index + 1}: ${visibleObjectName(item, 'object')}`);
    box.tabIndex = 0;

    const label = document.createElement('span');
    label.className = `detection-box-label${top < 28 ? ' inside' : ''}`;
    label.textContent = `${index + 1}. ${visibleObjectName(item)} · ${Math.round(Number(item.confidence || 0) * 100)}%`;
    box.append(label);

    const activate = () => selectObject(index);
    box.addEventListener('click', activate);
    box.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activate();
      }
    });
    detectionOverlay.append(box);
  });

  detectionOverlay.hidden = detectionOverlay.childElementCount === 0;
}

function renderActiveObject(result) {
  currentResult = result;
  if (!result) return;

  if (feedbackStatus) {
    if (result.source === 'demo') {
      feedbackStatus.textContent = 'Sample images are for demonstration only; they are not saved to history, used for feedback learning, or added to the dataset.';
    } else if (result.history_saved === false || !result.scan_id) {
      feedbackStatus.textContent = 'This object was not saved to history, so feedback is unavailable.';
    } else if (result.learning?.enabled === false) {
      feedbackStatus.textContent = 'You can confirm or correct the selected object; feedback learning is currently disabled.';
    } else if (result.learning?.feedback_available === false) {
      feedbackStatus.textContent = 'You can still provide feedback for the selected object, but it has no feature vector to use as a learning sample.';
    } else {
      feedbackStatus.textContent = 'Confirm or correct the selected object so the AI can learn from this exact crop.';
    }
  }
  if (feedbackCorrection) feedbackCorrection.hidden = true;
  populateFeedbackCategories(result.corrected_key || result.key);
  renderLearningMemory(result);

  const hasUserLabel = Boolean(result.corrected_key);
  document.getElementById('resultIcon').textContent = result.icon;
  const categoryEl = document.getElementById('resultCategory');
  if (result.uncertain && !hasUserLabel) {
    categoryEl.textContent = `${result.category} · Low confidence`;
    categoryEl.title = (result.uncertainty_reasons || []).join(', ') || 'AI confidence is below the threshold';
  } else {
    categoryEl.textContent = result.category;
    categoryEl.title = '';
  }
  document.getElementById('resultName').textContent = visibleObjectName(result);
  document.getElementById('resultBin').textContent = result.bin_name;
  document.getElementById('resultInstruction').textContent = result.instruction;
  document.getElementById('resultNotice').textContent = result.notice;

  const percentage = Math.round(Number(result.confidence || 0) * 100);
  document.getElementById('resultConfidence').textContent = `${percentage}%`;
  updateResultConfidenceLabel(result);

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

  updateControlState();
}

function selectObject(index, { scroll = false } = {}) {
  if (!Number.isInteger(index) || index < 0 || index >= currentObjects.length) return;
  activeObjectIndex = index;
  renderActiveObject(currentObjects[index]);
  renderObjectSelector();
  renderDetectionOverlay();
  if (scroll) resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderResult(payload) {
  currentResponse = payload;
  const payloadObjects = Array.isArray(payload.objects) ? payload.objects : null;

  if (payload.no_detection === true || (payloadObjects && payloadObjects.length === 0)) {
    currentObjects = [];
    currentResult = null;
    activeObjectIndex = 0;
    renderObjectSelector();
    if (detectionOverlay) {
      detectionOverlay.replaceChildren();
      detectionOverlay.hidden = true;
    }
    if (resultCard) resultCard.hidden = true;
    if (objectSummary) {
      objectSummary.textContent = payload.notice
        || 'No waste object was detected clearly. Move the object closer to the camera and capture another image.';
      objectSummary.hidden = false;
    }
    resultSection.hidden = false;
    updateControlState();
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }

  currentObjects = payloadObjects?.length ? payloadObjects : [payload];
  activeObjectIndex = 0;
  if (resultCard) resultCard.hidden = false;

  if (objectSummary) {
    const fallbackFullFrame = payload.detector?.fallback_full_frame === true
      || payload.fallback_full_frame === true;
    if (fallbackFullFrame) {
      objectSummary.textContent = 'No eligible object was detected. The full image was classified instead.';
    } else {
      const count = currentObjects.length;
      objectSummary.textContent = `Detected ${count} object${count === 1 ? '' : 's'}. Select a box or card below to inspect and provide feedback for each object individually.`;
    }
    objectSummary.hidden = false;
  }

  resultSection.hidden = false;
  selectObject(0);
  requestAnimationFrame(renderDetectionOverlay);
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

preview?.addEventListener('load', () => requestAnimationFrame(renderDetectionOverlay));
window.addEventListener('resize', () => {
  if (currentObjects.length) requestAnimationFrame(renderDetectionOverlay);
});


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

async function processSelectedFile(file, options = {}) {
  if (isBusy || isHistoryBusy || feedbackSubmitting) return;
  clearResult();
  setBusy(true);
  let classificationSucceeded = false;
  try {
    stopCamera();
    showPreview(file);
    const shouldPersist = options.persist !== false;
    const [inferenceBlob, classifierBlob] = await Promise.all([
      optimizeUploadedImage(file),
      prepareHighQualityImage(file)
    ]);
    selectedBlob = inferenceBlob;
    classificationSucceeded = await requestClassification(
      inferenceBlob,
      classifierBlob,
      shouldPersist ? classifierBlob : null,
      options
    );
  } catch (error) {
    console.error(error);
    showToast(error.message || 'Could not read or analyze the image.');
  } finally {
    setBusy(false);
    if (!classificationSucceeded) await recoverScannerAfterFailure();
  }
}

// --- Sample Quick Tests ---
const SAMPLES = {
  plastic_rigid: '/static/samples/plastic-bottle.jpg',
  cardboard: '/static/samples/cardboard-box.jpg',
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
      if (!response.ok) throw new Error('Could not load the sample image.');
      const blob = await response.blob();
      await processSelectedFile(blob, { source: 'demo', persist: false });
    } catch (error) {
      console.error(error);
      showToast(error.message || 'Could not open the sample image.');
    }
  });
});

// --- Waste Catalog Dynamic Loader ---
async function loadCatalog() {
  if (!catalogGrid) return;
  try {
    const res = await fetch('/api/categories');
    if (!res.ok) throw new Error(`Server returned error ${res.status}.`);
    const data = await res.json();
    catalogItems = Array.isArray(data) ? data : [];
    populateFeedbackCategories(currentResult?.corrected_key || currentResult?.key || '');
    // History can finish loading before the catalog request. Re-render once the
    // localized category names are available so corrected keys never remain as
    // raw sample ids until the next refresh.
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
    console.error('Error loading catalog:', err);
    const message = document.createElement('div');
    message.className = 'catalog-loading';
    message.textContent = 'Could not load the classification catalog. Reload the page or check your connection.';
    catalogGrid.replaceChildren(message);
  }
}

// --- History Drawer & Statistics ---
function formatDate(value) {
  const date = new Date(value);
  return new Intl.DateTimeFormat('en-US', {
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
  if (!key) return 'Unconfirmed';
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
    console.warn(`Could not load thumbnail for scan ${item.id}:`, error);
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
    placeholderElement.textContent = 'Image could not be loaded';
  }
}

async function openHistoryEditor(item) {
  if (!historyEditModal || !item) return;
  if (!catalogItems.length) await loadCatalog();
  historyEditItem = item;
  const effectiveKey = item.corrected_key || item.waste_key;
  populateCategorySelect(historyEditCategory, effectiveKey);

  if (historyEditObjectName) {
    historyEditObjectName.textContent = item.display_name;
  }
  const modelDisplayName = item.model_display_name || item.display_name;
  const modelCategory = item.model_category || item.category;
  const modelUncertain = item.model_uncertain ?? item.uncertain;
  const modelConfidence = Number(item.model_confidence ?? item.confidence ?? 0);
  if (historyEditPredicted) {
    historyEditPredicted.textContent = `${modelDisplayName} · ${modelCategory}${modelUncertain ? ' · AI uncertain' : ''}`;
  }
  if (historyEditConfidence) {
    historyEditConfidence.textContent = `${modelUncertain ? '~' : ''}${Math.round(modelConfidence * 100)}%`;
  }
  if (historyEditTime) historyEditTime.textContent = formatDate(item.created_at);
  if (historyEditCurrentLabel) {
    historyEditCurrentLabel.textContent = item.corrected_key
      ? displayNameForKey(item.corrected_key)
      : (item.memory_applied
          ? 'Unconfirmed · using the memory-adjusted result'
          : 'Unconfirmed · using the AI prediction');
  }
  if (historyEditStatus) {
    historyEditStatus.textContent = item.thumbnail_available
      ? 'Review the image and select the correct label. Saving will update learning memory immediately.'
      : 'This record no longer has a thumbnail. You can still edit the label if you recognize the scan from the nearby details.';
  }
  if (historyEditImage) {
    historyEditImage.removeAttribute('src');
    historyEditImage.hidden = true;
  }
  if (historyEditImagePlaceholder) {
    historyEditImagePlaceholder.textContent = item.thumbnail_available ? 'Loading image...' : 'No thumbnail is available for this older scan';
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
  if (historyEditStatus) historyEditStatus.textContent = 'Updating label...';
  try {
    const payload = await postFeedback(historyEditItem.id, correctKey);
    historyEditItem.corrected_key = payload.corrected_key;
    historyEditItem.is_correct = payload.is_correct;
    const liveObjectIndex = currentObjects.findIndex(object => object?.scan_id === historyEditItem.id);
    if (liveObjectIndex >= 0) {
      activeObjectIndex = liveObjectIndex;
      currentResult = currentObjects[liveObjectIndex];
      applyFeedbackResult(payload);
      populateFeedbackCategories(payload.corrected_key);
    }
    const cached = cachedHistoryItems.find(item => item.id === historyEditItem.id);
    if (cached) {
      cached.corrected_key = payload.corrected_key;
      cached.is_correct = payload.is_correct;
    }
    if (historyEditCurrentLabel) historyEditCurrentLabel.textContent = displayNameForKey(payload.corrected_key);
    if (historyEditStatus) historyEditStatus.textContent = payload.message || 'Label updated.';
    await loadHistory({ reset: true });
    void refreshLearningStats();
    showToast(payload.message || `Label updated to ${displayNameForKey(payload.corrected_key)}.`);
  } catch (error) {
    console.error(error);
    if (historyEditStatus) historyEditStatus.textContent = error.message || 'Could not update the record.';
    showToast(error.message || 'Could not update the record.');
  } finally {
    feedbackSubmitting = false;
    updateControlState();
  }
}

function requestHistoryDeletePassword() {
  const password = window.prompt('Enter the history deletion password:');
  if (password === null) return null;
  if (!password.trim()) {
    showToast('Incorrect password.');
    return null;
  }
  return password;
}

async function deleteHistoryItem(item) {
  if (!item?.id || isBusy || isHistoryBusy || feedbackSubmitting) return;

  const itemName = item.display_name;
  const confirmed = window.confirm(
    `Delete “${itemName}” from history? The image, feedback, and learning data for this scan will also be deleted. This action cannot be undone.`
  );
  if (!confirmed) return;
  const deletePassword = requestHistoryDeletePassword();
  if (deletePassword === null) return;

  setHistoryBusy(true);
  try {
    const response = await fetch(`/api/history/${item.id}`, {
      method: 'DELETE',
      headers: { 'X-Delete-Password': deletePassword }
    });
    let payload = {};
    try { payload = await response.json(); } catch (_) { /* ignore */ }
    if (!response.ok) {
      if (response.status === 401) throw new Error('Incorrect password.');
      if (response.status === 503) throw new Error(payload.detail || 'History service is temporarily unavailable.');
      throw new Error(payload.detail || 'Could not delete this scan.');
    }

    releaseHistoryThumbnailUrl(item.id);
    if (historyEditItem?.id === item.id) {
      historyEditModal.hidden = true;
      historyEditModal.setAttribute('aria-hidden', 'true');
      historyEditItem = null;
    }
    const liveObjectIndex = currentObjects.findIndex(object => object?.scan_id === item.id);
    if (liveObjectIndex >= 0) {
      // Keep the detected object visible, but invalidate only this object's
      // DB-backed feedback identity. Other objects from the same camera frame
      // remain independently editable.
      currentObjects[liveObjectIndex].scan_id = null;
      currentObjects[liveObjectIndex].history_saved = false;
      if (activeObjectIndex === liveObjectIndex) {
        currentResult = currentObjects[liveObjectIndex];
        if (feedbackCorrection) feedbackCorrection.hidden = true;
        if (feedbackStatus) feedbackStatus.textContent = 'This object was deleted from history; scan it again if you want to provide feedback.';
      }
      renderObjectSelector();
      updateControlState();
    }

    cachedHistoryItems = cachedHistoryItems.filter(entry => entry.id !== item.id);
    await loadHistory({ reset: true });
    await refreshLearningStats();

    showToast(`Deleted object #${item.id}. Used IDs will not be reused to prevent data from being matched incorrectly.`);
  } catch (error) {
    console.error(error);
    showToast(error.message || 'Could not delete this scan.');
  } finally {
    setHistoryBusy(false);
  }
}

function renderHistoryItems(items) {
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-history';
    empty.textContent = (historySearch?.value || '').trim()
      ? 'No matching history found.'
      : 'No waste scan history yet.';
    historyList.replaceChildren(empty);
    return;
  }

  historyList.replaceChildren(...items.map(item => {
    const row = document.createElement('article');
    row.className = 'history-item';

    const thumbButton = document.createElement('button');
    thumbButton.className = 'history-thumb-button';
    thumbButton.type = 'button';
    thumbButton.setAttribute('aria-label', `View image and edit label for ${item.display_name}`);
    thumbButton.addEventListener('click', () => { void openHistoryEditor(item); });

    const thumb = document.createElement('img');
    thumb.className = 'history-thumb';
    thumb.alt = '';
    thumb.hidden = true;
    const thumbPlaceholder = document.createElement('span');
    thumbPlaceholder.className = 'history-thumb-placeholder';
    thumbPlaceholder.textContent = item.thumbnail_available ? '…' : 'No image';
    thumbButton.append(thumb, thumbPlaceholder);
    void attachHistoryThumbnail(item, thumb, thumbPlaceholder);

    const details = document.createElement('div');
    details.className = 'history-item-info';

    const title = document.createElement('strong');
    title.textContent = item.display_name;

    const meta = document.createElement('small');
    const feedbackText = item.corrected_key
      ? (item.is_correct ? ' · Confirmed correct' : ' · Label corrected')
      : ' · Unconfirmed';
    meta.textContent = `${item.category}${item.uncertain ? ' · Uncertain' : ''}${feedbackText} · ${formatDate(item.created_at)}`;

    const itemActions = document.createElement('div');
    itemActions.className = 'history-item-actions';

    const editButton = document.createElement('button');
    editButton.className = 'history-edit-button';
    editButton.type = 'button';
    editButton.textContent = item.corrected_key ? 'View / edit again' : 'View / edit label';
    editButton.addEventListener('click', () => { void openHistoryEditor(item); });

    const deleteButton = document.createElement('button');
    deleteButton.className = 'history-delete-button';
    deleteButton.type = 'button';
    deleteButton.textContent = 'Delete';
    deleteButton.setAttribute('aria-label', `Delete ${item.display_name} from history`);
    deleteButton.addEventListener('click', () => { void deleteHistoryItem(item); });

    itemActions.append(editButton, deleteButton);
    details.append(title, meta, itemActions);

    const score = document.createElement('span');
    score.className = 'history-score';
    // The list title is the current effective/user-confirmed label. Therefore a
    // numeric score is only meaningful when it belongs to that same AI label.
    // If the user corrected the item to another class, show correction state
    // instead of attaching the original model confidence to the new label.
    if (item.corrected_key && item.is_correct === false) {
      score.textContent = 'Corrected';
    } else {
      const effectiveScore = Number(item.effective_score ?? item.confidence ?? 0);
      const effectiveUncertain = item.effective_uncertain ?? item.uncertain;
      score.textContent = `${effectiveUncertain ? '~' : ''}${Math.round(effectiveScore * 100)}%`;
    }

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
    if (!response.ok) throw new Error(payload.detail || 'Could not load history.');
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
      showToast(error.message || 'Could not load more history.');
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
        ? 'Could not switch cameras; restored the previous camera.'
        : 'Could not switch cameras and could not restore the previous camera.');
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
    'Clear all shared history? This deletes thumbnails, learning memory, and dataset images collected from these scans. Used IDs will not be reused to prevent incorrect data matching. This action cannot be undone.'
  );
  if (!confirmed) return;
  const deletePassword = requestHistoryDeletePassword();
  if (deletePassword === null) return;
  setHistoryBusy(true);
  try {
    const response = await fetch('/api/history', {
      method: 'DELETE',
      headers: { 'X-Delete-Password': deletePassword }
    });
    let payload = {};
    try { payload = await response.json(); } catch (_) { /* ignore */ }
    if (!response.ok) {
      if (response.status === 401) throw new Error('Incorrect password.');
      if (response.status === 503) throw new Error(payload.detail || 'History service is temporarily unavailable.');
      throw new Error(payload.detail || 'Could not clear history.');
    }
    releaseHistoryThumbnailUrls();
    closeHistoryEditor();
    // Keep the visible multi-object result for reference, but invalidate every
    // DB-backed scan_id because the shared history was cleared.
    currentObjects.forEach(object => {
      object.scan_id = null;
      object.history_saved = false;
    });
    currentResult = activeObject();
    if (feedbackCorrection) feedbackCorrection.hidden = true;
    if (feedbackStatus) feedbackStatus.textContent = currentResult
      ? 'History has been cleared; scan this object again if you want to provide feedback.'
      : '';
    renderObjectSelector();
    updateControlState();
    await loadHistory({ reset: true });
    await refreshLearningStats();
    showToast('History cleared. Previous IDs are retained as markers and will not be reused.');
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
  console.warn('The browser is blocking storage and cookies; the device ID will only remain stable for this session. Shared history is unaffected.');
} else if (!clientIdentity.fullySynced) {
  console.warn('The client ID could only be stored using one browser mechanism; it is device metadata only, and shared history is unaffected.');
}
renderHealthStatus('checking', 'Checking AI', 'Checking AI system status.');
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

