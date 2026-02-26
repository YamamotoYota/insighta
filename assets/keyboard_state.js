// Copyright (c) 2026 Yota Yamamoto
// SPDX-License-Identifier: MIT

(function () {
  if (window.__insightaKeyboardInitialized__) {
    return;
  }
  window.__insightaKeyboardInitialized__ = true;
  window.insightaKeyboardState = { ctrl: false, meta: false };

  function updateFromEvent(event) {
    window.insightaKeyboardState = {
      ctrl: !!event.ctrlKey,
      meta: !!event.metaKey,
    };
  }

  function clearState() {
    window.insightaKeyboardState = { ctrl: false, meta: false };
  }

  window.addEventListener("keydown", updateFromEvent, true);
  window.addEventListener("keyup", updateFromEvent, true);
  window.addEventListener("blur", clearState, true);

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState !== "visible") {
      clearState();
    }
  });
})();
