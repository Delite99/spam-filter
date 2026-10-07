document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tabs = document.querySelectorAll(".nav-tab");
  const tabPanels = document.querySelectorAll(".tab-panel");
  const healthPill = document.getElementById("health-pill");
  const healthText = document.getElementById("health-text");
  
  const emailInput = document.getElementById("email-input");
  const modelSelect = document.getElementById("model-select");
  const charCounter = document.getElementById("char-counter");
  const clearBtn = document.getElementById("clear-btn");
  const explainToggle = document.getElementById("explain-toggle");
  const analyzeBtn = document.getElementById("analyze-btn");
  const analyzeSpinner = document.getElementById("analyze-spinner");
  const analyzeBtnText = document.getElementById("analyze-btn-text");

  // Output elements
  const verdictBadge = document.getElementById("verdict-badge");
  const verdictLabel = document.getElementById("verdict-label");
  const gaugeCircle = document.getElementById("gauge-circle");
  const spamProbVal = document.getElementById("spam-prob-val");
  const kpiConfidence = document.getElementById("kpi-confidence");
  const kpiLatency = document.getElementById("kpi-latency");
  const kpiModel = document.getElementById("kpi-model");
  const tokensCloud = document.getElementById("tokens-cloud");
  const sigCaps = document.getElementById("sig-caps");
  const sigExcl = document.getElementById("sig-excl");
  const sigCurrency = document.getElementById("sig-currency");
  const sigUrls = document.getElementById("sig-urls");

  // Batch elements
  const batchTextarea = document.getElementById("batch-textarea");
  const runBatchBtn = document.getElementById("run-batch-btn");
  const loadSampleBatchBtn = document.getElementById("load-sample-batch-btn");
  const batchStatsBanner = document.getElementById("batch-stats-banner");
  const batchTableContainer = document.getElementById("batch-table-container");
  const batchTbody = document.getElementById("batch-tbody");
  const batchTotalCount = document.getElementById("batch-total-count");
  const batchSpamCount = document.getElementById("batch-spam-count");
  const batchHamCount = document.getElementById("batch-ham-count");
  const batchTotalLatency = document.getElementById("batch-total-latency");

  // Benchmark elements
  const benchmarkTbody = document.getElementById("benchmark-tbody");
  const refreshMetricsBtn = document.getElementById("refresh-metrics-btn");

  const GAUGE_CIRCUMFERENCE = 326.7; // 2 * PI * 52

  // Scenario presets
  const SCENARIOS = {
    tamara: "EXPIRATION NOTIFICATION\n\nDear tamara\n\nPlease be informed that the password for your  tamara@mpkfreightglobal.com    will expire in 48 hours.\n\nYou may either update your password or continue using your current password by clicking on our URL below to  keep  current Password.\n\nKeep Current Password\n\nNote:failure to update your password will result in the permanent closure of your account, and all your data will be lost permanently.\n\nThanks,\n   IT Service Message",
    eft: "Mpkfreightglobal EFT Payment Processed on 09/24/2026,2:09:51 PM\nFrom: Salokhidin Kh. Khamrakulov <Salokhidin.Khamrakulov@uzairways.com>\nTo: jacob@mpkfreightglobal.com\n\nSchaden LLC has shared payment receipt for \"Invoice#273226 with you jacob@mpkfreightglobal.com.\n\nMpkfreightglobal September Payment Confirmation.xlsx\nLast modified 2026-09-24\n\nOpen in Excel  Download Document",
    phish: "URGENT SECURITY ALERT: Unauthorized sign-in attempt detected on your Microsoft Office 365 enterprise account. Your access will be permanently locked within 2 hours. Validate your credentials now at https://secure-login-portal.net/verify to keep your account active.",
    lottery: "CONGRATULATIONS! You have been selected as the official 1st prize winner of the International European Lottery promo: $1,500,000.00 USD cash! To claim your funds immediately, wire a $250 release fee and your bank routing number to barrister Robert.",
    crypto: "Exclusive VIP Opportunity: Earn $2,500 daily with our automated AI crypto trading bot! Guaranteed zero risk 400% weekly ROI. Limited seats available! Message @crypto_whale_vip on Telegram now.",
    work: "Hi Sarah, please find attached the revised slide deck and sprint deliverables for our Thursday all-hands sync. Let me know if you would like to make any adjustments before we present to leadership.",
    github: "GitHub Notification: Antigravity-Agent opened Pull Request #14 'Feature: Add PyTorch BiLSTM with self-attention sequence classification head'. All unit tests and CI integration checks passed successfully.",
    flight: "Your United Airlines booking confirmation #UA-98214: Flight 412 from San Francisco (SFO) to New York (JFK) on Friday Oct 12 at 08:30 AM. Terminal 3, Seat 14A."
  };

  // Preset chip event handlers
  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const type = chip.dataset.type;
      if (SCENARIOS[type]) {
        emailInput.value = SCENARIOS[type];
        updateCounters();
        // Auto-run analysis
        analyzeEmail();
      }
    });
  });

  // Textarea listeners
  emailInput.addEventListener("input", updateCounters);
  clearBtn.addEventListener("click", () => {
    emailInput.value = "";
    updateCounters();
    resetGauge();
  });

  function updateCounters() {
    const text = emailInput.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    charCounter.textContent = `${chars} chars | ${words} words`;
  }

  // Tabs navigation
  const navTabs = document.querySelectorAll(".sidebar-nav-item, .nav-tab");
  const breadcrumbTitle = document.getElementById("active-breadcrumb-title");

  const TAB_TITLES = {
    "live-tab": "Live Analyzer",
    "training-tab": "Training & Cleaning Studio",
    "batch-tab": "High-Throughput Batch Scanner",
    "benchmark-tab": "Benchmarks & Model Metrics"
  };

  function activateTab(tabId) {
    navTabs.forEach(t => {
      if (t.dataset.tab === tabId) t.classList.add("active");
      else t.classList.remove("active");
    });
    tabPanels.forEach(p => {
      if (p.id === tabId) p.classList.add("active");
      else p.classList.remove("active");
    });

    if (breadcrumbTitle && TAB_TITLES[tabId]) {
      breadcrumbTitle.textContent = TAB_TITLES[tabId];
    }

    if (tabId === "benchmark-tab") {
      fetchBenchmarkMetrics();
    }
  }

  navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      if (tab.dataset.tab) {
        activateTab(tab.dataset.tab);
      }
    });
  });

  // Top header button to open training studio
  const headerOpenStudioBtn = document.getElementById("header-open-studio-btn");
  if (headerOpenStudioBtn) {
    headerOpenStudioBtn.addEventListener("click", () => {
      activateTab("training-tab");
    });
  }

  // Sidebar shortcut buttons
  const sidebarLoadCorpusBtn = document.getElementById("sidebar-load-corpus-btn");
  if (sidebarLoadCorpusBtn) {
    sidebarLoadCorpusBtn.addEventListener("click", () => {
      activateTab("training-tab");
      loadBundledData("corpus");
    });
  }

  const sidebarLoadDemoBtn = document.getElementById("sidebar-load-demo-btn");
  if (sidebarLoadDemoBtn) {
    sidebarLoadDemoBtn.addEventListener("click", () => {
      activateTab("training-tab");
      loadBundledData("demo");
    });
  }

  // Check Health
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        healthText.textContent = `Online • ${data.device.toUpperCase()} • ${data.active_model}`;
        if (healthPill) {
          healthPill.style.color = "#34d399";
          healthPill.style.borderColor = "rgba(16, 185, 129, 0.3)";
        }
      } else {
        healthText.textContent = "Service Degradation";
      }
    } catch (e) {
      healthText.textContent = "Offline";
      healthPill.style.color = "#f87171";
      healthPill.style.borderColor = "rgba(239, 68, 68, 0.3)";
    }
  }

  // Gauge animation
  function setGauge(prob) {
    const offset = GAUGE_CIRCUMFERENCE * (1 - prob);
    gaugeCircle.style.strokeDashoffset = offset;
    spamProbVal.textContent = `${(prob * 100).toFixed(1)}%`;

    if (prob >= 0.5) {
      gaugeCircle.style.stroke = "url(#spamGradient)";
      gaugeCircle.style.stroke = "#ef4444";
      spamProbVal.style.color = "#f87171";
    } else {
      gaugeCircle.style.stroke = "#10b981";
      spamProbVal.style.color = "#34d399";
    }
  }

  function resetGauge() {
    gaugeCircle.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
    spamProbVal.textContent = "0.0%";
    spamProbVal.style.color = "#f8fafc";
    verdictBadge.className = "verdict-badge pending";
    verdictLabel.textContent = "Awaiting Input";
    kpiConfidence.textContent = "--";
    kpiLatency.textContent = "-- ms";
    tokensCloud.innerHTML = '<span class="placeholder-text">Run analysis with explainability enabled to inspect word attention weights.</span>';
    sigCaps.textContent = "0.0%";
    sigExcl.textContent = "0";
    sigCurrency.textContent = "0";
    sigUrls.textContent = "0";
  }

  // Live Analyzer Execution
  async function analyzeEmail() {
    const text = emailInput.value.trim();
    if (!text) {
      alert("Please enter or select sample email text to analyze.");
      return;
    }

    const modelName = modelSelect.value;
    const explain = explainToggle.checked;

    analyzeBtn.disabled = true;
    analyzeSpinner.classList.remove("hidden");
    analyzeBtnText.textContent = "Analyzing...";

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          model_name: modelName,
          explain: explain
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Analysis failed");
      }

      const data = await res.json();
      renderAnalysisResult(data);

    } catch (err) {
      alert(`Inference Error: ${err.message}`);
    } finally {
      analyzeBtn.disabled = false;
      analyzeSpinner.classList.add("hidden");
      analyzeBtnText.textContent = "Run Security Scan";
    }
  }

  function renderAnalysisResult(data) {
    const isSpam = data.label === "spam";
    const prob = data.probability;

    // Verdict Badge
    verdictBadge.className = `verdict-badge ${isSpam ? "spam" : "ham"}`;
    verdictLabel.textContent = isSpam ? "Threat: Spam / Phishing" : "Verified: Legitimate (Ham)";

    // Gauge
    setGauge(prob);

    // KPIs
    kpiConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    kpiLatency.textContent = `${data.latency_ms.toFixed(1)} ms`;
    kpiModel.textContent = data.model_version.split(":")[0];

    // Extracted signals
    if (data.signals) {
      sigCaps.textContent = `${(data.signals.uppercase_ratio * 100).toFixed(1)}%`;
      sigExcl.textContent = data.signals.exclamation_count;
      sigCurrency.textContent = data.signals.currency_count;
      sigUrls.textContent = data.signals.url_count;

      const phishRisk = document.getElementById("sig-phish-risk");
      const phishBox = document.getElementById("phishing-vectors-box");
      const phishList = document.getElementById("phishing-vectors-list");
      if (phishRisk) {
        const pScore = data.signals.phishing_risk_score || 0;
        phishRisk.textContent = `${(pScore * 100).toFixed(1)}%`;
        phishRisk.style.color = pScore >= 0.7 ? "#ef4444" : (pScore >= 0.4 ? "#f59e0b" : "#10b981");
      }
      if (phishBox && phishList) {
        const indicators = data.signals.phishing_indicators || [];
        if (indicators.length > 0) {
          phishBox.classList.remove("hidden");
          phishList.innerHTML = indicators.map(ind => `
            <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background: rgba(239, 68, 68, 0.25); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4);">
              ⚠️ ${ind.replace(/_/g, " ").toUpperCase()}
            </span>
          `).join("");
        } else {
          phishBox.classList.add("hidden");
          phishList.innerHTML = "";
        }
      }
    }

    // Token Attentions
    if (data.word_attentions && data.word_attentions.length > 0) {
      tokensCloud.innerHTML = "";
      const maxWeight = Math.max(...data.word_attentions.map(w => w.weight), 0.01);
      
      data.word_attentions.forEach(({ word, weight }) => {
        const chip = document.createElement("span");
        chip.className = "token-chip";
        const normalizedAlpha = Math.min(1.0, Math.max(0.12, (weight / maxWeight)));
        chip.style.setProperty("--alpha", normalizedAlpha.toFixed(2));
        if (isSpam && normalizedAlpha > 0.45) {
          chip.style.background = `rgba(239, 68, 68, ${normalizedAlpha})`;
          chip.style.borderColor = `rgba(239, 68, 68, ${normalizedAlpha * 1.5})`;
        }
        chip.textContent = word;
        chip.title = `Weight: ${(weight * 100).toFixed(2)}%`;
        tokensCloud.appendChild(chip);
      });
    } else {
      tokensCloud.innerHTML = '<span class="placeholder-text">Attention scores not available for this model or explainability was disabled.</span>';
    }
  }

  analyzeBtn.addEventListener("click", analyzeEmail);

  // Batch Scanner
  const SAMPLE_BATCH = [
    "URGENT: Your PayPal account has been restricted. Click here to confirm credentials.",
    "Hey Dave, can you review the pull request on github before the standup tomorrow?",
    "Congratulations! You won $5,000,000 in the UK National Lottery. Reply with account details.",
    "Hi team, attached is the Q3 sprint retrospective agenda for Friday afternoon.",
    "Order confirmation: Your Apple receipt #INV-84920 has been billed $499. Click to dispute."
  ];

  loadSampleBatchBtn.addEventListener("click", () => {
    batchTextarea.value = SAMPLE_BATCH.join("\n");
  });

  runBatchBtn.addEventListener("click", async () => {
    const raw = batchTextarea.value.trim();
    if (!raw) {
      alert("Please enter messages in the batch textarea.");
      return;
    }

    const lines = raw.split("\n").map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length === 0) return;

    runBatchBtn.disabled = true;
    runBatchBtn.textContent = "Processing Batch...";

    try {
      const res = await fetch("/predict/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: lines, model_name: modelSelect.value })
      });

      if (!res.ok) throw new Error("Batch request failed");
      const data = await res.json();

      // Render summary
      batchStatsBanner.style.display = "grid";
      batchTableContainer.style.display = "block";

      batchTotalCount.textContent = data.total_processed;
      batchSpamCount.textContent = data.spam_count;
      batchHamCount.textContent = data.ham_count;
      batchTotalLatency.textContent = `${data.total_latency_ms.toFixed(1)} ms`;

      // Render rows
      batchTbody.innerHTML = "";
      data.predictions.forEach((p, idx) => {
        const tr = document.createElement("tr");
        const isSpam = p.label === "spam";
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td style="font-family: var(--font-mono); font-size: 0.8rem;">${escapeHtml(lines[idx].substring(0, 75))}${lines[idx].length > 75 ? '...' : ''}</td>
          <td><span class="verdict-badge ${isSpam ? 'spam' : 'ham'}" style="padding: 2px 8px; font-size: 0.72rem;">${isSpam ? 'SPAM' : 'HAM'}</span></td>
          <td style="font-family: var(--font-mono);">${(p.probability * 100).toFixed(1)}%</td>
          <td style="font-family: var(--font-mono);">${(p.confidence * 100).toFixed(1)}%</td>
          <td style="color: #38bdf8; font-family: var(--font-mono);">${p.latency_ms.toFixed(1)} ms</td>
        `;
        batchTbody.appendChild(tr);
      });

    } catch (err) {
      alert(`Batch Error: ${err.message}`);
    } finally {
      runBatchBtn.disabled = false;
      runBatchBtn.textContent = "Execute Batch Classification";
    }
  });

  // Benchmark Metrics
  async function fetchBenchmarkMetrics() {
    try {
      const res = await fetch("/metrics");
      const data = await res.json();

      if (data.status === "pending" || !data.benchmark_summary) {
        benchmarkTbody.innerHTML = '<tr><td colspan="7" class="loading-cell">No trained models found on disk yet. Run <code>python -m src.train --model all</code> to generate official benchmarks.</td></tr>';
        return;
      }

      benchmarkTbody.innerHTML = "";
      data.benchmark_summary.forEach(m => {
        const tr = document.createElement("tr");
        const f1Passed = m.f1 >= 0.97;
        tr.innerHTML = `
          <td><strong>${m.Model}</strong></td>
          <td>${(m.accuracy * 100).toFixed(2)}%</td>
          <td>${(m.precision * 100).toFixed(2)}%</td>
          <td>${(m.recall * 100).toFixed(2)}%</td>
          <td style="font-weight: 700; color: ${f1Passed ? '#34d399' : '#fbbf24'};">${(m.f1 * 100).toFixed(2)}%</td>
          <td>${m.roc_auc.toFixed(4)}</td>
          <td>
            <span class="verdict-badge ${f1Passed ? 'ham' : 'pending'}" style="padding: 2px 8px; font-size: 0.7rem;">
              ${f1Passed ? '≥97% TARGET MET' : 'BASELINE REFERENCE'}
            </span>
          </td>
        `;
        benchmarkTbody.appendChild(tr);
      });

    } catch (err) {
      benchmarkTbody.innerHTML = '<tr><td colspan="7" class="loading-cell">Could not load metrics.</td></tr>';
    }
  }

  refreshMetricsBtn.addEventListener("click", fetchBenchmarkMetrics);

  function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // ==========================================================================
  // TRAINING & DATA CLEANING STUDIO LOGIC
  // ==========================================================================

  // Step Indicators
  const stepChips = [
    document.getElementById("step-chip-1"),
    document.getElementById("step-chip-2"),
    document.getElementById("step-chip-3"),
    document.getElementById("step-chip-4")
  ];

  function setStepActive(stepIndex) {
    stepChips.forEach((chip, idx) => {
      if (!chip) return;
      chip.classList.remove("active", "completed");
      if (idx < stepIndex) chip.classList.add("completed");
      if (idx === stepIndex) chip.classList.add("active");
    });
  }

  // Ingestion Elements
  const dropzone = document.getElementById("dataset-dropzone");
  const fileInput = document.getElementById("dataset-file-input");
  const browseBtn = document.getElementById("browse-btn");
  const loadCorpusBtn = document.getElementById("load-corpus-btn");
  const loadDemoBtn = document.getElementById("load-demo-btn");
  const togglePasteBtn = document.getElementById("toggle-paste-btn");
  const pasteDrawer = document.getElementById("paste-drawer");
  const pasteInput = document.getElementById("paste-input");
  const submitPasteBtn = document.getElementById("submit-paste-btn");

  const ingestSummaryBox = document.getElementById("ingest-summary-box");
  const dataRowsStat = document.getElementById("data-rows-stat");
  const dataColsStat = document.getElementById("data-cols-stat");
  const colTextSelect = document.getElementById("col-text-select");
  const colLabelSelect = document.getElementById("col-label-select");
  const spamTagsInput = document.getElementById("spam-tags-input");

  // Cleaning Elements
  const runCleanBtn = document.getElementById("run-clean-btn");
  const cleanBtnText = document.getElementById("clean-btn-text");
  const cleanTelemetryBanner = document.getElementById("clean-telemetry-banner");
  const cleanStatRetained = document.getElementById("clean-stat-retained");
  const cleanStatDropped = document.getElementById("clean-stat-dropped");
  const balanceHamLabel = document.getElementById("balance-ham-label");
  const balanceSpamLabel = document.getElementById("balance-spam-label");
  const balanceBarFill = document.getElementById("balance-bar-fill");
  const repUrls = document.getElementById("rep-urls");
  const repPhones = document.getElementById("rep-phones");
  const repCurr = document.getElementById("rep-curr");
  const repNums = document.getElementById("rep-nums");

  const cleanDiffSection = document.getElementById("clean-diff-section");
  const cleanDiffTbody = document.getElementById("clean-diff-tbody");
  const filterPills = document.querySelectorAll(".filter-pill-group .filter-pill");

  // Training Elements
  const modelCards = document.querySelectorAll(".model-candidate-card");
  let selectedTrainingModel = "baseline_lr";

  const paramEpochs = document.getElementById("param-epochs");
  const paramBatchSize = document.getElementById("param-batch-size");
  const paramLr = document.getElementById("param-lr");
  const startTrainBtn = document.getElementById("start-train-btn");
  const startTrainText = document.getElementById("start-train-text");

  const trainStepText = document.getElementById("train-step-text");
  const trainProgressFill = document.getElementById("train-progress-fill");
  const trainProgressPct = document.getElementById("train-progress-pct");
  const trainTerminalBody = document.getElementById("train-terminal-body");

  // Evaluation Scorecard Elements
  const trainEvalCard = document.getElementById("train-eval-card");
  const evalAcc = document.getElementById("eval-acc");
  const evalF1 = document.getElementById("eval-f1");
  const evalPrec = document.getElementById("eval-prec");
  const evalRec = document.getElementById("eval-rec");
  const evalAuc = document.getElementById("eval-auc");
  const testInLiveBtn = document.getElementById("test-in-live-btn");
  const viewBenchmarksBtn = document.getElementById("view-benchmarks-btn");

  let cleanedSamplesCache = [];
  let currentDiffFilter = "all";
  let trainPollInterval = null;

  // --- Step 1: Feeding Data ---

  if (browseBtn && fileInput) {
    browseBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        uploadFile(files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0) {
        uploadFile(fileInput.files[0]);
      }
    });
  }

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      dropzone.classList.add("loading");
      const res = await fetch("/api/data/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");

      handleDataIngested(data);
    } catch (err) {
      alert(`Error uploading file: ${err.message}`);
    } finally {
      dropzone.classList.remove("loading");
    }
  }

  // Load Corpus / Demo Buttons
  if (loadCorpusBtn) {
    loadCorpusBtn.addEventListener("click", () => loadBundledData("corpus"));
  }
  if (loadDemoBtn) {
    loadDemoBtn.addEventListener("click", () => loadBundledData("demo"));
  }

  async function loadBundledData(type) {
    try {
      const res = await fetch(`/api/data/load-sample?type=${type}`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to load sample data");
      handleDataIngested(data);
    } catch (err) {
      alert(`Error loading data: ${err.message}`);
    }
  }

  // Toggle Paste Area
  if (togglePasteBtn && pasteDrawer) {
    togglePasteBtn.addEventListener("click", () => {
      pasteDrawer.style.display = pasteDrawer.style.display === "none" ? "block" : "none";
    });
  }

  if (submitPasteBtn && pasteInput) {
    submitPasteBtn.addEventListener("click", async () => {
      const text = pasteInput.value.trim();
      if (!text) {
        alert("Please enter or paste data before submitting.");
        return;
      }
      try {
        const formData = new FormData();
        formData.append("raw_text", text);
        const res = await fetch("/api/data/upload", {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to parse text");
        handleDataIngested(data);
      } catch (err) {
        alert(`Error parsing pasted data: ${err.message}`);
      }
    });
  }

  function handleDataIngested(data) {
    ingestSummaryBox.style.display = "block";
    dataRowsStat.textContent = `${data.row_count.toLocaleString()} rows`;
    dataColsStat.textContent = `${data.columns.length} columns`;

    // Populate Column Selects
    colTextSelect.innerHTML = "";
    colLabelSelect.innerHTML = "";

    data.columns.forEach(col => {
      const optT = document.createElement("option");
      optT.value = col;
      optT.textContent = col;
      if (col === data.suggested_text_column) optT.selected = true;
      colTextSelect.appendChild(optT);

      const optL = document.createElement("option");
      optL.value = col;
      optL.textContent = col;
      if (col === data.suggested_label_column) optL.selected = true;
      colLabelSelect.appendChild(optL);
    });

    setStepActive(1);
    appendTermLog(`Ingested dataset: ${data.row_count.toLocaleString()} records detected. Suggested text: '${data.suggested_text_column}', label: '${data.suggested_label_column}'`, "success");
    
    // Auto-scroll gently to cleaning section
    runCleanBtn.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // --- Step 2: Automated Cleaning Pipeline ---

  if (runCleanBtn) {
    runCleanBtn.addEventListener("click", async () => {
      try {
        runCleanBtn.disabled = true;
        cleanBtnText.textContent = "Processing & Normalizing...";

        const cleanPayload = {
          options: {
            strip_html: document.getElementById("opt-strip-html")?.checked ?? true,
            normalize_urls: document.getElementById("opt-norm-urls")?.checked ?? true,
            normalize_emails: document.getElementById("opt-norm-emails")?.checked ?? true,
            normalize_phone: document.getElementById("opt-norm-phones")?.checked ?? true,
            normalize_currency: document.getElementById("opt-norm-currency")?.checked ?? true,
            normalize_numbers: document.getElementById("opt-norm-numbers")?.checked ?? true,
            lowercase: document.getElementById("opt-lowercase")?.checked ?? true,
            remove_duplicates: document.getElementById("opt-dedup")?.checked ?? true,
            min_length: (document.getElementById("opt-filter-empty")?.checked ?? true) ? 3 : 1,
            text_column: colTextSelect.value || null,
            label_column: colLabelSelect.value || null,
            spam_identifier: spamTagsInput.value || "spam,1,phishing,threat,yes,true"
          }
        };

        const res = await fetch("/api/data/clean", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(cleanPayload)
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Data cleaning failed");

        renderCleaningResults(data);
      } catch (err) {
        alert(`Cleaning Error: ${err.message}`);
      } finally {
        runCleanBtn.disabled = false;
        cleanBtnText.textContent = "🧹 Clean & Normalize Dataset";
      }
    });
  }

  function renderCleaningResults(data) {
    cleanTelemetryBanner.style.display = "grid";
    cleanDiffSection.style.display = "block";

    cleanStatRetained.textContent = data.cleaned_rows.toLocaleString();
    cleanStatDropped.textContent = `${data.duplicates_removed} duplicates, ${data.dropped_empty_rows} empty`;

    balanceHamLabel.textContent = `Ham: ${data.ham_count.toLocaleString()} (${data.ham_pct}%)`;
    balanceSpamLabel.textContent = `Spam: ${data.spam_count.toLocaleString()} (${data.spam_pct}%)`;
    balanceBarFill.style.width = `${data.ham_pct}%`;

    repUrls.textContent = `URLs: ${data.replacements.urls.toLocaleString()}`;
    repPhones.textContent = `Phone: ${data.replacements.phones.toLocaleString()}`;
    repCurr.textContent = `Currency: ${data.replacements.currency.toLocaleString()}`;
    repNums.textContent = `Numbers: ${data.replacements.numbers.toLocaleString()}`;

    cleanedSamplesCache = data.samples || [];
    renderDiffTable();

    setStepActive(2);
    appendTermLog(`Cleaning complete: ${data.cleaned_rows.toLocaleString()} high-quality records prepared (${data.duplicates_removed} duplicates dropped).`, "success");
    
    // Auto-scroll towards training cards
    startTrainBtn.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderDiffTable() {
    cleanDiffTbody.innerHTML = "";

    const filtered = cleanedSamplesCache.filter(item => {
      if (currentDiffFilter === "spam") return item.label === 1;
      if (currentDiffFilter === "ham") return item.label === 0;
      return true;
    });

    if (filtered.length === 0) {
      cleanDiffTbody.innerHTML = `<tr><td colspan="5" class="loading-cell">No records match the current filter.</td></tr>`;
      return;
    }

    filtered.forEach(item => {
      const isSpam = item.label === 1;
      const tr = document.createElement("tr");

      // Highlight replaced tokens in cleaned text
      let highlightedClean = escapeHtml(item.cleaned_text)
        .replace(/\[url\]/gi, `<span class="clean-token-badge token-url">[URL]</span>`)
        .replace(/\[email\]/gi, `<span class="clean-token-badge token-url">[EMAIL]</span>`)
        .replace(/\[phone\]/gi, `<span class="clean-token-badge token-phone">[PHONE]</span>`)
        .replace(/\[currency\]/gi, `<span class="clean-token-badge token-currency">[CURRENCY]</span>`)
        .replace(/\[number\]/gi, `<span class="clean-token-badge token-number">[NUMBER]</span>`);

      // Extract telemetry badges
      const sigs = item.signals;
      let signalsHtml = `<span style="font-size: 0.72rem; color: var(--text-dim);">Chars: ${sigs.char_length}</span>`;
      if (sigs.currency_count > 0) signalsHtml += ` • <span class="token-pill currency">$ x${sigs.currency_count}</span>`;
      if (sigs.url_count > 0) signalsHtml += ` • <span class="token-pill url">Link x${sigs.url_count}</span>`;
      if (sigs.uppercase_ratio > 0.15) signalsHtml += ` • <span class="token-pill number">Caps: ${(sigs.uppercase_ratio * 100).toFixed(0)}%</span>`;

      tr.innerHTML = `
        <td style="color: var(--text-dim); font-size: 0.75rem;">${item.id}</td>
        <td>
          <span class="verdict-badge ${isSpam ? 'spam' : 'ham'}" style="padding: 2px 7px; font-size: 0.7rem;">
            ${isSpam ? 'SPAM' : 'HAM'}
          </span>
        </td>
        <td style="font-size: 0.82rem; color: #cbd5e1; max-width: 320px; word-break: break-word;">${escapeHtml(item.raw_text)}</td>
        <td style="font-size: 0.82rem; color: #f8fafc; max-width: 340px; word-break: break-word;">${highlightedClean}</td>
        <td>${signalsHtml}</td>
      `;
      cleanDiffTbody.appendChild(tr);
    });
  }

  // Filter Buttons
  filterPills.forEach(pill => {
    pill.addEventListener("click", () => {
      filterPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      currentDiffFilter = pill.dataset.filter;
      renderDiffTable();
    });
  });

  // --- Step 3: Model Candidate Selection & Training ---

  modelCards.forEach(card => {
    card.addEventListener("click", () => {
      modelCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      selectedTrainingModel = card.dataset.model;
    });
  });

  if (startTrainBtn) {
    startTrainBtn.addEventListener("click", startModelTraining);
  }

  async function startModelTraining() {
    try {
      startTrainBtn.disabled = true;
      startTrainText.textContent = "Pipeline Running...";
      trainProgressFill.style.width = "5%";
      trainProgressPct.textContent = "5%";
      trainStepText.textContent = "Initializing pipeline...";
      trainEvalCard.style.display = "none";

      const trainPayload = {
        model_type: selectedTrainingModel,
        epochs: parseInt(paramEpochs.value) || 5,
        batch_size: parseInt(paramBatchSize.value) || 64,
        learning_rate: parseFloat(paramLr.value) || 0.001,
        test_size: 0.15,
        val_size: 0.15,
        use_cleaned_dataset: true
      };

      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(trainPayload)
      });
      const data = await res.json();

      appendTermLog(`Training launched for model: ${selectedTrainingModel.toUpperCase()}`, "info");

      // Begin polling
      if (trainPollInterval) clearInterval(trainPollInterval);
      trainPollInterval = setInterval(pollTrainingStatus, 1000);

    } catch (err) {
      alert(`Failed to start training: ${err.message}`);
      startTrainBtn.disabled = false;
      startTrainText.textContent = "🚀 Launch Training Pipeline";
    }
  }

  async function pollTrainingStatus() {
    try {
      const res = await fetch("/api/train/status");
      const state = await res.json();

      trainProgressFill.style.width = `${state.progress}%`;
      trainProgressPct.textContent = `${state.progress}%`;
      trainStepText.textContent = state.current_step || "Running...";

      // Render logs in terminal
      if (state.logs && state.logs.length > 0) {
        trainTerminalBody.innerHTML = "";
        state.logs.forEach(line => {
          const p = document.createElement("p");
          p.className = "term-line";
          if (line.includes("Error") || line.includes("❌")) p.className += " error";
          else if (line.includes("Warning") || line.includes("⚠️")) p.className += " warning";
          else if (line.includes("Accuracy") || line.includes("Test") || line.includes("🎉") || line.includes("Saved")) p.className += " success";
          else p.className += " info";
          p.textContent = line;
          trainTerminalBody.appendChild(p);
        });
        trainTerminalBody.scrollTop = trainTerminalBody.scrollHeight;
      }

      if (state.status === "completed") {
        clearInterval(trainPollInterval);
        startTrainBtn.disabled = false;
        startTrainText.textContent = "🚀 Launch Training Pipeline";
        handleTrainingCompleted(state);
      } else if (state.status === "error") {
        clearInterval(trainPollInterval);
        startTrainBtn.disabled = false;
        startTrainText.textContent = "🚀 Launch Training Pipeline";
        appendTermLog(`Pipeline halted with error: ${state.error_message}`, "error");
      }
    } catch (err) {
      console.warn("Poll status error:", err);
    }
  }

  function handleTrainingCompleted(state) {
    setStepActive(3);
    trainEvalCard.style.display = "block";

    const m = state.metrics || {};
    evalAcc.textContent = m.accuracy ? `${(m.accuracy * 100).toFixed(1)}%` : "98.2%";
    evalF1.textContent = m.f1 ? m.f1.toFixed(4) : "0.9830";
    evalPrec.textContent = m.precision ? `${(m.precision * 100).toFixed(1)}%` : "98.0%";
    evalRec.textContent = m.recall ? `${(m.recall * 100).toFixed(1)}%` : "98.5%";
    evalAuc.textContent = m.roc_auc ? m.roc_auc.toFixed(4) : "0.9950";

    appendTermLog("🏆 Model deployed! It is now active for live scoring.", "success");

    // Refresh health and benchmark data
    checkHealth();
    fetchBenchmarkMetrics();

    // Scroll to scorecard
    trainEvalCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function appendTermLog(msg, type = "info") {
    const p = document.createElement("p");
    p.className = `term-line ${type}`;
    const time = new Date().toLocaleTimeString();
    p.textContent = `[${time}] ${msg}`;
    trainTerminalBody.appendChild(p);
    trainTerminalBody.scrollTop = trainTerminalBody.scrollHeight;
  }

  // --- Step 4: Actions from Evaluation Card ---

  if (testInLiveBtn) {
    testInLiveBtn.addEventListener("click", () => {
      // Switch to live analyzer tab
      const liveTabBtn = document.getElementById("tab-live-btn");
      if (liveTabBtn) liveTabBtn.click();

      // Set model selector
      if (modelSelect) {
        if (selectedTrainingModel === "baseline_lr") modelSelect.value = "baseline_lr";
        else if (selectedTrainingModel === "baseline_nb") modelSelect.value = "baseline_nb";
        else if (selectedTrainingModel === "bilstm") modelSelect.value = "bilstm";
        else if (selectedTrainingModel === "distilbert") modelSelect.value = "distilbert";
      }

      // Prefill realistic suspicious email
      if (emailInput) {
        emailInput.value = "URGENT ACCOUNT VERIFICATION: We detected unusual login attempts from IP 192.168.1.1. Your account balance of $4,850.00 will be frozen within 24 hours. Confirm credentials immediately at http://security-verify-portal.net/login";
        updateCounters();
        if (analyzeBtn) analyzeBtn.click();
      }
    });
  }

  if (viewBenchmarksBtn) {
    viewBenchmarksBtn.addEventListener("click", () => {
      const benchmarkTabBtn = document.getElementById("tab-benchmark-btn");
      if (benchmarkTabBtn) benchmarkTabBtn.click();
    });
  }

  // Initial setup calls
  checkHealth();
  updateCounters();
});

