/* Analyze your own scan: runs the real pipeline in the cloud.
 *
 * Set CLOUD_API to the deployed Cloud Run service URL (no trailing slash),
 * e.g. "https://ctpipe-cloud-abc123-uc.a.run.app".
 *
 * Behavior:
 * - CLOUD_API empty: explains cloud analysis is not connected yet.
 * - GET /status says uploads_enabled: shows the live upload form.
 * - Otherwise: shows the saved example cloud run
 *   (samples/cloud_example/results.json). Uploads are switched on/off with
 *   the UPLOADS_ENABLED env var on the Cloud Run service; the switch lives
 *   in the GCP console and needs no code change.
 */
var CLOUD_API = "";

(function () {
  "use strict";
  var body = document.getElementById("ownscan-body");
  if (!body) return;

  function fmt(n) { return n.toLocaleString("en-US"); }

  function renderResults(box, d) {
    box.hidden = false;
    var html = "";
    var series = d.series || [];
    if (!series.length) {
      html += "<h3>No eligible series found</h3><p>The pipeline ran, but no series " +
        "passed the eligibility rule. Check the validation details below.</p>";
    } else {
      var s = series[0];
      html += '<div class="big"><span>' + s.infection_pct.toFixed(2) +
        '</span><span class="unit">% of lung infected</span></div>' +
        '<div class="ci">95% CI [' + s.ci_low.toFixed(2) + ", " + s.ci_high.toFixed(2) + "]</div>" +
        '<table class="datatable">' +
        "<tr><th>Field</th><th>Value</th></tr>" +
        "<tr><td>Slices</td><td class='num'>" + s.slices + "</td></tr>" +
        "<tr><td>Lung voxels</td><td class='num'>" + fmt(s.lung_voxels) + "</td></tr>" +
        "<tr><td>Infected voxels</td><td class='num'>" + fmt(s.infected_voxels) + "</td></tr>" +
        "</table>";
      if (series.length > 1) {
        html += '<p class="note">' + series.length + " eligible series found; " +
          "showing the first. Full per-series table is in the downloadable results.</p>";
      }
    }
    var v = d.validation || {};
    html += "<h3>Validation</h3>" +
      '<div class="statrow">' +
      '<div class="stat"><span class="statnum">' + (v.n_series || 0) + '</span><span class="statlbl">series checked</span></div>' +
      '<div class="stat"><span class="statnum">' + (v.n_eligible || 0) + '</span><span class="statlbl">eligible</span></div>' +
      '<div class="stat"><span class="statnum">' + (v.n_quarantined || 0) + '</span><span class="statlbl">quarantined</span></div>' +
      "</div>";
    if ((v.quarantined || []).length) {
      html += '<table class="datatable"><tr><th>Quarantined series</th><th>Reason</th></tr>';
      v.quarantined.forEach(function (q) {
        html += "<tr><td>" + String(q.series_description || q.series_uid).replace(/</g, "&lt;") +
          "</td><td>" + String(q.reason).replace(/</g, "&lt;") + "</td></tr>";
      });
      html += "</table>";
    }
    var urls = d.overlay_urls || {};
    var names = Object.keys(urls);
    if (names.length) {
      html += "<h3>Infection overlays</h3><div class='overlays'>";
      names.forEach(function (n) {
        html += "<figure><img src='" + urls[n] + "' alt='Infection overlay'>" +
          "<figcaption>" + n.replace(/</g, "&lt;") + "</figcaption></figure>";
      });
      html += "</div>";
    }
    var p = d.pipeline || {};
    html += "<p class='fine'>Analyzed " + (d.created_at || "") +
      " &middot; code hash <code>" + (p.code_hash || "?") +
      "</code> &middot; git <code>" + (p.git_sha || "?") + "</code>.</p>";
    box.innerHTML = html;
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function showNotConnected() {
    body.innerHTML =
      '<p class="note">Cloud analysis isn\'t connected yet. The pipeline itself ' +
      'runs in the repo (<code>pipeline/</code>); the cloud runner that powers ' +
      'this section lives in <code>cloud/</code> and is deployed separately.</p>';
  }

  function showExample() {
    fetch("samples/cloud_example/results.json").then(function (r) {
      if (!r.ok) throw new Error("no saved example");
      return r.json();
    }).then(function (d) {
      body.innerHTML =
        '<p class="note">Live uploads are currently switched off. Below is a real ' +
        'analysis of the example RICORD series, run end to end through this same ' +
        'cloud pipeline and saved.</p><div id="cloud-result"></div>';
      renderResults(document.getElementById("cloud-result"), d);
    }).catch(function () {
      body.innerHTML =
        '<p class="note">Cloud analysis is not available right now.</p>';
    });
  }

  function initUploadForm() {
    var fileInput = document.getElementById("cloud-file");
    var startBtn = document.getElementById("cloud-start");
    var progBox = document.getElementById("cloud-progress");
    var progFill = document.getElementById("cloud-fill");
    var progStatus = document.getElementById("cloud-status");
    var resultBox = document.getElementById("cloud-result");

    function setStatus(t) { progStatus.textContent = t; }
    function setBar(pct) { progFill.style.width = Math.max(0, Math.min(100, pct)) + "%"; }

    function fail(msg) {
      setStatus("Failed: " + msg);
      startBtn.disabled = false;
    }

    function poll(jobId, onDone) {
      var timer = setInterval(function () {
        fetch(CLOUD_API + "/jobs/" + jobId).then(function (r) { return r.json(); })
          .then(function (d) {
            if (d.status === "done") { clearInterval(timer); onDone(null, d); }
            else if (d.status === "failed") { clearInterval(timer); onDone(d.error || "analysis failed"); }
          }).catch(function () { /* keep polling */ });
      }, 10000);
      return timer;
    }

    function uploadWithProgress(url, file, onProgress) {
      return new Promise(function (resolve, reject) {
        var xhr = new XMLHttpRequest();
        xhr.open("PUT", url, true);
        xhr.setRequestHeader("Content-Type", "application/zip");
        xhr.upload.addEventListener("progress", function (ev) {
          if (ev.lengthComputable) onProgress(100 * ev.loaded / ev.total);
        });
        xhr.onload = function () {
          if (xhr.status >= 200 && xhr.status < 300) resolve();
          else reject(new Error("upload returned HTTP " + xhr.status));
        };
        xhr.onerror = function () { reject(new Error("upload failed (network error)")); };
        xhr.send(file);
      });
    }

    startBtn.addEventListener("click", function () {
      var f = fileInput.files[0];
      if (!f) { setStatus("Choose a .zip file first."); return; }
      if (!/\.zip$/i.test(f.name)) { setStatus("Please choose a .zip file of DICOM slices."); return; }
      if (f.size > 200 * 1024 * 1024) { setStatus("That file is over the 200 MB cap."); return; }
      startBtn.disabled = true;
      resultBox.hidden = true;
      progBox.hidden = false;
      setBar(0);

      var finished = false;
      function done(err, data) {
        if (finished) return;
        finished = true;
        startBtn.disabled = false;
        if (err) fail(err);
        else renderResults(resultBox, data);
      }

      setStatus("Getting an upload URL\u2026");
      fetch(CLOUD_API + "/upload-url?filename=" + encodeURIComponent(f.name))
        .then(function (r) {
          if (r.status === 503) throw new Error("uploads are currently switched off");
          if (!r.ok) throw new Error("could not get an upload URL (HTTP " + r.status + ")");
          return r.json();
        })
        .then(function (u) {
          setStatus("Uploading " + f.name + "\u2026");
          return uploadWithProgress(u.upload_url, f, setBar).then(function () { return u; });
        })
        .then(function (u) {
          setStatus("Upload done. Running the pipeline \u2014 this takes a few minutes for a full series\u2026");
          setBar(100);
          var timer = poll(u.job_id, done);
          fetch(CLOUD_API + "/jobs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ gcs_uri: u.gcs_uri })
          }).then(function (r) {
            if (!r.ok) return r.json().then(function (e) {
              throw new Error((e && e.detail) || ("HTTP " + r.status));
            });
            return r.json();
          }).then(function (d) {
            clearInterval(timer);
            done(null, d);
          }).catch(function (e) {
            /* The POST connection may drop on slow networks; the poller keeps
               watching for the finished results. */
            setStatus("Connection hiccup \u2014 still watching for your results\u2026 (" + e.message + ")");
          });
        })
        .catch(function (e) { done(e.message); });
    });
  }

  if (!CLOUD_API) { showNotConnected(); return; }
  fetch(CLOUD_API + "/status").then(function (r) { return r.json(); })
    .then(function (s) {
      if (s && s.uploads_enabled) initUploadForm();
      else showExample();
    })
    .catch(function () { showExample(); });
})();
