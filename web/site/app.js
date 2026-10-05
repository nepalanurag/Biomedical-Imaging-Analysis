/* COVID CT demo: 255-slice viewer, pipeline stepper, sweep explorer,
   rough upload estimate, across-patients histogram. */
(function () {
  "use strict";
  function fmt(n) { return n.toLocaleString("en-US"); }
  function pad3(n) { return ("00" + n).slice(-3); }

  /* Wilson score interval, same formula as pipeline/quantify.py. */
  function wilson(k, n, z) {
    z = z || 1.96;
    if (!n) return [0, 1];
    var p = k / n, den = 1 + z * z / n;
    var c = p + z * z / (2 * n);
    var d = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    return [(c - d) / den, (c + d) / den];
  }

  /* ---------- Slice viewer ---------- */
  var NSLICE = 255;
  var vstate = { slice: 127, overlay: false, data: null, loaded: {} };
  var vct = document.getElementById("v-ct");
  var vov = document.getElementById("v-ov");
  var vslider = document.getElementById("v-slider");

  function sliceURL(kind, i) {
    return "samples/slices/" + kind + "_" + pad3(i) + ".png";
  }

  function ensureLoaded(i) {
    if (i < 0 || i >= NSLICE || vstate.loaded[i]) return;
    vstate.loaded[i] = true;
    var a = new Image(); a.src = sliceURL("ct", i);
    var b = new Image(); b.src = sliceURL("ov", i);
  }

  function showSlice(i) {
    i = Math.max(0, Math.min(NSLICE - 1, i));
    vstate.slice = i;
    vct.src = sliceURL("ct", i);
    vov.src = sliceURL("ov", i);
    vov.style.display = vstate.overlay ? "block" : "none";
    ensureLoaded(i - 2); ensureLoaded(i - 1);
    ensureLoaded(i + 1); ensureLoaded(i + 2);
    vslider.value = i;
    renderSliceStats(i);
  }

  function renderSliceStats(i) {
    var s = vstate.data ? vstate.data.slices[i] : null;
    document.getElementById("v-title").textContent = "Slice " + (i + 1) + " of " + NSLICE;
    document.getElementById("v-slice").textContent = i;
    if (!s) return;
    var pct = s.infection_pct;
    document.getElementById("v-pct").textContent = pct.toFixed(2);
    document.getElementById("v-lung").textContent = fmt(s.lung_voxels);
    document.getElementById("v-inf").textContent = fmt(s.infected_voxels);
    document.getElementById("v-hu").textContent =
      s.hu_min.toFixed(0) + " / " + s.hu_mean.toFixed(0) + " / " + s.hu_max.toFixed(0);
    var lo, hi;
    if (s.lung_voxels > 0) {
      var ci = wilson(s.infected_voxels, s.lung_voxels);
      lo = 100 * ci[0]; hi = 100 * ci[1];
      document.getElementById("v-ci").textContent =
        "[" + lo.toFixed(2) + ", " + hi.toFixed(2) + "]";
    } else {
      lo = 0; hi = 0;
      document.getElementById("v-ci").textContent = "n/a (no lung voxels)";
    }
    var span = Math.max(hi * 1.15, 1);
    var fill = document.getElementById("v-cifill");
    fill.style.left = (lo / span * 100) + "%";
    fill.style.width = ((hi - lo) / span * 100) + "%";
    document.getElementById("v-cidot").style.left = "calc(" + (pct / span * 100) + "% - 1px)";
  }

  function setOverlay(on) {
    vstate.overlay = on;
    vov.style.display = on ? "block" : "none";
    document.getElementById("vbtn-ct").classList.toggle("active", !on);
    document.getElementById("vbtn-overlay").classList.toggle("active", on);
  }

  document.getElementById("vbtn-ct").addEventListener("click", function () { setOverlay(false); });
  document.getElementById("vbtn-overlay").addEventListener("click", function () { setOverlay(true); });
  document.getElementById("v-prev").addEventListener("click", function () { showSlice(vstate.slice - 1); });
  document.getElementById("v-next").addEventListener("click", function () { showSlice(vstate.slice + 1); });
  vslider.addEventListener("input", function () { showSlice(parseInt(vslider.value, 10)); });
  document.addEventListener("keydown", function (ev) {
    if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;
    if (ev.key === "ArrowLeft") showSlice(vstate.slice - 1);
    else if (ev.key === "ArrowRight") showSlice(vstate.slice + 1);
  });

  fetch("samples/slices/slices.json").then(function (r) { return r.json(); }).then(function (d) {
    vstate.data = d;
    showSlice(vstate.slice);
  }).catch(function () {
    document.getElementById("v-title").textContent = "Slice data could not be loaded";
  });

  /* ---------- Pipeline stepper ---------- */
  var stepbtns = document.querySelectorAll("#stepper .stepbtn");
  var panels = document.querySelectorAll(".stagepanel");
  stepbtns.forEach(function (b) {
    b.addEventListener("click", function () {
      var s = b.getAttribute("data-stage");
      stepbtns.forEach(function (x) { x.classList.toggle("active", x === b); });
      panels.forEach(function (p) {
        p.classList.toggle("active", p.getAttribute("data-stage") === s);
      });
    });
  });

  /* ---------- Sweep explorer ---------- */
  function bandStr(b) { return "[" + b[0] + ", " + b[1] + "] HU"; }

  fetch("samples/sweep.json").then(function (r) { return r.json(); }).then(function (cfgs) {
    drawSweep(cfgs);
    selectSweep(cfgs[0], cfgs);
  }).catch(function () {});

  function drawSweep(cfgs) {
    var svg = document.getElementById("sweep-chart");
    var W = 720, H = 300, padL = 46, padR = 14, padT = 14, padB = 96;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", "100%");
    var NS = "http://www.w3.org/2000/svg";
    var maxV = 0;
    cfgs.forEach(function (c) { maxV = Math.max(maxV, c.mean_pct); });
    maxV = Math.ceil(maxV / 5) * 5;
    var bw = (W - padL - padR) / cfgs.length;

    function el(tag, attrs) {
      var e = document.createElementNS(NS, tag);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      return e;
    }

    var g, yy, i;
    for (g = 0; g <= 3; g++) {
      var v = maxV * g / 3;
      yy = H - padB - (v / maxV) * (H - padT - padB);
      svg.appendChild(el("line", { x1: padL, y1: yy, x2: W - padR, y2: yy, stroke: "#e3e3e3" }));
      var t = el("text", { x: 8, y: yy + 4, "font-size": 11, fill: "#666" });
      t.textContent = v.toFixed(0) + "%";
      svg.appendChild(t);
    }
    /* baseline line at 9.69 */
    var by = H - padB - (9.69 / maxV) * (H - padT - padB);
    svg.appendChild(el("line", { x1: padL, y1: by, x2: W - padR, y2: by,
      stroke: "#1a4f8a", "stroke-dasharray": "5,4", "stroke-width": 1.2 }));
    var bt = el("text", { x: W - padR - 64, y: by - 6, "font-size": 11, fill: "#1a4f8a" });
    bt.textContent = "baseline 9.69%";
    svg.appendChild(bt);

    cfgs.forEach(function (c, idx) {
      var h = (c.mean_pct / maxV) * (H - padT - padB);
      var x = padL + idx * bw;
      var bar = el("rect", {
        x: x + 2, y: H - padB - h, width: bw - 4, height: h,
        fill: c.id === "baseline" ? "#1a4f8a" : (c.id === "lung_lo_m1000" ? "#8a1f16" : "#b3352b"),
        opacity: 0.82, style: "cursor:pointer", "data-i": idx, "class": "sweepbar"
      });
      bar.addEventListener("click", function () { selectSweep(c, cfgs); });
      var title = document.createElementNS(NS, "title");
      title.textContent = c.label + ": " + c.mean_pct + "%";
      bar.appendChild(title);
      svg.appendChild(bar);
      var lab = el("text", {
        x: x + bw / 2, y: H - padB + 12, "font-size": 10, fill: "#666",
        "text-anchor": "middle", transform: "rotate(38 " + (x + bw / 2) + " " + (H - padB + 12) + ")"
      });
      lab.textContent = c.label.replace("infection band ", "").replace(" (standard bands)", "");
      svg.appendChild(lab);
    });
  }

  function selectSweep(c, cfgs) {
    document.getElementById("sd-title").textContent = c.label;
    var html = '<table class="datatable">' +
      "<tr><th>Field</th><th>Value</th></tr>" +
      "<tr><td>Lung band</td><td class='num'>" + bandStr(c.lung_band) + "</td></tr>" +
      "<tr><td>Infection band</td><td class='num'>" + bandStr(c.inf_band) + "</td></tr>" +
      "<tr><td>Varied vs baseline</td><td>" + c.varied + "</td></tr>" +
      "<tr><td>Mean infection</td><td class='num'>" + c.mean_pct.toFixed(2) + "%</td></tr>" +
      "<tr><td>Runtime</td><td class='num'>" + c.seconds.toFixed(1) + " s</td></tr>" +
      "</table>";
    if (c.note) html += '<p class="note">' + c.note + "</p>";
    document.getElementById("sd-body").innerHTML = html;
    var bars = document.querySelectorAll(".sweepbar");
    bars.forEach(function (b) {
      b.style.stroke = (cfgs[parseInt(b.getAttribute("data-i"), 10)].id === c.id) ? "#222" : "none";
      b.style.strokeWidth = "2";
    });
  }

  /* ---------- Upload: rough on-screen estimate only ---------- */
  document.getElementById("upload").addEventListener("change", function (ev) {
    var f = ev.target.files[0];
    if (!f) return;
    var url = URL.createObjectURL(f);
    var img = new Image();
    img.onload = function () {
      var c = document.createElement("canvas");
      c.width = img.width; c.height = img.height;
      var ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0);
      var d = ctx.getImageData(0, 0, c.width, c.height).data;
      var lung = 0, inf = 0;
      for (var i = 0; i < d.length; i += 4) {
        var g = (d[i] + d[i + 1] + d[i + 2]) / 3 / 255;
        var hu = g * 1500 - 1350; /* WL -600, WW 1500 */
        if (hu >= -950 && hu <= -300) lung++;
        if (hu >= -700 && hu <= -200) inf++;
      }
      var box = document.getElementById("upload-result");
      box.hidden = false;
      document.getElementById("upload-img").src = url;
      var txt = document.getElementById("upload-text");
      if (lung < 1000) {
        txt.textContent = "Could not find much lung-like tissue in this image, so no estimate. " +
          "Try a lung-window CT slice like the samples above.";
      } else {
        var p = 100 * inf / lung;
        txt.textContent = "Rough on-screen estimate: about " + p.toFixed(1) + "% of lung-like pixels " +
          "fall in the infection band. Approximate only, no lung mask was computed in the browser.";
      }
    };
    img.src = url;
  });
})();

/* Across patients: histogram of infection percentage from the study CSV.
   Rows with infection_percentage > 100 are dropped as outliers, as in the
   original analysis. */
(function () {
  var canvas = document.getElementById("hist");
  if (!canvas) return;

  function parseCSV(text) {
    var lines = text.trim().split(/\r?\n/);
    var head = lines[0].split(",");
    var ci = head.indexOf("infection_percentage");
    var si = head.indexOf("Subject ID");
    var vals = [], subs = {};
    for (var i = 1; i < lines.length; i++) {
      var p = lines[i].split(",");
      if (p.length <= Math.max(ci, si)) continue;
      var v = parseFloat(p[ci]);
      if (isNaN(v) || v > 100) continue;
      vals.push(v);
      subs[p[si]] = true;
    }
    return { vals: vals, nPatients: Object.keys(subs).length };
  }

  function stats(vals) {
    var s = vals.slice().sort(function (a, b) { return a - b; });
    var mean = s.reduce(function (a, b) { return a + b; }, 0) / s.length;
    var mid = s.length >> 1;
    var median = s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
    return { mean: mean, median: median };
  }

  function draw(vals) {
    var NB = 20, MAXV = 100;
    var bins = new Array(NB).fill(0);
    vals.forEach(function (v) {
      var b = Math.min(NB - 1, Math.floor(v / MAXV * NB));
      bins[b]++;
    });
    var ctx = canvas.getContext("2d");
    var W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    var padL = 44, padR = 12, padT = 12, padB = 32;
    var maxC = Math.max.apply(null, bins.concat([1]));
    var bw = (W - padL - padR) / NB;
    ctx.font = "11px Georgia, serif";
    var g, yy;
    for (g = 0; g <= 4; g++) {
      var c = (maxC * g) / 4;
      yy = H - padB - (c / maxC) * (H - padT - padB);
      ctx.strokeStyle = "#e3e3e3";
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W - padR, yy); ctx.stroke();
      ctx.fillStyle = "#666";
      ctx.fillText(String(Math.round(c)), 8, yy + 4);
    }
    ctx.fillStyle = "#666";
    for (g = 0; g <= 4; g++) {
      var xv = (MAXV * g) / 4;
      ctx.fillText(xv + "%", padL + (g / 4) * (W - padL - padR) - 8, H - 10);
    }
    ctx.fillStyle = "#b3352b";
    ctx.globalAlpha = 0.8;
    bins.forEach(function (c, i) {
      var h = (c / maxC) * (H - padT - padB);
      ctx.fillRect(padL + i * bw + 1, H - padB - h, bw - 2, h);
    });
    ctx.globalAlpha = 1;
  }

  fetch("infection_quantification_by_subject.csv").then(function (r) { return r.text(); })
    .then(function (t) {
      var d = parseCSV(t);
      var s = stats(d.vals);
      document.getElementById("st-n").textContent = d.vals.length;
      document.getElementById("st-pat").textContent = d.nPatients;
      document.getElementById("st-mean").textContent = s.mean.toFixed(1) + "%";
      document.getElementById("st-med").textContent = s.median.toFixed(1) + "%";
      draw(d.vals);
    })
    .catch(function () {
      canvas.getContext("2d").fillText("Could not load data", 20, 40);
    });
})();
