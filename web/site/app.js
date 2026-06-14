/* COVID CT demo: sample picker, overlay toggle, rough upload estimate. */
(function () {
  var state = { samples: [], current: 0, mode: "ct" };
  var mainimg = document.getElementById("mainimg");

  function fmt(n) { return n.toLocaleString("en-US"); }

  fetch("samples/results.json").then(function (r) { return r.json(); }).then(function (data) {
    state.samples = data.samples;
    buildThumbs();
    select(0);
  });

  function buildThumbs() {
    var box = document.getElementById("thumbs");
    state.samples.forEach(function (s, i) {
      var b = document.createElement("button");
      b.className = "thumb";
      b.innerHTML = '<img src="samples/sample' + s.id + '.png" alt="sample ' + s.id + '">' +
                    "<span>S" + s.id + ": " + s.infection_pct + "% infected<br>" + s.label + "</span>";
      b.addEventListener("click", function () { select(i); });
      box.appendChild(b);
    });
  }

  function select(i) {
    state.current = i;
    var s = state.samples[i];
    var thumbs = document.querySelectorAll(".thumb");
    thumbs.forEach(function (t, j) { t.classList.toggle("active", j === i); });
    draw();
    document.getElementById("r-title").textContent = "Sample S" + s.id + " (" + s.label + ")";
    document.getElementById("r-pct").textContent = s.infection_pct.toFixed(2);
    document.getElementById("r-ci").textContent = "[" + s.ci_low.toFixed(2) + ", " + s.ci_high.toFixed(2) + "]";
    document.getElementById("r-slice").textContent = s.slice;
    document.getElementById("r-lung").textContent = fmt(s.lung_px);
    document.getElementById("r-inf").textContent = fmt(s.infected_px);
    var lo = s.ci_low, hi = s.ci_high, p = s.infection_pct;
    var span = Math.max(hi * 1.15, 1);
    var fill = document.getElementById("r-cifill");
    fill.style.left = (lo / span * 100) + "%";
    fill.style.width = ((hi - lo) / span * 100) + "%";
    document.getElementById("r-cidot").style.left = "calc(" + (p / span * 100) + "% - 1px)";
  }

  function draw() {
    var s = state.samples[state.current];
    mainimg.src = state.mode === "overlay"
      ? "samples/sample" + s.id + "_overlay.png"
      : "samples/sample" + s.id + ".png";
  }

  document.getElementById("btn-ct").addEventListener("click", function () {
    state.mode = "ct"; setToggle(); draw();
  });
  document.getElementById("btn-overlay").addEventListener("click", function () {
    state.mode = "overlay"; setToggle(); draw();
  });
  function setToggle() {
    document.getElementById("btn-ct").classList.toggle("active", state.mode === "ct");
    document.getElementById("btn-overlay").classList.toggle("active", state.mode === "overlay");
  }

  /* Upload: rough on-screen estimate only. Invert the standard lung window
     (WL -600, WW 1500) back to approximate HU, then count pixels in the
     infection band over pixels in a broad lung-ish band. Labeled approximate. */
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
    ctx.fillStyle = "#666";
    ctx.fillText("Infection percentage", padL, H - 24 + 14);
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
