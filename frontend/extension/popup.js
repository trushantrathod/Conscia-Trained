const btn = document.getElementById("analyzeBtn");

function getColor(score) {
  if (score > 70) return "#22c55e";
  if (score > 40) return "#facc15";
  return "#ef4444";
}

if (btn) {
  btn.addEventListener("click", async () => {
    const input = document.getElementById("productInput").value.toLowerCase();

    const box = document.getElementById("resultBox");
    const name = document.getElementById("productName");
    const scoreText = document.getElementById("scoreValue");
    const bar = document.getElementById("scoreBar");
    const summary = document.getElementById("summary");
    const recommendationEl = document.getElementById("recommendation");

    if (!input) return;

    box.classList.remove("hidden");
    summary.innerText = "Loading...";

    try {
      // ==============================
      // 🥇 GET RAW PRODUCT (WEBSITE SCORE)
      // ==============================
      const res1 = await fetch("http://127.0.0.1:5000/api/products");
      const products = await res1.json();

      const product = products.find(p =>
        p.product_name.toLowerCase().includes(input)
      );

      if (!product) {
        summary.innerText = "Product not found";
        return;
      }

      // 🎯 USE WEBSITE SCORE ONLY
      const score = Math.round(product.public_sentiment_score || 0);

      name.innerText = product.product_name;
      scoreText.innerText = score + "/100";
      bar.style.width = score + "%";
      bar.style.background = getColor(score);

      // ==============================
      // 🥈 GET AI SUMMARY ONLY
      // ==============================
      const res2 = await fetch("http://127.0.0.1:5000/api/product-analysis", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          product: product.product_name
        })
      });

      const data = await res2.json();

      // 🤖 Gemini summary
      summary.innerText = data.summary;

      // 🛒 Recommendation
      recommendationEl.innerText = data.recommendation;

      // 💸 Better options
      if (data.better_options && data.better_options.length > 0) {
        summary.innerText += "\n\nBetter options:\n" +
          data.better_options
            .map(p => `- ${p.product_name} (${Math.round(p.public_sentiment_score)})`)
            .join("\n");
      }

    } catch (err) {
      console.error(err);
      summary.innerText = "⚠️ Error connecting to backend";
    }
  });
}