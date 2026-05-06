const API_URL = 'http://127.0.0.1:5000/api';

const ETHICAL_DIMS = [
  { key: 'environmental_impact', label: 'Environment', color: '#10b981' },
  { key: 'labor_rights',         label: 'Labor',       color: '#3b82f6' },
  { key: 'animal_welfare',       label: 'Animals',     color: '#f59e0b' },
  { key: 'corporate_governance', label: 'Governance',  color: '#8b5cf6' },
];

function getColor(score) {
  if (score >= 70) return '#10b981';
  if (score >= 40) return '#f59e0b';
  return '#ef4444';
}

// Determines the recommendation badge text and styles
function getRecommendation(sentiment, avgEthical) {
  if (sentiment >= 65 || avgEthical >= 60) {
    return { text: '✨ Should Buy', color: '#10b981', bg: 'rgba(16, 185, 129, 0.15)' };
  }
  if (sentiment <= 40 || avgEthical <= 40) {
    return { text: '🚨 Don\'t Buy', color: '#ef4444', bg: 'rgba(239, 68, 68, 0.15)' };
  }
  return { text: '🤔 Consider', color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.15)' };
}

document.addEventListener('DOMContentLoaded', () => {
  // ── Elements ──
  const statusIndicator = document.getElementById('statusIndicator');
  
  // Tabs
  const tabBtns = document.querySelectorAll('.tab-btn');
  const views = document.querySelectorAll('.view');
  
  // Insight UI
  const searchInput = document.getElementById('searchInput');
  const searchBtn = document.getElementById('searchBtn');
  const initialState = document.getElementById('initialState');
  const loadingState = document.getElementById('loadingState');
  const errorState = document.getElementById('errorState');
  const productData = document.getElementById('productData');
  const pReco = document.getElementById('pReco');
  const altList = document.getElementById('altList');
  
  // Chat UI
  const chatMessages = document.getElementById('chatMessages');
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const chatSendBtn = document.getElementById('chatSendBtn');

  // ── Connection Test ──
  fetch(`${API_URL}/products?limit=1`)
    .then(r => r.ok ? statusIndicator.style.background = '#10b981' : null)
    .catch(() => statusIndicator.style.background = '#ef4444');

  // ── Tab Switching ──
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      views.forEach(v => v.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.target).classList.add('active');
    });
  });

  // ── Search Product Flow ──
  async function handleSearch(searchTerm) {
    const query = (searchTerm || searchInput.value).trim().toLowerCase();
    if (!query) return;

    searchInput.value = query; // update input if triggered by alternative click

    // UI Updates
    initialState.classList.add('hidden');
    errorState.classList.add('hidden');
    productData.classList.add('hidden');
    loadingState.classList.remove('hidden');

    try {
      // 1. Fetch enough products to find a match and alternatives
      const res = await fetch(`${API_URL}/products?limit=1000`);
      if (!res.ok) throw new Error('Server error');
      
      const products = await res.json();
      const match = products.find(p => p.product_name.toLowerCase().includes(query));

      if (!match) {
        loadingState.classList.add('hidden');
        errorState.classList.remove('hidden');
        return;
      }

      // 2. Populate basic UI
      document.getElementById('pName').innerText = match.product_name;
      document.getElementById('pCategory').innerText = match.category;
      document.getElementById('pPrice').innerText = `₹${parseFloat(match.product_price).toFixed(0)}`;
      
      const sentScore = match.public_sentiment_score || 50;
      const sentEl = document.getElementById('pSentiment');
      sentEl.innerText = Math.round(sentScore);
      sentEl.style.color = getColor(sentScore);

      // 3. Populate Ethical Bars & Calculate Average
      const ethicalList = document.getElementById('ethicalList');
      ethicalList.innerHTML = '';
      let totalEthical = 0;
      
      ETHICAL_DIMS.forEach(dim => {
        const score = Math.round(match[dim.key] || 50);
        totalEthical += score;
        ethicalList.innerHTML += `
          <div class="ethical-row">
            <div class="e-label">${dim.label}</div>
            <div class="e-bar-bg">
              <div class="e-bar-fill" style="width: ${score}%; background: ${dim.color};"></div>
            </div>
            <div class="e-num" style="color: ${score > 60 ? dim.color : '#cbd5e1'}">${score}</div>
          </div>
        `;
      });

      const avgEthical = totalEthical / ETHICAL_DIMS.length;

      // 4. Set Recommendation Badge
      const reco = getRecommendation(sentScore, avgEthical);
      pReco.innerText = reco.text;
      pReco.style.color = reco.color;
      pReco.style.background = reco.bg;

      // 5. Find Better Alternatives
      const alternatives = products
        .filter(p => p.category === match.category && p.product_id !== match.product_id)
        .filter(p => p.public_sentiment_score > sentScore) // Strictly better sentiment
        .sort((a, b) => b.public_sentiment_score - a.public_sentiment_score)
        .slice(0, 3); // Top 3

      altList.innerHTML = '';
      if (alternatives.length > 0) {
        alternatives.forEach(alt => {
          const score = Math.round(alt.public_sentiment_score || 50);
          const card = document.createElement('div');
          card.className = 'alt-card';
          card.innerHTML = `
            <span class="alt-name">${alt.product_name}</span>
            <span class="alt-score" style="color: ${getColor(score)}">${score}</span>
          `;
          // Clicking an alternative searches it
          card.addEventListener('click', () => {
            handleSearch(alt.product_name);
            productData.scrollIntoView({ behavior: 'smooth' });
          });
          altList.appendChild(card);
        });
      } else {
        altList.innerHTML = `<span class="no-alts">No better alternatives found in this category. You've picked a top tier product!</span>`;
      }

      // 6. Fetch AI Snapshot
      const snapRes = await fetch(`${API_URL}/snapshot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product: match.product_name })
      });
      const snapData = await snapRes.json();
      document.getElementById('pSnapshot').innerText = snapData.snapshot || "No summary available.";

      // 7. Show Data
      loadingState.classList.add('hidden');
      productData.classList.remove('hidden');

    } catch (err) {
      console.error(err);
      loadingState.classList.add('hidden');
      document.getElementById('errorMsg').innerText = "Connection failed.";
      errorState.classList.remove('hidden');
    }
  }

  searchBtn.addEventListener('click', () => handleSearch());
  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSearch();
  });

  // ── Chat Flow ──
  chatInput.addEventListener('input', () => {
    chatSendBtn.disabled = chatInput.value.trim().length === 0;
  });

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    // Add User Message
    chatMessages.innerHTML += `<div class="chat-msg user">${text}</div>`;
    chatInput.value = '';
    chatSendBtn.disabled = true;
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Add Loading Indicator
    const loaderId = 'loader-' + Date.now();
    chatMessages.innerHTML += `<div id="${loaderId}" class="chat-msg bot"><div class="spinner" style="width:14px;height:14px;border-width:2px;"></div></div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      const data = await res.json();
      
      // Replace Loader with Bot Message
      document.getElementById(loaderId).remove();
      chatMessages.innerHTML += `<div class="chat-msg bot">${data.reply}</div>`;
    } catch (err) {
      document.getElementById(loaderId).remove();
      chatMessages.innerHTML += `<div class="chat-msg bot">Error connecting to the AI brain.</div>`;
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;
  });
});