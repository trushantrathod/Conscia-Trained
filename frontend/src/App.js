import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css'

// ── Icons ─────────────────────────────────────────────────
const Spinner = ({ size = 24 }) => (
  <div className="spinner" style={{ width: size, height: size }} />
);

const IconSearch = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>;
const IconX = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>;
const IconLeaf = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM12 22V12"/></svg>;
const IconUsers = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>;
const IconHeart = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>;
const IconShield = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>;
const IconSparkle = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/></svg>;
const IconSun = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>;
const IconMoon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>;
const IconChat = () => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>;
const IconSend = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>;

// ── Constants ─────────────────────────────────────────────
const ETHICAL_DIMENSIONS = [
  { key: 'environmental_impact', label: 'Environmental Impact', icon: <IconLeaf/>,   color: '#10b981', desc: 'Evaluates carbon footprint, sustainable packaging, and eco-friendly manufacturing processes.' },
  { key: 'labor_rights',         label: 'Labor Rights',         icon: <IconUsers/>,  color: '#3b82f6', desc: 'Assesses fair wages, safe working conditions, and transparent supply chain practices.' },
  { key: 'animal_welfare',       label: 'Animal Welfare',       icon: <IconHeart/>,  color: '#f59e0b', desc: 'Monitors cruelty-free testing, responsible sourcing, and animal care standards.' },
  { key: 'corporate_governance', label: 'Corporate Governance', icon: <IconShield/>, color: '#8b5cf6', desc: 'Measures brand transparency, ethical leadership, and consumer accountability.' },
];

const AVAILABLE_CATEGORIES = ['All', 'Beauty', 'Electronics', 'Fashion', 'Groceries'];

// ── Category Meta (Added Icons for the New Design) ────────
const CATEGORY_META = {
  'Beauty':      { color: 'linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)', icon: '✨' },
  'Electronics': { color: 'linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)', icon: '📱' },
  'Fashion':     { color: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)', icon: '👗' },
  'Groceries':   { color: 'linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)', icon: '🛒' },
};

const PAGE_SIZE = 48;
const VISIBLE_STEP = 24;
const API_URL = 'http://localhost:5000/api';

// ── Helpers ───────────────────────────────────────────────
const getColor = (score) => {
  if (score >= 70) return '#10b981';
  if (score >= 40) return '#f59e0b';
  return '#ef4444';
};

const getAvgEthical = (p) => {
  const vals = ETHICAL_DIMENSIONS.map(d => p[d.key]).filter(v => v != null);
  return vals.length ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : null;
};

// Generates initials like "Beauty Miss" -> "BM"
const getInitials = (name) => {
  if (!name) return '??';
  const parts = name.split(' ').filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return parts[0].substring(0, 2).toUpperCase();
};

// ── Components ────────────────────────────────────────────

const AboutPage = () => (
  <div className="about-page">
    <div className="about-hero">
      <h1>Shop your values.</h1>
      <p>Conscia is an AI-powered ethical shopping assistant. We analyze thousands of real consumer reviews using natural language processing to score products across four critical ethical dimensions, empowering you to make informed, conscious purchases.</p>
    </div>
    <div className="pillars-grid">
      {ETHICAL_DIMENSIONS.map(dim => (
        <div key={dim.key} className="pillar-card">
          <div className="pillar-icon" style={{ color: dim.color }}>{dim.icon}</div>
          <h3>{dim.label}</h3>
          <p>{dim.desc}</p>
        </div>
      ))}
    </div>
  </div>
);


// ── Main App ──────────────────────────────────────────────
export default function App() {
  const [currentView, setCurrentView]       = useState('home');
  const [theme, setTheme]                   = useState('dark');

  const [products, setProducts]             = useState([]);
  const [filteredProducts, setFiltered]     = useState([]);
  const [visibleCount, setVisibleCount]     = useState(VISIBLE_STEP);
  const [offset, setOffset]                 = useState(0);
  const [hasMore, setHasMore]               = useState(true);
  const [isFetching, setIsFetching]         = useState(false);
  const [isInitialLoading, setInitLoading]  = useState(true);

  const [selectedCategory, setCategory]     = useState('All');
  const [searchTerm, setSearch]             = useState('');
  const [sortBy, setSortBy]                 = useState('default');

  const [selectedProduct, setSelectedProduct] = useState(null);
  const [explanation, setExplanation]         = useState(null);
  const [isExplaining, setIsExplaining]       = useState(false);
  const [summary, setSummary]                 = useState('');
  const [isSummaryLoading, setIsSummaryLoading] = useState(false);
  const [newReview, setNewReview]             = useState('');
  const [isReviewSubmitting, setIsReviewSub]  = useState(false);

  const [isChatOpen, setIsChatOpen]         = useState(false);
  const [chatMessages, setChatMessages]     = useState([{ sender: 'bot', text: 'Hi there! I am Conscia, your smart ethical shopping assistant. How can I help you today?' }]);
  const [chatInput, setChatInput]           = useState('');
  const [isChatting, setIsChatting]         = useState(false);
  const chatEndRef                          = useRef(null);

  const loaderRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  useEffect(() => {
    if (chatEndRef.current) chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, isChatOpen]);

  useEffect(() => {
    const loadCategory = async () => {
      setInitLoading(true);
      setIsFetching(true);
      const cacheKey = `conscia_products_${selectedCategory}`;
      const cached = sessionStorage.getItem(cacheKey);

      if (cached) {
        try {
          const parsed = JSON.parse(cached);
          if (parsed.length > 0) {
            setProducts(parsed);
            setOffset(parsed.length);
            setHasMore(parsed.length >= PAGE_SIZE);
            setInitLoading(false);
            setIsFetching(false);
            return;
          }
        } catch (e) {
          console.error('Cache parsing failed', e);
        }
      }

      try {
        const res = await fetch(`${API_URL}/products?limit=${PAGE_SIZE}&offset=0&category=${encodeURIComponent(selectedCategory)}`);
        const data = await res.json();
        setProducts(data);
        setOffset(data.length);
        setHasMore(data.length === PAGE_SIZE);
        sessionStorage.setItem(cacheKey, JSON.stringify(data.slice(0, 150)));
      } catch (e) {
        console.error('Failed to fetch products', e);
      } finally {
        setInitLoading(false);
        setIsFetching(false);
      }
    };

    loadCategory();
  }, [selectedCategory]);

  const fetchMore = useCallback(async () => {
    if (isFetching || !hasMore) return;
    setIsFetching(true);
    try {
      const res = await fetch(`${API_URL}/products?limit=${PAGE_SIZE}&offset=${offset}&category=${encodeURIComponent(selectedCategory)}`);
      const data = await res.json();
      if (data.length < PAGE_SIZE) setHasMore(false);
      setProducts(prev => {
        const newArr = [...prev, ...data];
        const cacheKey = `conscia_products_${selectedCategory}`;
        sessionStorage.setItem(cacheKey, JSON.stringify(newArr.slice(0, 150)));
        return newArr;
      });
      setOffset(prev => prev + data.length);
    } catch (e) {
      console.error(e);
    } finally {
      setIsFetching(false);
    }
  }, [isFetching, hasMore, offset, selectedCategory]);

  useEffect(() => {
    let r = products;
    if (searchTerm.trim()) {
      r = r.filter(p => p.product_name?.toLowerCase().includes(searchTerm.toLowerCase()));
    }

    if (sortBy === 'score') r = [...r].sort((a, b) => (b.public_sentiment_score || 0) - (a.public_sentiment_score || 0));
    else if (sortBy === 'ethical') r = [...r].sort((a, b) => (getAvgEthical(b) || 0) - (getAvgEthical(a) || 0));
    else if (sortBy === 'price_asc') r = [...r].sort((a, b) => parseFloat(a.product_price) - parseFloat(b.product_price));
    else if (sortBy === 'price_desc') r = [...r].sort((a, b) => parseFloat(b.product_price) - parseFloat(a.product_price));

    setFiltered(r);
    setVisibleCount(VISIBLE_STEP);
  }, [searchTerm, sortBy, products]);

  useEffect(() => {
    if (!loaderRef.current) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        if (visibleCount < filteredProducts.length) {
          setVisibleCount(c => c + VISIBLE_STEP);
        } else if (hasMore && !isFetching && currentView === 'home') {
          fetchMore();
        }
      }
    }, { rootMargin: '300px' });
    
    observer.observe(loaderRef.current);
    return () => observer.disconnect();
  }, [visibleCount, filteredProducts.length, hasMore, isFetching, fetchMore, currentView]);

  const handleProductClick = (product) => {
    setSelectedProduct(product);
    setExplanation(null);
    setSummary('');
    setNewReview('');
  };

  const handleReviewSubmit = async (e) => {
    e.preventDefault();
    if (!newReview.trim()) return;
    setIsReviewSub(true);
    try {
      const res = await fetch(`${API_URL}/products/${selectedProduct.product_id}/reviews`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: newReview }),
      });
      const updated = await res.json();
      
      setProducts(ps => {
        const newProds = ps.map(p => p.product_id === updated.product_id ? updated : p);
        
        // 1. Update the cache for the category we are currently viewing
        const cacheKey = `conscia_products_${selectedCategory}`;
        sessionStorage.setItem(cacheKey, JSON.stringify(newProds.slice(0, 150)));
        
        // 2. FIX: Clear all OTHER category caches to prevent stale/mismatched scores
        AVAILABLE_CATEGORIES.forEach(cat => {
          if (cat !== selectedCategory) {
            sessionStorage.removeItem(`conscia_products_${cat}`);
          }
        });

        return newProds;
      });
      setSelectedProduct(updated);
      setNewReview('');
      setExplanation(null);
    } catch (e) {
      console.error("Review submission failed", e);
    } finally { 
      setIsReviewSub(false); 
    }
  };

  const handleGetSnapshot = async () => {
    setIsSummaryLoading(true);
    try {
      const res = await fetch(`${API_URL}/snapshot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product: selectedProduct.product_name }),
      });
      const data = await res.json();
      setSummary(data.snapshot);
    } catch (e) {
      console.error("Snapshot generation failed", e);
    } finally { 
      setIsSummaryLoading(false); 
    }
  };

  const handleAnalyzeReviews = async () => {
    setIsExplaining(true);
    try {
      const res = await fetch(`${API_URL}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviews: selectedProduct.reviews }),
      });
      setExplanation(await res.json());
    } catch (e) {
      console.error("XAI explanation failed", e);
    } finally { 
      setIsExplaining(false); 
    }
  };

  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    
    const userMsg = chatInput.trim();
    setChatMessages(prev => [...prev, { sender: 'user', text: userMsg }]);
    setChatInput('');
    setIsChatting(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg }),
      });
      const data = await res.json();
      setChatMessages(prev => [...prev, { sender: 'bot', text: data.reply }]);
    } catch (e) {
      setChatMessages(prev => [...prev, { sender: 'bot', text: 'Sorry, I am having trouble connecting to the server right now.' }]);
    } finally {
      setIsChatting(false);
    }
  };

  // ── Product Card Component (Upgraded to Monogram Design) ────────────────
  const ProductCard = ({ product }) => {
    const avg = getAvgEthical(product);
    const sentScore = product.public_sentiment_score || 50;
    const catMeta = CATEGORY_META[product.category] || { color: '#475569', icon: '📦' };

    return (
      <div className="product-card" onClick={() => handleProductClick(product)}>
        
        {/* New Professional Monogram Placeholder */}
        <div className="card-image-placeholder" style={{ background: catMeta.color }}>
          <div className="card-pattern-overlay"></div>
          
          <div className="card-glass-monogram">
            <span className="card-placeholder-icon">{catMeta.icon}</span>
            <span className="card-placeholder-initials">{getInitials(product.product_name)}</span>
          </div>

          <div className="badge-sentiment" style={{ background: getColor(sentScore) }}>
            {Math.round(sentScore)}
          </div>
        </div>

        <div className="card-body">
          <div className="card-meta">
            <span className="cat-tag">{product.category}</span>
            <span className="price-tag">₹{parseFloat(product.product_price).toFixed(0)}</span>
          </div>
          <h3 className="card-title">{product.product_name}</h3>
          
          <div className="card-ethical-overview">
            <div className="overall-ethical">
              <span>Ethical Score</span>
              <strong style={{ color: getColor(avg) }}>{avg}</strong>
            </div>
            <div className="ethical-icons-row">
              {ETHICAL_DIMENSIONS.map(({ key, icon, color }) => (
                <div key={key} className="icon-wrap" style={{ color: product[key] > 60 ? color : 'var(--text-tertiary)' }} title={`${key}: ${Math.round(product[key])}`}>
                  {icon}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <>
      <div className="app-container">
        
        {/* Navigation Bar */}
        <nav className="navbar">
          <div className="brand" onClick={() => setCurrentView('home')}>Conscia.</div>
          
          <div className="nav-center">
            {currentView === 'home' && (
              <div className="search-bar">
                <IconSearch />
                <input 
                  type="text" 
                  placeholder="Search conscious products..." 
                  value={searchTerm} 
                  onChange={e => setSearch(e.target.value)} 
                />
              </div>
            )}
          </div>

          <div className="nav-actions">
            <button 
              className="nav-btn text-btn" 
              onClick={() => setCurrentView(currentView === 'about' ? 'home' : 'about')}
            >
              {currentView === 'about' ? 'Shop Products' : 'About'}
            </button>
            <button className="nav-btn icon-btn" onClick={toggleTheme} aria-label="Toggle Theme">
              {theme === 'light' ? <IconMoon /> : <IconSun />}
            </button>
          </div>
        </nav>

        {/* Main Content Area */}
        <main className="main-content">
          {currentView === 'about' ? (
            <AboutPage />
          ) : (
            <>
              <header className="hero-section">
                <h1>Shop your values.</h1>
                <p>We analyze environmental impact, labor practices, and public sentiment so you can buy better.</p>
                
                <div className="filter-row">
                  <div className="categories-pill">
                    {AVAILABLE_CATEGORIES.map(c => (
                      <button key={c} className={selectedCategory === c ? 'active' : ''} onClick={() => setCategory(c)}>{c}</button>
                    ))}
                  </div>
                  <select className="sort-select" value={sortBy} onChange={e => setSortBy(e.target.value)}>
                    <option value="default">Sort: Recommended</option>
                    <option value="score">Top Rated</option>
                    <option value="ethical">Most Ethical</option>
                    <option value="price_asc">Price: Low to High</option>
                    <option value="price_desc">Price: High to Low</option>
                  </select>
                </div>
              </header>

              {isInitialLoading ? (
                <div className="loading-grid"><Spinner size={40}/></div>
              ) : (
                <div className="product-grid">
                  {filteredProducts.slice(0, visibleCount).map(p => <ProductCard key={p.product_id} product={p} />)}
                </div>
              )}
              
              <div ref={loaderRef} className="scroll-anchor" style={{ height: '10px', marginTop: '20px' }} />
              
              {isFetching && !isInitialLoading && (
                <div className="loading-grid" style={{ padding: '2rem' }}><Spinner size={30}/></div>
              )}
            </>
          )}
        </main>

        {/* Chat Assistant Floating UI */}
        <button className="chat-fab" onClick={() => setIsChatOpen(!isChatOpen)}>
          {isChatOpen ? <IconX /> : <IconChat />}
        </button>

        {isChatOpen && (
          <div className="chat-window">
            <div className="chat-header">
              <h3><IconSparkle /> Conscia Assistant</h3>
              <button className="chat-close" onClick={() => setIsChatOpen(false)}><IconX /></button>
            </div>
            
            <div className="chat-body">
              {chatMessages.map((msg, idx) => (
                <div key={idx} className={`chat-msg ${msg.sender}`}>
                  {msg.text}
                </div>
              ))}
              {isChatting && (
                <div className="chat-msg bot">
                  <Spinner size={16} />
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <form className="chat-input-area" onSubmit={handleChatSubmit}>
              <input 
                type="text" 
                placeholder="Ask about ethics, scores, or products..." 
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
              />
              <button type="submit" className="chat-send" disabled={!chatInput.trim() || isChatting}>
                <IconSend />
              </button>
            </form>
          </div>
        )}

        {/* Product Detail Modal */}
        {selectedProduct && (
          <div className="modal-overlay" onClick={() => setSelectedProduct(null)}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
              <button className="close-btn" onClick={() => setSelectedProduct(null)}><IconX /></button>
              
              <div className="modal-split">
                <div className="modal-left">
                  <h2>{selectedProduct.product_name}</h2>
                  <div className="modal-meta">
                    <span className="price">₹{parseFloat(selectedProduct.product_price).toFixed(0)}</span>
                    <span className="cat">{selectedProduct.category}</span>
                  </div>

                  <div className="score-boxes">
                    <div className="score-box main-score">
                      <span>Public Sentiment</span>
                      <h1 style={{ color: getColor(selectedProduct.public_sentiment_score) }}>
                        {Math.round(selectedProduct.public_sentiment_score)}
                      </h1>
                    </div>
                    <div className="score-box ethical-score">
                      <span>Avg Ethical Score</span>
                      <h1 style={{ color: getColor(getAvgEthical(selectedProduct)) }}>
                        {getAvgEthical(selectedProduct)}
                      </h1>
                    </div>
                  </div>

                  <div className="ethical-breakdown">
                    <h4>Ethical Breakdown</h4>
                    {ETHICAL_DIMENSIONS.map(({ key, label, color, icon }) => (
                      <div key={key} className="breakdown-row">
                        <div className="bd-label">{icon} {label.split(' ')[0]}</div>
                        <div className="bd-bar-bg">
                          <div className="bd-bar-fill" style={{ width: `${selectedProduct[key]}%`, background: color }} />
                        </div>
                        <span className="bd-num">{Math.round(selectedProduct[key])}</span>
                      </div>
                    ))}
                  </div>

                  <div className="ai-snapshot-card">
                    <button onClick={handleGetSnapshot} disabled={isSummaryLoading} className="btn-secondary">
                      <IconSparkle /> {isSummaryLoading ? 'Analyzing...' : 'Generate AI Snapshot'}
                    </button>
                    {summary && <p className="snapshot-text">{summary}</p>}
                  </div>
                </div>

                <div className="modal-right">
                  <div className="review-section">
                    <h4>Consumer Reviews</h4>
                    <div className="reviews-list">
                      {(selectedProduct.reviews || "No reviews yet.").split(' | ').map((rev, i) => {
                        if(!rev.trim()) return null;
                        return <div key={i} className="review-bubble">{rev}</div>;
                      })}
                    </div>
                    
                    {!explanation ? (
                      <button onClick={handleAnalyzeReviews} className="btn-text" disabled={isExplaining}>
                        {isExplaining ? 'Running XAI Model (This may take a moment)...' : 'Why these scores? (Run Analysis)'}
                      </button>
                    ) : (
                      <div className="xai-results">
                        <h5>Key Phrase Drivers:</h5>
                        {explanation.map(([phrase, weight], i) => (
                          <div key={i} className="xai-pill" style={{ background: weight > 0 ? '#d1fae5' : '#fee2e2', color: weight > 0 ? '#065f46' : '#991b1b' }}>
                            {phrase} ({weight > 0 ? '+' : ''}{(weight * 100).toFixed(1)}%)
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="add-review-box">
                    <h4>Add Your Experience</h4>
                    <form onSubmit={handleReviewSubmit}>
                      <textarea 
                        placeholder="Leave your review here..." 
                        value={newReview} 
                        onChange={e => setNewReview(e.target.value)} 
                      />
                      <button type="submit" disabled={isReviewSubmitting || !newReview.trim()} className="btn-primary">
                        {isReviewSubmitting ? 'Updating Scores...' : 'Submit & Update Sentiment'}
                      </button>
                    </form>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}