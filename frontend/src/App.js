import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css'; // Import the CSS file

// --- Helper & Icon Components ---
const LoadingSpinner = () => ( <svg style={{height: '2rem', width: '2rem', color: 'white', animation: 'spin 1s linear infinite'}} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle style={{opacity: 0.25}} cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path style={{opacity: 0.75}} fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>);
const ChatIcon = ({ style }) => ( <svg style={style} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M4.913 2.658c1.32-.977 2.944-.977 4.264 0l8.165 6.02a4.5 4.5 0 0 1 2.223 3.865v2.92a4.5 4.5 0 0 1-1.226 3.125l-2.89 2.89a4.5 4.5 0 0 1-6.364 0l-.707-.707a4.5 4.5 0 0 1 0-6.364l2.89-2.89a4.5 4.5 0 0 1 3.125-1.226h2.92a4.5 4.5 0 0 1 3.865 2.223l.235.348a.75.75 0 0 1-1.28.854l-.235-.348a3 3 0 0 0-2.576-1.485h-2.92a3 3 0 0 0-2.083.818l-2.89 2.89a3 3 0 0 0 0 4.243l.707.707a3 3 0 0 0 4.243 0l2.89-2.89a3 3 0 0 0 .818-2.083v-2.92a3 3 0 0 0-1.485-2.576l-8.165-6.02a3 3 0 0 0-2.842 0l-8.165 6.02a3 3 0 0 0-1.485 2.576v2.92a3 3 0 0 0 .818 2.083l2.89 2.89a3 3 0 0 0 4.243 0l.707.707a3 3 0 0 0 0-4.243l-2.89-2.89a3 3 0 0 0-2.083-.818H3.34a3 3 0 0 0-2.576 1.485l-.235.348a.75.75 0 0 1-1.28-.854l-.235-.348A4.5 4.5 0 0 1 3.34 9.54V6.62a4.5 4.5 0 0 1 1.573-3.412L4.913 2.658Z" /></svg>);
const CloseIcon = ({ style }) => ( <svg style={style} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M5.47 5.47a.75.75 0 0 1 1.06 0L12 10.94l5.47-5.47a.75.75 0 1 1 1.06 1.06L13.06 12l5.47 5.47a.75.75 0 1 1-1.06 1.06L12 13.06l-5.47 5.47a.75.75 0 0 1-1.06-1.06L10.94 12 5.47 6.53a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" /></svg>);
const SearchIcon = ({ className }) => ( <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M10.5 3.75a6.75 6.75 0 1 0 0 13.5 6.75 6.75 0 0 0 0-13.5ZM2.25 10.5a8.25 8.25 0 1 1 14.59 5.28l4.69 4.69a.75.75 0 1 1-1.06 1.06l-4.69-4.69A8.25 8.25 0 0 1 2.25 10.5Z" clipRule="evenodd" /></svg>);

export default function App() {
    const [products, setProducts] = useState([]);
    const [showTooltip, setShowTooltip] = useState(true);
    const [filteredProducts, setFilteredProducts] = useState([]);
    const [selectedCategory, setSelectedCategory] = useState('All');
    const [searchTerm, setSearchTerm] = useState('');
    const [visibleCount, setVisibleCount] = useState(20);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedProduct, setSelectedProduct] = useState(null);
    const [isModalLoading, setIsModalLoading] = useState(false);
    const [explanation, setExplanation] = useState(null);
    const [summary, setSummary] = useState('');
    const [isSummaryLoading, setIsSummaryLoading] = useState(false);
    const [isChatOpen, setIsChatOpen] = useState(false);
    const [chatHistory, setChatHistory] = useState([]);
    const [chatInput, setChatInput] = useState('');
    const [isChatLoading, setIsChatLoading] = useState(false);
    const observer = useRef();
    const chatBodyRef = useRef(null);
    const API_URL = 'http://localhost:5000/api';

    useEffect(() => {
        const fetchProducts = async () => {
            try {
                setIsLoading(true); setError(null);
                const response = await fetch(`${API_URL}/products`);
                if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}. Is the backend server running?`);
                const data = await response.json();
                setProducts(data); setFilteredProducts(data);
            } catch (e) {
                console.error("Failed to fetch products:", e); setError(e.message);
            } finally {
                setIsLoading(false);
            }
        };
        fetchProducts();
    }, []);

    useEffect(() => {
        let result = products;
        if (selectedCategory !== 'All') result = result.filter(p => p.category === selectedCategory);
        if (searchTerm) result = result.filter(p => p.product_name.toLowerCase().includes(searchTerm.toLowerCase()));
        setFilteredProducts(result); setVisibleCount(20);
    }, [selectedCategory, searchTerm, products]);

    const lastProductElementRef = useCallback(node => {
        if (isLoading) return;
        if (observer.current) observer.current.disconnect();
        observer.current = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && visibleCount < filteredProducts.length) {
                setVisibleCount(prevCount => prevCount + 20);
            }
        });
        if (node) observer.current.observe(node);
    }, [isLoading, visibleCount, filteredProducts.length]);

    const handleProductClick = async (product) => {
        setSelectedProduct(product); 
        setIsModalLoading(true);
        setExplanation(null); 
        setSummary('');
        try {
            const explainRes = await fetch(`${API_URL}/explain`, { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify({ reviews: product.reviews }) 
            });
            if (explainRes.ok) setExplanation(await explainRes.json());
        } catch (e) {
            console.error("Failed to fetch explanation:", e);
        } finally {
            setIsModalLoading(false);
        }
    };

    const handleGetSummary = async () => {
        if (!selectedProduct) return;
        const scores = {
            environmental_impact_score: selectedProduct.environmental_impact_score,
            labor_rights_score: selectedProduct.labor_rights_score,
            animal_welfare_score: selectedProduct.animal_welfare_score,
            corporate_governance_score: selectedProduct.corporate_governance_score
        };
        setIsSummaryLoading(true); setSummary('');
        try {
            const response = await fetch(`${API_URL}/snapshot`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ product_name: selectedProduct.product_name, scores }) });
            const data = await response.json();
            setSummary(data.summary);
        } catch (e) {
            console.error("Failed to get summary:", e); setSummary('Could not generate summary at this time.');
        } finally {
            setIsSummaryLoading(false);
        }
    };

    useEffect(() => {
        if (chatBodyRef.current) chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }, [chatHistory, isChatLoading]);

    const handleChatSubmit = async (e) => {
        e.preventDefault();
        if (!chatInput.trim() || isChatLoading) return;
        const newUserMessage = { role: 'user', parts: [{ text: chatInput }] };
        const newHistory = [...chatHistory, newUserMessage];
        setChatHistory(newHistory); setChatInput(''); setIsChatLoading(true);
        try {
            const response = await fetch(`${API_URL}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ history: newHistory }) });
            const data = await response.json();
            const botMessage = { role: 'model', parts: [{ text: data.reply || "I'm sorry, I couldn't get a response." }] };
            setChatHistory([...newHistory, botMessage]);
        } catch (e) {
            console.error("Chat API error:", e);
            const errorMessage = { role: 'model', parts: [{ text: "I'm having trouble connecting. Please try again later." }] };
            setChatHistory([...newHistory, errorMessage]);
        } finally {
            setIsChatLoading(false);
        }
    };
    
    const parseMarkdown = (text) => {
        if (!text) return { __html: '' };
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/^- (.*$)/gm, '<li style="margin-left: 1.25rem; list-style-type: disc;">$1</li>');
        text = text.replace(/(?:\r\n|\r|\n){2,}/g, '<br /><br />');
        return { __html: text };
    };

    const categories = ['All', ...new Set(products.map(p => p.category))];

    const HighlightedReview = ({ text, explanation }) => {
        if (!text) return null;
        if (!explanation) return <span>{text}</span>;
        const words = text.split(/(\s+)/);
        const colorMap = { environmental: 'rgba(52, 211, 153, 0.3)', labor: 'rgba(96, 165, 250, 0.3)', animal_welfare: 'rgba(250, 204, 21, 0.3)', governance: 'rgba(192, 132, 252, 0.3)' };
        const wordMap = new Map();
        Object.entries(explanation).forEach(([category, catWords]) => {
            (catWords || []).forEach(word => wordMap.set(word.toLowerCase(), colorMap[category]));
        });
        return ( <p style={{color: '#d1d5db', lineHeight: 1.6}}>{words.map((word, i) => { const cleanWord = word.replace(/[.,!?]/g, '').toLowerCase(); const bgColor = wordMap.get(cleanWord); return bgColor ? <span key={i} style={{padding: '0 0.25rem', borderRadius: '0.375rem', backgroundColor: bgColor}}>{word}</span> : <span key={i}>{word}</span>; })}</p> );
    };

    const renderModalContent = () => {
        const scores = {
            environmental_impact_score: selectedProduct.environmental_impact_score,
            labor_rights_score: selectedProduct.labor_rights_score,
            animal_welfare_score: selectedProduct.animal_welfare_score,
            corporate_governance_score: selectedProduct.corporate_governance_score
        };

        return (
            <>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem'}}>
                    <div>
                        <h2 style={{fontSize: '1.875rem', fontWeight: 700, color: '#f1f5f9', margin: 0}}>{selectedProduct.product_name}</h2>
                        <p style={{fontSize: '1.25rem', fontWeight: 600, color: '#67e8f9', margin: '0.25rem 0 0 0'}}>${selectedProduct.product_price ? parseFloat(selectedProduct.product_price).toFixed(2) : '0.00'}</p>
                    </div>
                    <button onClick={() => setSelectedProduct(null)} style={{background: 'none', border: 'none', cursor: 'pointer', padding: '0.5rem'}}>
                        <CloseIcon style={{height: '1.5rem', width: '1.5rem', color: '#94a3b8'}} />
                    </button>
                </div>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(1, 1fr)', gap: '2rem'}} id="modal-grid">
                    <div style={{display: 'flex', flexDirection: 'column', gap: '1rem'}}>
                        <h3 style={{fontSize: '1.25rem', fontWeight: 600, borderBottom: '1px solid #334155', paddingBottom: '0.5rem', margin: 0}}>Ethical Scores</h3>
                        <div style={{display: 'flex', flexDirection: 'column', gap: '0.75rem'}}>
                            {[
                                { label: 'Environmental', key: 'environmental_impact_score', colors: ['#2dd4bf', '#0d9488'] },
                                { label: 'Labor Rights', key: 'labor_rights_score', colors: ['#60a5fa', '#2563eb'] },
                                { label: 'Animal Welfare', key: 'animal_welfare_score', colors: ['#facc15', '#d97706'] },
                                { label: 'Governance', key: 'corporate_governance_score', colors: ['#c084fc', '#7e22ce'] },
                            ].map(item => (
                                <div key={item.key}>
                                    <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem'}}>
                                        <span style={{fontWeight: 500, color: '#cbd5e1'}}>{item.label}</span>
                                        <span style={{fontWeight: 700, color: 'white'}}>{(scores[item.key] * 10).toFixed(0)} / 100</span>
                                    </div>
                                    <div style={{width: '100%', height: '0.75rem', borderRadius: '9999px', backgroundColor: '#1e293b'}}>
                                        <div style={{height: '0.75rem', borderRadius: '9999px', backgroundImage: `linear-gradient(90deg, ${item.colors[0]}, ${item.colors[1]})`, width: `${scores[item.key] * 10}%`}}></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                         <div style={{paddingTop: '1rem'}}>
                            <button onClick={handleGetSummary} disabled={isSummaryLoading} style={{width: '100%', backgroundColor: isSummaryLoading ? '#475569' : 'var(--color-accent)', color: 'white', fontWeight: 700, padding: '0.75rem 1rem', borderRadius: '0.5rem', border: 'none', cursor: 'pointer', transition: 'background-color 0.2s ease'}}>
                                {isSummaryLoading ? 'Generating...' : 'Get AI Ethical Snapshot'}
                            </button>
                            {summary && <div style={{marginTop: '1rem', padding: '1rem', backgroundColor: 'rgba(30, 41, 59, 0.6)', borderRadius: '0.5rem', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: 1.6}}>{summary}</div>}
                        </div>
                    </div>
                    <div style={{display: 'flex', flexDirection: 'column', gap: '1rem'}}>
                         <h3 style={{fontSize: '1.25rem', fontWeight: 600, borderBottom: '1px solid #334155', paddingBottom: '0.5rem', margin:0}}>Review Analysis (XAI)</h3>
                         <div style={{padding: '1rem', backgroundColor: 'rgba(30, 41, 59, 0.6)', borderRadius: '0.5rem', maxHeight: '20rem', overflowY: 'auto'}}>
                            {isModalLoading ? <div style={{display: 'flex', justifyContent: 'center'}}><LoadingSpinner /></div> : <HighlightedReview text={selectedProduct.reviews} explanation={explanation} />}
                        </div>
                        <p style={{fontSize: '0.75rem', color: '#64748b', textAlign: 'center', margin: 0}}>Words are highlighted based on their influence on the AI's scores.</p>
                    </div>
                </div>
                <style>
                  {`@media (min-width: 1024px) { #modal-grid { grid-template-columns: repeat(2, 1fr); }}`}
                </style>
            </>
        )
    };

     const renderChatWindow = () => (
        <div className="chat-window glass-effect">
            <header style={{padding: '1rem', borderBottom: '1px solid #334155', flexShrink: 0}}>
                <h3 style={{fontWeight: 700, fontSize: '1.125rem', textAlign: 'center', margin: 0}}>Conscia Assistant</h3>
            </header>
            <div ref={chatBodyRef} style={{flexGrow: 1, padding: '1rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem'}}>
                {chatHistory.length === 0 && (
                    <div style={{textAlign: 'center', color: '#94a3b8', fontSize: '0.875rem', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
                        <p>Ask me to find products, compare items, or get brand insights!</p>
                    </div>
                )}
                {chatHistory.map((msg, index) => (
                    <div key={index} style={{display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start'}}>
                        <div style={{padding: '0.75rem 1rem', borderRadius: '1rem', maxWidth: '80%', fontSize: '0.875rem', backgroundColor: msg.role === 'user' ? 'var(--color-accent)' : '#334155', color: msg.role === 'user' ? 'white' : '#e2e8f0', borderBottomRightRadius: msg.role === 'user' ? '0' : '1rem', borderBottomLeftRadius: msg.role === 'user' ? '1rem' : '0', overflowWrap: 'break-word'}}>
                           <div dangerouslySetInnerHTML={parseMarkdown(msg.parts[0].text)} />
                        </div>
                    </div>
                ))}
                {isChatLoading && (
                    <div style={{display: 'flex', justifyContent: 'flex-start'}}>
                        <div style={{padding: '0.75rem', borderRadius: '1rem', backgroundColor: '#334155', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', borderBottomLeftRadius: 0}}>
                            <div style={{width: '0.5rem', height: '0.5rem', backgroundColor: '#94a3b8', borderRadius: '9999px', animation: 'bounce 1.4s infinite ease-in-out both', animationDelay: '0.16s'}}></div>
                            <div style={{width: '0.5rem', height: '0.5rem', backgroundColor: '#94a3b8', borderRadius: '9999px', animation: 'bounce 1.4s infinite ease-in-out both', animationDelay: '0.32s'}}></div>
                            <div style={{width: '0.5rem', height: '0.5rem', backgroundColor: '#94a3b8', borderRadius: '9999px', animation: 'bounce 1.4s infinite ease-in-out both', animationDelay: '0.48s'}}></div>
                        </div>
                    </div>
                )}
            </div>
            <form onSubmit={handleChatSubmit} style={{padding: '1rem', borderTop: '1px solid #334155', flexShrink: 0}}>
                <input type="text" value={chatInput} onChange={(e) => setChatInput(e.target.value)} placeholder="Ask anything..." className="search-input" style={{paddingLeft: '1rem', paddingRight: '1rem'}}/>
            </form>
            <style>
              {`
                @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1.0); } }
                @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
              `}
            </style>
        </div>
    );

    return (
        <div className="main-container">
            <header className="app-header">
                <h1 className="cursive-title">Conscia</h1>
                <p className="app-subtitle">Your AI-Powered Ethical Shopping Companion. Uncover the story behind every product.</p>
            </header>

            <main>
                <div className="filter-container glass-effect">
                    <h2 className="filter-title">Explore Categories</h2>
                    <div className="category-buttons">
                        {categories.map(cat => <button key={cat} onClick={() => setSelectedCategory(cat)} className={`category-btn ${selectedCategory === cat ? 'active' : ''}`}>{cat}</button>)}
                    </div>
                    <div className="search-container">
                        <SearchIcon className="search-icon" />
                        <input type="text" placeholder="Search products..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} className="search-input"/>
                    </div>
                </div>

                {isLoading ? <div className="center-flex"><LoadingSpinner /></div>
                : error ? <div className="error-box"><h3 className="error-title">Connection Error</h3><p className="error-message">{error}</p><p className="error-subtext">Please make sure the Python backend server is running correctly on `localhost:5000`.</p></div>
                : (
                    <>
                        {filteredProducts.length > 0 ? (
                          <div className="product-grid">
                            {filteredProducts.slice(0, visibleCount).map((product, index) => (
                              <div
                                key={product.product_id}
                                ref={index === visibleCount - 1 ? lastProductElementRef : null}
                                onClick={() => handleProductClick(product)}
                                className="product-card glass-effect"
                              >
                                <div className="product-info">
                                  <h3 className="product-name">{product.product_name}</h3>
                                  <p className="product-category">{product.category}</p>
                                  <p className="product-price">
                                    ${product.product_price ? parseFloat(product.product_price).toFixed(2) : '0.00'}
                                  </p>
                                </div>

                                {/* --- New Score Bars --- */}
                                  <div className="card-scores-container">
                                    <div className="card-score-bar-wrapper">
                                      <span className="card-score-label">Env</span>
                                      <div className="card-score-bar-bg">
                                        <div
                                          className="card-score-bar-fill bar-env"
                                          style={{ width: `${(product.environmental_impact_score*10)}%` }}
                                        />
                                      </div>
                                      <span className="card-score-value">{product.environmental_impact_score*10}/100</span>
                                    </div>

                                    <div className="card-score-bar-wrapper">
                                      <span className="card-score-label">Labor</span>
                                      <div className="card-score-bar-bg">
                                        <div
                                          className="card-score-bar-fill bar-labor"
                                          style={{ width: `${(product.labor_rights_score*10)}%` }}
                                        />
                                      </div>
                                      <span className="card-score-value">{product.labor_rights_score*10}/100</span>
                                    </div>

                                    <div className="card-score-bar-wrapper">
                                      <span className="card-score-label">Animal</span>
                                      <div className="card-score-bar-bg">
                                        <div
                                          className="card-score-bar-fill bar-animal"
                                          style={{ width: `${(product.animal_welfare_score*10)}%` }}
                                        />
                                      </div>
                                      <span className="card-score-value">{product.animal_welfare_score*10}/100</span>
                                    </div>

                                    <div className="card-score-bar-wrapper">
                                      <span className="card-score-label">Gov</span>
                                      <div className="card-score-bar-bg">
                                        <div
                                          className="card-score-bar-fill bar-gov"
                                          style={{ width: `${(product.corporate_governance_score*10)}%` }}
                                        />
                                      </div>
                                      <span className="card-score-value">{product.corporate_governance_score*10}/100</span>
                                    </div>
                                  </div>

                              </div>
                            ))}
                          </div>
                        ) : (
                            <div className="empty-state"><h3 className="empty-title">No Products Found</h3><p>Try adjusting your search or category filters.</p></div>
                        )}
                        {visibleCount < filteredProducts.length && !isLoading && <div className="center-flex" style={{marginTop: '2rem'}}><LoadingSpinner /></div>}
                    </>
                )}
            </main>
            {selectedProduct && (
                <div className="modal-overlay" onClick={() => setSelectedProduct(null)}>
                    <div className="modal-content glass-effect" onClick={(e) => e.stopPropagation()}>
                        {renderModalContent()}
                    </div>
                </div>
            )}
              <>
                {/* Chat Tooltip */}
                {showTooltip && (
                  <div className="chat-tooltip glass-effect">
                    <div className="chat-tooltip-header">
                      <span className="chat-tooltip-title">💡 Need Help?</span>
                      <button
                        className="chat-tooltip-close"
                        onClick={() => setShowTooltip(false)}
                      >
                        ×
                      </button>
                    </div>
                    <div>
                      Ask our <span style={{ color: "var(--color-accent)" }}>AI Assistant</span> anything!
                    </div>
                  </div>
                )}

                {/* Chatbot Floating Button */}
                <div className="chat-fab">
                  <button
                    className="chat-fab-button"
                    onClick={() => setIsChatOpen(!isChatOpen)}
                  >
                    💬
                  </button>
                </div>
              </>
            );
            
            <div className="chat-fab">
                <button onClick={() => setIsChatOpen(!isChatOpen)} className="chat-fab-button">
                    {isChatOpen ? <CloseIcon style={{height: '2rem', width: '2rem'}}/> : <ChatIcon style={{height: '2rem', width: '2rem'}}/>}
                </button>
            </div>
            
            {isChatOpen && renderChatWindow()}
        </div>
    );
}

