import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css'; 

// --- Helper & Icon Components ---
const LoadingSpinner = () => ( <svg style={{height: '2rem', width: '2rem', color: 'white', animation: 'spin 1s linear infinite'}} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle style={{opacity: 0.25}} cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path style={{opacity: 0.75}} fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>);
const ChatIcon = ({ style }) => ( <svg style={style} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M4.913 2.658c1.32-.977 2.944-.977 4.264 0l8.165 6.02a4.5 4.5 0 0 1 2.223 3.865v2.92a4.5 4.5 0 0 1-1.226 3.125l-2.89 2.89a4.5 4.5 0 0 1-6.364 0l-.707-.707a4.5 4.5 0 0 1 0-6.364l2.89-2.89a4.5 4.5 0 0 1 3.125-1.226h2.92a4.5 4.5 0 0 1 3.865 2.223l.235.348a.75.75 0 0 1-1.28.854l-.235-.348a3 3 0 0 0-2.576-1.485h-2.92a3 3 0 0 0-2.083.818l-2.89 2.89a3 3 0 0 0 0 4.243l.707.707a3 3 0 0 0 4.243 0l2.89-2.89a3 3 0 0 0 .818-2.083v-2.92a3 3 0 0 0-1.485-2.576l-8.165-6.02a3 3 0 0 0-2.842 0l-8.165 6.02a3 3 0 0 0-1.485 2.576v2.92a3 3 0 0 0 .818 2.083l2.89 2.89a3 3 0 0 0 4.243 0l.707.707a3 3 0 0 0 0-4.243l-2.89-2.89a3 3 0 0 0-2.083-.818H3.34a3 3 0 0 0-2.576 1.485l-.235.348a.75.75 0 0 1-1.28-.854l-.235-.348A4.5 4.5 0 0 1 3.34 9.54V6.62a4.5 4.5 0 0 1 1.573-3.412L4.913 2.658Z" /></svg>);
const CloseIcon = ({ style }) => ( <svg style={style} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M5.47 5.47a.75.75 0 0 1 1.06 0L12 10.94l5.47-5.47a.75.75 0 1 1 1.06 1.06L13.06 12l5.47 5.47a.75.75 0 1 1-1.06 1.06L12 13.06l-5.47 5.47a.75.75 0 0 1-1.06-1.06L10.94 12 5.47 6.53a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" /></svg>);
const SearchIcon = ({ className }) => ( <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M10.5 3.75a6.75 6.75 0 1 0 0 13.5 6.75 6.75 0 0 0 0-13.5ZM2.25 10.5a8.25 8.25 0 1 1 14.59 5.28l4.69 4.69a.75.75 0 1 1-1.06 1.06l-4.69-4.69A8.25 8.25 0 0 1 2.25 10.5Z" clipRule="evenodd" /></svg>);
const HamburgerIcon = ({ style }) => (<svg style={style} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M3 6.75A.75.75 0 0 1 3.75 6h16.5a.75.75 0 0 1 0 1.5H3.75A.75.75 0 0 1 3 6.75ZM3 12a.75.75 0 0 1 .75-.75h16.5a.75.75 0 0 1 0 1.5H3.75A.75.75 0 0 1 3 12Zm0 5.25a.75.75 0 0 1 .75-.75h16.5a.75.75 0 0 1 0 1.5H3.75a.75.75 0 0 1-.75-.75Z" clipRule="evenodd" /></svg>);

// --- Refactored Sub-Components ---

const CircularProgressBar = ({ score }) => {
    const radius = 36;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;

    return (
        <div className="progress-ring">
            <svg className="progress-ring__svg" width="80" height="80">
                <circle className="progress-ring__bg" r={radius} cx="40" cy="40" />
                <circle
                    className="progress-ring__fg"
                    r={radius}
                    cx="40"
                    cy="40"
                    style={{ strokeDasharray: circumference, strokeDashoffset: offset }}
                />
            </svg>
            <span className="progress-ring__text">{score.toFixed(0)}</span>
        </div>
    );
};

const Navbar = React.memo(({ activeView, onNavClick, onToggleNav, isNavOpen }) => (
    <nav className="navbar">
        {activeView === 'about' ? (<div className="navbar-brand" onClick={() => onNavClick('home')}>Conscia</div>) : (<div></div>)}
        <div className="navbar-menu-container">
            <button className="hamburger-btn" onClick={onToggleNav}><HamburgerIcon style={{height: '1.75rem', width: '1.75rem'}} /></button>
            {isNavOpen && (<div className="nav-menu glass-effect">
                <button onClick={() => onNavClick('home')} className="nav-menu-item">Home</button>
                <button onClick={() => onNavClick('about')} className="nav-menu-item">About</button>
            </div>)}
        </div>
    </nav>
));

const HighlightedReview = React.memo(({ explanationData }) => {
    if (!explanationData || Object.keys(explanationData).length === 0) return <div style={{color: 'var(--color-text-secondary)'}}>No review text available or analysis is pending.</div>;

    const colorMap = {
        positive: 'rgba(52, 211, 153, 0.25)',
        negative: 'rgba(239, 68, 68, 0.25)'
    };

    const wordMap = new Map();
    Object.values(explanationData).forEach(item => {
        const { positive, negative } = item.explanation;
        Object.values(positive).flat().forEach(word => wordMap.set(word.toLowerCase(), colorMap.positive));
        Object.values(negative).flat().forEach(word => wordMap.set(word.toLowerCase(), colorMap.negative));
    });

    return Object.values(explanationData).map((item, index) => (
        <p key={index} className="review-item">
            {item.review_text.split(/(\s+)/).map((word, i) => {
                const cleanWord = word.replace(/[.,!?]/g, '').toLowerCase();
                const bgColor = wordMap.get(cleanWord);
                return bgColor ? 
                    <span key={i} style={{backgroundColor: bgColor, padding: '2px 4px', borderRadius: '4px'}}>{word}</span> : 
                    <span key={i}>{word}</span>;
            })}
        </p>
    ));
});

const AboutPage = React.memo(() => (
    <div className="about-page glass-effect">
        <h2 className="about-title-font">About Conscia</h2>
        <h3 className="about-heading-font">Our Mission</h3>
        <p>Our mission is to empower consumers with transparent ethical data. We show you the company's verified score alongside the public's real-world sentiment, so you can make informed decisions that align with your values.</p>
        <h3 className="about-heading-font">How It Works</h3>
        <p>We present two scores: a static Factual Score based on verified data, and a dynamic Public Sentiment Score from AI-analyzed reviews. When these scores are close, it builds confidence. When they are far apart, it empowers you to dig deeper.</p>
    </div>
));

const ProductModal = React.memo(({ product, factualScores, sentimentScores, explanation, summary, newReview, userSubmittedReviews, isModalLoading, isSummaryLoading, isReviewSubmitting, onClose, onGetSummary, onReviewChange, onReviewSubmit }) => {
    if (!product) return null;

    const avgSentiment = sentimentScores ? 
        (sentimentScores.environmental_impact_score + sentimentScores.labor_rights_score + sentimentScores.animal_welfare_score + sentimentScores.corporate_governance_score) / 4 * 10 
        : 0;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content glass-effect" onClick={(e) => e.stopPropagation()}>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem'}}>
                    <div>
                        <h2 style={{fontSize: '1.875rem', fontWeight: 700, color: '#f1f5f9', margin: 0}}>{product.product_name}</h2>
                        <p style={{fontSize: '1.25rem', fontWeight: 600, color: 'var(--color-accent)', margin: '0.25rem 0 0 0'}}>${product.product_price ? parseFloat(product.product_price).toFixed(2) : '0.00'}</p>
                    </div>
                    <button onClick={onClose} style={{background: 'none', border: 'none', cursor: 'pointer', padding: '0.5rem'}}><CloseIcon style={{height: '1.5rem', width: '1.5rem', color: '#94a3b8'}} /></button>
                </div>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(1, 1fr)', gap: '2rem'}} id="modal-grid">
                    <div style={{display: 'flex', flexDirection: 'column', gap: '1.5rem'}}>
                        <div>
                            <h3 className="score-heading">Brand's Factual Score</h3>
                            <div className="score-card-grid">
                                {isModalLoading || !factualScores ? <div style={{display: 'flex', justifyContent: 'center'}}><LoadingSpinner /></div> : Object.entries(factualScores).map(([key, value]) => (
                                    <div className="score-card-item" key={key}>
                                        <div className="score-card-label"><span>{key}</span><span>{value}</span></div>
                                        <div className="score-card-bar-bg"><div className={`score-card-bar-fill bar-${key.split(' ')[0].toLowerCase()}`} style={{width: `${value}%`}}></div></div>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div>
                             <h3 className="score-heading">Public Sentiment Score</h3>
                             <div className="single-score-bar-container">
                                {isModalLoading || !sentimentScores ? <div style={{display: 'flex', justifyContent: 'center'}}><LoadingSpinner /></div> : (<>
                                    <div className="single-score-bar-bg"><div className="single-score-bar-fill" style={{width: `${avgSentiment}%`}}></div></div>
                                    <div className="single-score-value">{avgSentiment.toFixed(0)} / 100</div>
                                </>)}
                             </div>
                            <div style={{paddingTop: '1.5rem'}}>
                                <button onClick={onGetSummary} disabled={isSummaryLoading} className="review-submit-btn">{isSummaryLoading ? 'Generating...' : 'Get AI Ethical Snapshot'}</button>
                                {summary && <div className="summary-box">{summary}</div>}
                            </div>
                        </div>
                    </div>
                    <div style={{display: 'flex', flexDirection: 'column', gap: '1rem'}}>
                        <h3 className="score-heading">Review Analysis (XAI)</h3>
                        <div className="review-analysis-box">
                            {isModalLoading ? <div style={{display: 'flex', justifyContent: 'center'}}><LoadingSpinner /></div> : <HighlightedReview explanationData={explanation} />}
                        </div>
                        <p className="xai-explainer">Words are highlighted based on their influence on the public sentiment scores.</p>
                        
                        {userSubmittedReviews.length > 0 && (<>
                            <h3 className="score-heading">Your Submitted Reviews</h3>
                            <div className="user-reviews-box">
                                {userSubmittedReviews.map((review, index) => <p key={index} className="user-review-item">{review}</p>)}
                            </div>
                        </>)}

                        <form onSubmit={onReviewSubmit} style={{ paddingTop: '1rem' }}>
                            <h3 className="review-form-title">Add Your Review</h3>
                            <textarea className="review-textarea" value={newReview} onChange={onReviewChange} placeholder="Share your thoughts on this product..."/>
                            <button type="submit" disabled={isReviewSubmitting} className="review-submit-btn">{isReviewSubmitting ? 'Submitting...' : 'Submit Review'}</button>
                        </form>
                    </div>
                </div>
                <style>{`@media (min-width: 1024px) { #modal-grid { grid-template-columns: repeat(2, 1fr); }}`}</style>
            </div>
        </div>
    );
});

const ChatWindow = React.memo(({ chatHistory, isChatLoading, chatInput, onChatInputChange, onChatSubmit, chatBodyRef }) => (
    <div className="chat-window glass-effect">
        <header className="chat-header"><h3>Conscia Assistant</h3></header>
        <div ref={chatBodyRef} className="chat-body">
            {chatHistory.length === 0 && (<div className="chat-empty-state"><p>Ask me to find products, compare items, or get brand insights!</p></div>)}
            {chatHistory.map((msg, index) => (<div key={index} className={`chat-message ${msg.role}`}><div className="chat-bubble" dangerouslySetInnerHTML={{ __html: msg.parts[0].text.replace(/\n/g, '<br />') }} /></div>))}
            {isChatLoading && (<div className="chat-message model"><div className="chat-bubble"><div className="typing-indicator"><span></span><span></span><span></span></div></div></div>)}
        </div>
        <form onSubmit={onChatSubmit} className="chat-input-form"><input type="text" value={chatInput} onChange={onChatInputChange} placeholder="Ask anything..." className="chat-input"/></form>
    </div>
));


// --- Main App Component ---
export default function App() {
    const [products, setProducts] = useState([]);
    const [filteredProducts, setFilteredProducts] = useState([]);
    const [selectedCategory, setSelectedCategory] = useState(null);
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
    const [newReview, setNewReview] = useState('');
    const [userSubmittedReviews, setUserSubmittedReviews] = useState([]);
    const [isReviewSubmitting, setIsReviewSubmitting] = useState(false);
    const [activeView, setActiveView] = useState('home');
    const [isNavOpen, setIsNavOpen] = useState(false);
    const [showTooltip, setShowTooltip] = useState(true);
    const [factualScores, setFactualScores] = useState(null);
    const [sentimentScores, setSentimentScores] = useState(null);
    const observer = useRef();
    const chatBodyRef = useRef(null);
    const API_URL = 'http://localhost:5000/api';

    const categories = ['All', ...new Set(products.map(p => p.category))];
    
    useEffect(() => {
        const timer = setTimeout(() => setShowTooltip(false), 5000);
        return () => clearTimeout(timer);
    }, []);

    useEffect(() => {
        const fetchProducts = async () => {
            try {
                setIsLoading(true); setError(null);
                const response = await fetch(`${API_URL}/products`);
                if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}. Is the backend server running?`);
                const data = await response.json();
                setProducts(data);
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
        if (selectedCategory && selectedCategory !== 'All') result = products.filter(p => p.category === selectedCategory);
        if (searchTerm) result = result.filter(p => p.product_name.toLowerCase().includes(searchTerm.toLowerCase()));
        setFilteredProducts(result);
        setVisibleCount(20);
    }, [selectedCategory, searchTerm, products]);
    
    const handleCategoryClick = (category) => { setSelectedCategory(category); setActiveView('products'); };
    const handleNavClick = (view) => {
        if (activeView === view) { setIsNavOpen(false); return; }
        setActiveView(view); setIsNavOpen(false);
        if (view === 'home') setSelectedCategory(null);
    };

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
        setExplanation(null); setSummary(''); setNewReview(''); setFactualScores(null); setUserSubmittedReviews([]);
        
        setSentimentScores({
            environmental_impact_score: product.environmental_impact_score,
            labor_rights_score: product.labor_rights_score,
            animal_welfare_score: product.animal_welfare_score,
            corporate_governance_score: product.corporate_governance_score,
        });

        try {
            const [factualRes, explainRes] = await Promise.all([
                fetch(`${API_URL}/factual-score/${product.product_id}`),
                fetch(`${API_URL}/explain`, { 
                    method: 'POST', 
                    headers: { 'Content-Type': 'application/json' }, 
                    body: JSON.stringify({ reviews: product.reviews }) 
                })
            ]);
            
            if (factualRes.ok) setFactualScores(await factualRes.json());
            if (explainRes.ok) setExplanation(await explainRes.json());

        } catch (e) {
            console.error("Failed to fetch product details:", e);
        } finally {
            setIsModalLoading(false);
        }
    };

    const handleGetSummary = async () => {
        if (!selectedProduct || !sentimentScores) return;
        setIsSummaryLoading(true); setSummary('');
        try {
            const response = await fetch(`${API_URL}/snapshot`, { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify({ product_name: selectedProduct.product_name, scores: sentimentScores }) 
            });
            const data = await response.json();
            setSummary(data.summary);
        } catch (e) {
            console.error("Failed to get summary:", e); setSummary('Could not generate summary at this time.');
        } finally {
            setIsSummaryLoading(false);
        }
    };

    const handleReviewSubmit = async (e) => {
        e.preventDefault();
        if (!newReview.trim()) return;
        setIsReviewSubmitting(true);
        try {
            const reviewResponse = await fetch(`${API_URL}/submit-review`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ product_id: selectedProduct.product_id, review_text: newReview })
            });

            if (!reviewResponse.ok) throw new Error('Failed to submit review.');
            
            const { scores } = await reviewResponse.json();
            setSentimentScores(scores);
            setUserSubmittedReviews(prev => [...prev, newReview]);

            const updatedReviews = selectedProduct.reviews + "|||" + newReview;
            const updatedProductInList = { ...selectedProduct, reviews: updatedReviews, ...scores };
            
            setProducts(products.map(p => p.product_id === selectedProduct.product_id ? updatedProductInList : p));
            setSelectedProduct(updatedProductInList); 
            
            const explainRes = await fetch(`${API_URL}/explain`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reviews: updatedReviews })
            });
            if (explainRes.ok) setExplanation(await explainRes.json());
            
            setNewReview('');
        } catch (error) {
            console.error("Error submitting review:", error);
        } finally {
            setIsReviewSubmitting(false);
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
    
    return (
        <div className="main-container">
            <Navbar 
                activeView={activeView}
                onNavClick={handleNavClick}
                isNavOpen={isNavOpen}
                onToggleNav={() => setIsNavOpen(!isNavOpen)}
            />
            <header className="app-header">
                {activeView !== 'about' && (<>
                    <h1 className="main-title">Conscia</h1>
                    <p className="app-subtitle">Your AI-Powered Ethical Shopping Companion. Uncover the story behind every product.</p>
                </>)}
            </header>
            <main>
                {activeView === 'home' || activeView === 'products' ? (<>
                    <div className="controls-container glass-effect">
                        <h2 className="controls-title">Explore Products</h2>
                        <div className="category-buttons">{categories.map((cat) => (<button key={cat} onClick={() => handleCategoryClick(cat)} className={`category-btn ${selectedCategory === cat ? "active" : ""}`}>{cat}</button>))}</div>
                        {selectedCategory && (<div className="search-container">
                            <SearchIcon className="search-icon" />
                            <input type="text" placeholder={`Search in ${selectedCategory}...`} value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} className="search-input"/>
                        </div>)}
                    </div>
                    {activeView === 'products' && (
                        isLoading ? <div style={{ display: 'flex', justifyContent: 'center', marginTop: '4rem' }}><LoadingSpinner /></div>
                        : error ? <div className="error-box"><h3 className="error-title">Connection Error</h3><p>{error}</p></div>
                        : (<>
                            {filteredProducts.length > 0 ? (
                                <div className="product-grid">
                                    {filteredProducts.slice(0, visibleCount).map((product, index) => {
                                        const avgScore = (product.environmental_impact_score + product.labor_rights_score + product.animal_welfare_score + product.corporate_governance_score) / 4 * 10;
                                        return (
                                            <div key={product.product_id} ref={index === visibleCount - 1 ? lastProductElementRef : null} onClick={() => handleProductClick(product)} className="product-card glass-effect">
                                                <div className="product-card-info">
                                                    <h3 className="product-name">{product.product_name}</h3>
                                                    <p className="product-category">{product.category}</p>
                                                    <p className="product-price">₹{product.product_price ? parseFloat(product.product_price).toFixed(2) : '0.00'}</p>
                                                </div>
                                                <CircularProgressBar score={avgScore} />
                                            </div>
                                        )
                                    })}
                                </div>
                            ) : (<div className="empty-state glass-effect"><h3 className="empty-title">No Products Found</h3><p>Try adjusting your search or category filters.</p></div>)}
                            {visibleCount < filteredProducts.length && !isLoading && <div style={{ display: 'flex', justifyContent: 'center', marginTop: '2rem' }}><LoadingSpinner /></div>}
                        </>)
                    )}
                </>) : (<AboutPage />)}
            </main>
            
            <ProductModal
                product={selectedProduct}
                onClose={() => setSelectedProduct(null)}
                factualScores={factualScores}
                sentimentScores={sentimentScores}
                explanation={explanation}
                summary={summary}
                newReview={newReview}
                userSubmittedReviews={userSubmittedReviews}
                isModalLoading={isModalLoading}
                isSummaryLoading={isSummaryLoading}
                isReviewSubmitting={isReviewSubmitting}
                onGetSummary={handleGetSummary}
                onReviewChange={(e) => setNewReview(e.target.value)}
                onReviewSubmit={handleReviewSubmit}
            />

            <div className="chat-fab-container">
                {showTooltip && !isChatOpen && (<div className="chat-tooltip">Ask our AI Assistant!</div>)}
                <button onClick={() => { setIsChatOpen(!isChatOpen); setShowTooltip(false); }} className="chat-fab-button">{isChatOpen ? <CloseIcon style={{height: '2rem', width: '2rem'}}/> : <ChatIcon style={{height: '2rem', width: '2rem'}}/>}</button>
            </div>
            
            {isChatOpen && 
                <ChatWindow 
                    chatHistory={chatHistory}
                    isChatLoading={isChatLoading}
                    chatInput={chatInput}
                    onChatInputChange={(e) => setChatInput(e.target.value)}
                    onChatSubmit={handleChatSubmit}
                    chatBodyRef={chatBodyRef}
                />
            }
        </div>
    );
}

