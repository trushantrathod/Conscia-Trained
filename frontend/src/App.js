import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css'

// --- Helper & Icon Components ---
const LoadingSpinner = () => ( <svg style={{height: '2rem', width: '2rem', animation: 'spin 1s linear infinite'}} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle style={{opacity: 0.25}} cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path style={{opacity: 0.75}} fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>);
const SunIcon = ({ style }) => ( <svg style={style} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.25a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0112 2.25zM7.5 12a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0zM18.894 6.106a.75.75 0 010 1.06l-1.06 1.06a.75.75 0 01-1.06-1.06l1.06-1.06a.75.75 0 011.06 0zM5.106 18.894a.75.75 0 010-1.06l1.06-1.06a.75.75 0 011.06 1.06l-1.06 1.06a.75.75 0 01-1.06 0zM18.894 18.894a.75.75 0 01-1.06 0l-1.06-1.06a.75.75 0 011.06-1.06l1.06 1.06a.75.75 0 010 1.06zM21.75 12a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5a.75.75 0 01.75.75zM2.25 12a.75.75 0 01.75-.75h1.5a.75.75 0 010 1.5H3a.75.75 0 01-.75-.75zM6.106 5.106a.75.75 0 011.06 0l1.06 1.06a.75.75 0 01-1.06 1.06L6.106 6.166a.75.75 0 010-1.06zM12 18a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0112 18z"></path></svg>);
const MoonIcon = ({ style }) => ( <svg style={style} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M9.528 1.718a.75.75 0 01.162.819A8.97 8.97 0 009 6a9 9 0 009 9 8.97 8.97 0 003.463-.69.75.75 0 01.981.98 10.503 10.503 0 01-9.694 6.46c-5.799 0-10.5-4.701-10.5-10.5 0-3.51 1.72-6.636 4.43-8.441a.75.75 0 01.819.162z" clipRule="evenodd" /></svg>);
const ChatBubbleIcon = ({ style }) => ( <svg style={style} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M4.804 21.644A6.707 6.707 0 006 21.75a6.75 6.75 0 006.75-6.75v-2.5A6.75 6.75 0 006 5.75a6.707 6.707 0 00-1.196.094 7.5 7.5 0 00-4.661 6.06c0 1.63.533 3.16 1.44 4.357.907 1.197 2.129 2.14 3.521 2.783zM14.25 15v2.5a6.75 6.75 0 006.75 6.75c.217 0 .431-.01.644-.028a7.5 7.5 0 004.661-6.06c0-1.63-.533-3.16-1.44-4.357-.907-1.197-2.129-2.14-3.521-2.783A6.707 6.707 0 0018 5.75a6.75 6.75 0 00-6.75 6.75v2.5z" clipRule="evenodd" /></svg>);
const CloseIcon = ({ style }) => ( <svg style={style} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M5.47 5.47a.75.75 0 011.06 0L12 10.94l5.47-5.47a.75.75 0 111.06 1.06L13.06 12l5.47 5.47a.75.75 0 11-1.06 1.06L12 13.06l-5.47 5.47a.75.75 0 01-1.06-1.06L10.94 12 5.47 6.53a.75.75 0 010-1.06z" clipRule="evenodd" /></svg>);
const SearchIcon = ({ className }) => ( <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M10.5 3.75a6.75 6.75 0 100 13.5 6.75 6.75 0 000-13.5ZM2.25 10.5a8.25 8.25 0 1114.59 5.28l4.69 4.69a.75.75 0 11-1.06 1.06l-4.69-4.69A8.25 8.25 0 012.25 10.5z" clipRule="evenodd" /></svg>);
const HamburgerIcon = ({ style }) => (<svg style={style} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M3 6.75A.75.75 0 013.75 6h16.5a.75.75 0 010 1.5H3.75A.75.75 0 013 6.75ZM3 12a.75.75 0 01.75-.75h16.5a.75.75 0 010 1.5H3.75A.75.75 0 013 12Zm0 5.25a.75.75 0 01.75-.75h16.5a.75.75 0 010 1.5H3.75a.75.75 0 01-.75-.75z" clipRule="evenodd" /></svg>);


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
    const [isReviewSubmitting, setIsReviewSubmitting] = useState(false);
    const [activeView, setActiveView] = useState('home');
    const [isNavOpen, setIsNavOpen] = useState(false);
    const [showTooltip, setShowTooltip] = useState(true);
    const [theme, setTheme] = useState('dark'); // 'dark' or 'light'
    const observer = useRef();
    const chatBodyRef = useRef(null);
    const API_URL = 'http://localhost:5000/api';
    
    // --- Theme Switching Effect ---
    useEffect(() => {
        // Apply the theme to the <html> element
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    const toggleTheme = () => {
        setTheme(prevTheme => (prevTheme === 'dark' ? 'light' : 'dark'));
    };
    
    useEffect(() => {
        const timer = setTimeout(() => {
            setShowTooltip(false);
        }, 5000);
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
        if (selectedCategory && selectedCategory !== 'All') {
            result = products.filter(p => p.category === selectedCategory);
        }
        if (searchTerm) {
            result = result.filter(p => p.product_name.toLowerCase().includes(searchTerm.toLowerCase()));
        }
        setFilteredProducts(result);
        setVisibleCount(20);
    }, [selectedCategory, searchTerm, products]);
    
    const handleCategoryClick = (category) => {
        setSelectedCategory(category);
        setActiveView('products');
    };

    const handleNavClick = (view) => {
        if (activeView === view) {
            setIsNavOpen(false);
            return;
        }
        setActiveView(view);
        setIsNavOpen(false);
        if (view === 'home') {
            setSelectedCategory(null);
        }
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
        setExplanation(null); 
        setSummary('');
        setNewReview('');
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
            'environmental impact': selectedProduct['environmental impact'],
            'labor rights': selectedProduct['labor rights'],
            'animal welfare': selectedProduct['animal welfare'],
            'corporate governance': selectedProduct['corporate governance']
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

    const handleReviewSubmit = async (e) => {
        e.preventDefault();
        if (!newReview.trim()) return;

        setIsReviewSubmitting(true);
        try {
            const reviewResponse = await fetch(`${API_URL}/products/${selectedProduct.product_id}/reviews`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ review: newReview })
            });

            if (!reviewResponse.ok) throw new Error('Failed to submit review.');
            
            const updatedProduct = await reviewResponse.json();
            
            setProducts(products.map(p => p.product_id === updatedProduct.product_id ? updatedProduct : p));
            setSelectedProduct(updatedProduct); 
            
            const explainRes = await fetch(`${API_URL}/explain`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reviews: updatedProduct.reviews })
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
    
    const parseMarkdown = (text) => {
        if (!text) return { __html: '' };
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/^- (.*$)/gm, '<li style="margin-left: 1.25rem; list-style-type: disc;">$1</li>');
        text = text.replace(/(?:\r\n|\r|\n){2,}/g, '<br /><br />');
        return { __html: text };
    };

    const categories = ['All', ...new Set(products.map(p => p.category))];

    const Navbar = () => (
        <nav className="navbar">
            {activeView === 'about' && (
                <div className="navbar-brand" onClick={() => handleNavClick('home')}>
                    Conscia
                </div>
            )}
            
            <button 
                className="navbar-btn theme-toggle-btn" 
                onClick={toggleTheme}
                aria-label={theme === 'dark' ? 'Activate Light Mode' : 'Activate Dark Mode'}
            >
                {theme === 'dark' ? (
                    <SunIcon style={{height: '1.5rem', width: '1.5rem'}} />
                ) : (
                    <MoonIcon style={{height: '1.5rem', width: '1.5rem'}} />
                )}
            </button>
            
            <div className="nav-menu-container">
                <button className="navbar-btn hamburger-btn" onClick={() => setIsNavOpen(!isNavOpen)}>
                    <HamburgerIcon style={{height: '1.75rem', width: '1.75rem'}} />
                </button>
                {isNavOpen && (
                    <div className="nav-menu glass-effect">
                        <button onClick={() => handleNavClick('home')} className="nav-menu-item">
                            Home
                        </button>
                        <button onClick={() => handleNavClick('about')} className="nav-menu-item">
                            About
                        </button>
                    </div>
                )}
            </div>
        </nav>
    );

    const HighlightedReview = ({ text, explanation }) => {
        if (!text) return <div style={{ color: 'var(--color-text-secondary)' }}>No review text available.</div>;

        const colorMap = {
            'environmental impact': 'rgba(52, 211, 153, 0.3)',
            'labor rights': 'rgba(96, 165, 250, 0.3)',
            'animal welfare': 'rgba(250, 204, 21, 0.3)',
            'corporate governance': 'rgba(192, 132, 252, 0.3)'
        };
        const wordMap = new Map();

        if (explanation) {
            Object.entries(explanation).forEach(([category, catWords]) => {
                if (Array.isArray(catWords)) {
                    catWords.forEach(word => wordMap.set(word.toLowerCase(), colorMap[category]));
                }
            });
        }

        const individualReviews = text.split(' | ').filter(review => review.trim() !== '');

        const renderSingleReview = (reviewText) => {
            const words = reviewText.split(/(\s+)/);
            return words.map((word, wordIndex) => {
                const cleanWord = word.replace(/[.,!?]/g, '').toLowerCase();
                const bgColor = wordMap.get(cleanWord);
                return bgColor ?
                    <span key={wordIndex} style={{ padding: '0 0.25rem', borderRadius: '0.375rem', backgroundColor: bgColor }}>{word}</span> :
                    <span key={wordIndex}>{word}</span>;
            });
        };

        return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {individualReviews.map((review, index) => (
                    <p key={index} style={{ color: 'var(--color-text-secondary)', lineHeight: 1.6, margin: 0 }}>
                        {renderSingleReview(review)}
                    </p>
                ))}
            </div>
        );
    };


    const AboutPage = React.memo(() => (
        <div className="about-page glass-effect">
            <h2>About Conscia 💡</h2>
            <h3>Our Mission</h3>
            <p>
                Our mission is to empower consumers with the information they need to shop according to their values. We analyze product reviews and data across four key pillars: Environmental Impact, Labor Rights, Animal Welfare, and Corporate Governance.
            </p>
            <h3>How It Works</h3>
            <p>
                Conscia uses a custom-trained neural network to analyze thousands of user reviews, identifying keywords and sentiment related to our ethical pillars. This analysis generates a score for each category, allowing you to see at a glance how a product aligns with ethical standards.
            </p>
        </div>
    ));

    const renderModalContent = () => {
        if (!selectedProduct) return null;
        
        const ethicalPillars = [
            { label: 'Environmental', key: 'environmental impact', colorClass: 'bar-env' },
            { label: 'Labor Rights', key: 'labor rights', colorClass: 'bar-labor' },
            { label: 'Animal Welfare', key: 'animal welfare', colorClass: 'bar-animal' },
            { label: 'Governance', key: 'corporate governance', colorClass: 'bar-gov' },
        ];

        const sentimentScore = { label: 'Public Sentiment', key: 'public_sentiment_score', colorClass: 'bar-senti' };

        return (
            <>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem'}}>
                    <div>
                        <h2 style={{fontSize: '1.875rem', fontWeight: 700, color: 'var(--color-text-primary)', margin: 0}}>{selectedProduct.product_name}</h2>
                        <p style={{fontSize: '1.25rem', fontWeight: 600, color: 'var(--color-accent)', margin: '0.25rem 0 0 0'}}>${selectedProduct.product_price ? parseFloat(selectedProduct.product_price).toFixed(2) : '0.00'}</p>
                    </div>
                    <button onClick={() => setSelectedProduct(null)} style={{background: 'none', border: 'none', cursor: 'pointer', padding: '0.5rem'}}>
                        <CloseIcon style={{height: '1.5rem', width: '1.5rem', color: '#94a3b8'}} />
                    </button>
                </div>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(1, 1fr)', gap: '2rem'}} id="modal-grid">
                    <div style={{display: 'flex', flexDirection: 'column', gap: '1rem'}}>
                        {/* Ethical Pillars Section */}
                        <div>
                            <h3 style={{fontSize: '1.25rem', fontWeight: 600, borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem', margin: '0 0 1rem 0'}}>Ethical Scores</h3>
                            <div style={{display: 'flex', flexDirection: 'column', gap: '0.75rem'}}>
                                {ethicalPillars.map(item => (
                                    <div key={item.key}>
                                        <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem'}}>
                                            <span style={{fontWeight: 500, color: 'var(--color-text-primary)'}}>{item.label}</span>
                                            <span style={{fontWeight: 700, color: 'var(--color-text-primary)'}}>{(selectedProduct[item.key] || 0).toFixed(0)} / 100</span>
                                        </div>
                                        <div className="card-score-bar-bg">
                                            <div className={`card-score-bar-fill ${item.colorClass}`} style={{ width: `${selectedProduct[item.key] || 0}%` }}></div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Public Sentiment Section */}
                        <div style={{marginTop: '1.5rem'}}>
                             <h3 style={{fontSize: '1.25rem', fontWeight: 600, borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem', margin: '0 0 1rem 0'}}>Public Sentiment</h3>
                             <div>
                                <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem'}}>
                                    <span style={{fontWeight: 500, color: 'var(--color-text-primary)'}}>{sentimentScore.label}</span>
                                    <span style={{fontWeight: 700, color: 'var(--color-text-primary)'}}>{(selectedProduct[sentimentScore.key] || 0).toFixed(0)} / 100</span>
                                </div>
                                <div className="card-score-bar-bg">
                                    <div className={`card-score-bar-fill ${sentimentScore.colorClass}`} style={{ width: `${selectedProduct[sentimentScore.key] || 0}%` }}></div>
                                </div>
                            </div>
                        </div>

                         <div style={{paddingTop: '1rem'}}>
                            <button onClick={handleGetSummary} disabled={isSummaryLoading} className="review-submit-btn">
                                {isSummaryLoading ? 'Generating...' : 'Get AI Ethical Snapshot'}
                            </button>
                            {summary && <div style={{marginTop: '1rem', padding: '1rem', backgroundColor: 'var(--color-background-light)', borderRadius: '8px', color: 'var(--color-text-secondary)', fontSize: '0.875rem', lineHeight: 1.6}}>{summary}</div>}
                        </div>
                    </div>
                    <div style={{display: 'flex', flexDirection: 'column', gap: '1rem'}}>
                         <h3 style={{fontSize: '1.25rem', fontWeight: 600, borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem', margin:0}}>Review Analysis (XAI)</h3>
                         <div style={{padding: '1rem', backgroundColor: 'var(--color-input-bg)', borderRadius: '8px', maxHeight: '20rem', overflowY: 'auto'}}>
                            {isModalLoading ? <div style={{display: 'flex', justifyContent: 'center'}}><LoadingSpinner /></div> : <HighlightedReview text={selectedProduct.reviews} explanation={explanation} />}
                        </div>
                        <p style={{fontSize: '0.75rem', color: 'var(--color-text-secondary)', textAlign: 'center', margin: 0}}>Words are highlighted based on their influence on the AI's scores.</p>
                        
                        <form onSubmit={handleReviewSubmit} style={{ paddingTop: '1rem' }}>
                            <h3 className="review-form-title">Add Your Review ✍️</h3>
                            <textarea
                                className="review-textarea"
                                value={newReview}
                                onChange={(e) => setNewReview(e.target.value)}
                                placeholder="Share your thoughts on this product..."
                            />
                            <button type="submit" disabled={isReviewSubmitting} className="review-submit-btn">
                                {isReviewSubmitting ? 'Submitting...' : 'Submit Review & Update Scores'}
                            </button>
                        </form>
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
            <header className="chat-header">
                <h3>Conscia Assistant 🤖</h3>
            </header>
            <div ref={chatBodyRef} className="chat-body">
                {chatHistory.length === 0 && (
                    <div className="chat-empty-state">
                        <p>Ask me to find products, compare items, or get brand insights!</p>
                    </div>
                )}
                {chatHistory.map((msg, index) => (
                    <div key={index} className={`chat-message ${msg.role}`}>
                        <div className="chat-bubble" dangerouslySetInnerHTML={parseMarkdown(msg.parts[0].text)} />
                    </div>
                ))}
                {isChatLoading && (
                    <div className="chat-message model">
                        <div className="chat-bubble">
                            <div className="typing-indicator">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
            <form onSubmit={handleChatSubmit} className="chat-input-form">
                <input 
                    type="text" 
                    value={chatInput} 
                    onChange={(e) => setChatInput(e.target.value)} 
                    placeholder="Ask anything..." 
                    className="chat-input"
                />
            </form>
        </div>
    );

    return (
        <div className="main-container">
            <Navbar />
            <header className="app-header">
                {activeView !== 'about' && (
                    <>
                        <h1 className="main-title">Conscia</h1>
                        <p className="app-subtitle">Your AI-Powered Ethical Shopping Companion. 🤖 Uncover the story behind every product.</p>
                    </>
                )}
            </header>

            <main>
                {activeView === 'home' || activeView === 'products' ? (
                    <>
                        <div className="controls-container glass-effect">
                            <h2 className="controls-title">Explore Products 🛍️</h2>
                            <div className="category-buttons">
                                {categories.map((cat) => (
                                <button
                                    key={cat}
                                    onClick={() => handleCategoryClick(cat)}
                                    className={`category-btn ${selectedCategory === cat ? "active" : ""}`}
                                >
                                    {cat}
                                </button>
                                ))}
                            </div>

                            {selectedCategory && (
                                <div className="search-container">
                                    <SearchIcon className="search-icon" />
                                    <input
                                        type="text"
                                        placeholder={`Search in ${selectedCategory}...`}
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="search-input"
                                    />
                                </div>
                            )}
                        </div>

                        {activeView === 'products' && (
                            isLoading ? <div style={{ display: 'flex', justifyContent: 'center', marginTop: '4rem' }}><LoadingSpinner /></div>
                            : error ? <div className="error-box glass-effect"><h3 className="error-title">Connection Error 🔌</h3><p>{error}</p></div>
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
                                                    <div>
                                                        <h3 className="product-name">{product.product_name}</h3>
                                                        <p className="product-category">{product.category}</p>
                                                        <p className="product-price">
                                                            ${product.product_price ? parseFloat(product.product_price).toFixed(2) : '0.00'}
                                                        </p>
                                                    </div>
                                                    <div className="card-scores-container">
                                                        <div className="card-score-bar-wrapper">
                                                            <span className="card-score-label">Environment</span>
                                                            <div className="card-score-bar-bg">
                                                                <div className="card-score-bar-fill bar-env" style={{ width: `${product['environmental impact'] || 0}%` }}/>
                                                            </div>
                                                            <span className="card-score-value">{Math.round(product['environmental impact'] || 0)}</span>
                                                        </div>
                                                        <div className="card-score-bar-wrapper">
                                                            <span className="card-score-label">Labor</span>
                                                            <div className="card-score-bar-bg">
                                                                <div className="card-score-bar-fill bar-labor" style={{ width: `${product['labor rights'] || 0}%` }}/>
                                                            </div>
                                                            <span className="card-score-value">{Math.round(product['labor rights'] || 0)}</span>
                                                        </div>
                                                        <div className="card-score-bar-wrapper">
                                                            <span className="card-score-label">Animal Welfare</span>
                                                            <div className="card-score-bar-bg">
                                                                <div className="card-score-bar-fill bar-animal" style={{ width: `${product['animal welfare'] || 0}%` }}/>
                                                            </div>
                                                            <span className="card-score-value">{Math.round(product['animal welfare'] || 0)}</span>
                                                        </div>
                                                        <div className="card-score-bar-wrapper">
                                                            <span className="card-score-label">Governance</span>
                                                            <div className="card-score-bar-bg">
                                                                <div className="card-score-bar-fill bar-gov" style={{ width: `${product['corporate governance'] || 0}%` }}/>
                                                            </div>
                                                            <span className="card-score-value">{Math.round(product['corporate governance'] || 0)}</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <div className="empty-state glass-effect"><h3 className="empty-title">No Products Found 😔</h3><p>Try adjusting your search or category filters.</p></div>
                                    )}
                                    {visibleCount < filteredProducts.length && !isLoading && <div style={{ display: 'flex', justifyContent: 'center', marginTop: '2rem' }}><LoadingSpinner /></div>}
                                </>
                            )
                        )}
                    </>
                ) : (
                    <AboutPage />
                )}
            </main>

            {selectedProduct && (
                <div className="modal-overlay" onClick={() => setSelectedProduct(null)}>
                    <div className="modal-content glass-effect" onClick={(e) => e.stopPropagation()}>
                        {renderModalContent()}
                    </div>
                </div>
            )}
            
            <div className="chat-fab-container">
                {showTooltip && !isChatOpen && (
                    <div className="chat-tooltip">
                        Ask our AI Assistant!
                    </div>
                )}
                <button onClick={() => { setIsChatOpen(!isChatOpen); setShowTooltip(false); }} className="chat-fab-button">
                    {isChatOpen ? <CloseIcon style={{height: '2rem', width: '2rem'}}/> : <ChatBubbleIcon style={{height: '2rem', width: '2rem'}}/>}
                </button>
            </div>
            
            {isChatOpen && renderChatWindow()}
        </div>
    );
}