"""
Customer Support AI Agent - Streamlit Web Application
Main application file for deployment on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta
import sqlite3
import pickle
from pathlib import Path
import hashlib
import base64

# Set page configuration
st.set_page_config(
    page_title="Customer Support AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    .user-message {
        background-color: #DBEAFE;
        margin-left: 2rem;
    }
    .ai-message {
        background-color: #F3F4F6;
        margin-right: 2rem;
    }
    .confidence-bar {
        height: 10px;
        background: linear-gradient(90deg, #EF4444 0%, #F59E0B 50%, #10B981 100%);
        border-radius: 5px;
        margin: 5px 0;
    }
    .ticket-card {
        border: 1px solid #D1D5DB;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
        background-color: white;
    }
    .ticket-open { border-left: 4px solid #EF4444; }
    .ticket-in-progress { border-left: 4px solid #F59E0B; }
    .ticket-resolved { border-left: 4px solid #10B981; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        margin: 10px;
    }
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = []
    if 'tickets' not in st.session_state:
        st.session_state.tickets = []
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {
            'total_queries': 0,
            'resolved': 0,
            'escalated': 0,
            'categories': {},
            'languages': {},
            'satisfaction': {'thumbs_up': 0, 'thumbs_down': 0}
        }
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Chat"
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False

# Database functions
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('customer_support.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            query_text TEXT,
            response_text TEXT,
            confidence REAL,
            ticket_created BOOLEAN,
            ticket_id TEXT,
            category TEXT,
            language TEXT,
            channel TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE,
            query_text TEXT,
            customer_contact TEXT,
            assigned_to TEXT,
            status TEXT,
            priority TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at DATETIME
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            content TEXT,
            category TEXT,
            embedding BLOB
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# AI Model Classes
class KnowledgeBase:
    """Simple knowledge base management"""
    def __init__(self):
        self.chunks = []
        
    def add_document(self, text, source="manual"):
        """Add document to knowledge base"""
        chunks = self._chunk_text(text)
        for chunk in chunks:
            self.chunks.append({
                'source': source,
                'content': chunk,
                'category': self._categorize_text(chunk)
            })
        return len(chunks)
    
    def _chunk_text(self, text, chunk_size=500):
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _categorize_text(self, text):
        """Categorize text content"""
        text_lower = text.lower()
        categories = {
            'billing': ['payment', 'bill', 'invoice', 'refund', 'price'],
            'technical': ['error', 'bug', 'technical', 'issue', 'problem'],
            'product': ['product', 'feature', 'specification', 'model'],
            'warranty': ['warranty', 'guarantee', 'repair'],
            'service': ['service', 'install', 'setup', 'maintenance'],
            'general': ['contact', 'location', 'hours', 'information']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def search(self, query, top_k=3):
        """Simple search in knowledge base"""
        query_lower = query.lower()
        results = []
        
        for chunk in self.chunks:
            score = self._calculate_similarity(query_lower, chunk['content'].lower())
            if score > 0.1:  # Minimum similarity threshold
                results.append({
                    'content': chunk['content'],
                    'score': score,
                    'category': chunk['category'],
                    'source': chunk['source']
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _calculate_similarity(self, text1, text2):
        """Calculate simple text similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

class AIResponseGenerator:
    """Generate AI responses using knowledge base"""
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        
    def generate_response(self, query, language='en'):
        """Generate response for a query"""
        # Search knowledge base
        results = self.knowledge_base.search(query)
        
        if results:
            # Use the best matching result
            best_result = results[0]
            confidence = best_result['score']
            
            # Generate response based on knowledge
            response = self._format_response(best_result['content'], query, language)
            
            # Check if ticket should be created
            ticket_created = confidence < 0.3  # Low confidence threshold
            
            if ticket_created:
                ticket_id = self._generate_ticket_id()
                response += f"\n\nI've created a support ticket for you. Ticket ID: {ticket_id}"
            else:
                ticket_id = None
        else:
            # No relevant knowledge found
            response = self._get_fallback_response(language)
            confidence = 0.0
            ticket_created = True
            ticket_id = self._generate_ticket_id()
            response += f"\n\nI've created a support ticket for you. Ticket ID: {ticket_id}"
        
        return {
            'response': response,
            'confidence': min(confidence * 100, 100),  # Convert to percentage
            'ticket_created': ticket_created,
            'ticket_id': ticket_id,
            'sources': results
        }
    
    def _format_response(self, knowledge, query, language):
        """Format knowledge into a response"""
        # Simple formatting - in production, use LLM
        responses = {
            'en': f"Based on our knowledge base: {knowledge}",
            'hi': f"हमारे ज्ञान आधार के अनुसार: {knowledge}",
            'mr': f"आमच्या ज्ञान आधारानुसार: {knowledge}"
        }
        return responses.get(language, responses['en'])
    
    def _get_fallback_response(self, language):
        """Get fallback response when no knowledge found"""
        responses = {
            'en': "I couldn't find specific information about that in our knowledge base.",
            'hi': "मुझे हमारे ज्ञान आधार में इसके बारे में विशिष्ट जानकारी नहीं मिली।",
            'mr': "आमच्या ज्ञान आधारात याबद्दल विशिष्ट माहिती सापडली नाही."
        }
        return responses.get(language, responses['en'])
    
    def _generate_ticket_id(self):
        """Generate unique ticket ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"TICKET-{timestamp}"

class TicketSystem:
    """Support ticket management"""
    def __init__(self):
        self.tickets = []
    
    def create_ticket(self, query, customer_info=None, priority='medium'):
        """Create a new support ticket"""
        ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        ticket = {
            'ticket_id': ticket_id,
            'query': query,
            'customer_info': customer_info or {},
            'priority': priority,
            'status': 'open',
            'assigned_to': None,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'notes': []
        }
        
        self.tickets.append(ticket)
        return ticket
    
    def update_ticket(self, ticket_id, status=None, assigned_to=None, note=None):
        """Update ticket status and information"""
        for ticket in self.tickets:
            if ticket['ticket_id'] == ticket_id:
                if status:
                    ticket['status'] = status
                if assigned_to:
                    ticket['assigned_to'] = assigned_to
                if note:
                    ticket['notes'].append({
                        'timestamp': datetime.now(),
                        'note': note
                    })
                ticket['updated_at'] = datetime.now()
                return ticket
        return None
    
    def get_tickets(self, status=None):
        """Get tickets, optionally filtered by status"""
        if status:
            return [t for t in self.tickets if t['status'] == status]
        return self.tickets

# Authentication Functions
def hash_password(password):
    """Hash password for storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    """Authenticate user"""
    # For demo purposes - in production, use proper authentication
    users = {
        'admin': hash_password('admin123'),
        'support': hash_password('support123'),
        'client': hash_password('client123')
    }
    
    if username in users and users[username] == hash_password(password):
        return True
    return False

def login_form():
    """Display login form"""
    st.sidebar.markdown("## 🔐 Login")
    
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if authenticate_user(username, password):
                st.session_state.user_authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

# UI Components
def sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/customer-support.png", width=100)
        st.markdown("# 🤖 AI Support Agent")
        
        if not st.session_state.user_authenticated:
            login_form()
            return
        
        st.markdown(f"### 👋 Welcome, {st.session_state.username}")
        
        # Navigation
        st.markdown("## 📋 Navigation")
        tabs = {
            "🏠 Dashboard": "Dashboard",
            "💬 Chat Support": "Chat",
            "📚 Knowledge Base": "Knowledge",
            "🎫 Ticket Management": "Tickets",
            "📊 Analytics": "Analytics",
            "⚙️ Settings": "Settings"
        }
        
        for icon, tab_name in tabs.items():
            if st.button(icon, key=f"nav_{tab_name}"):
                st.session_state.current_tab = tab_name
                st.rerun()
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", st.session_state.analytics['total_queries'])
            st.metric("Tickets Open", len([t for t in st.session_state.tickets if t['status'] == 'open']))
        with col2:
            st.metric("Resolved", st.session_state.analytics['resolved'])
            st.metric("Satisfaction", 
                     f"{st.session_state.analytics['satisfaction']['thumbs_up']} 👍")
        
        st.markdown("---")
        
        # Logout
        if st.button("🚪 Logout"):
            st.session_state.user_authenticated = False
            st.rerun()

def dashboard_tab():
    """Dashboard tab content"""
    st.markdown("# 📊 Dashboard")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", st.session_state.analytics['total_queries'], 
                 delta="+12 today")
    
    with col2:
        st.metric("Auto-Resolved", st.session_state.analytics['resolved'],
                 delta=f"+{st.session_state.analytics['resolved'] - st.session_state.analytics['escalated']}")
    
    with col3:
        st.metric("Escalated Tickets", st.session_state.analytics['escalated'])
    
    with col4:
        satisfaction_rate = 0
        total_feedback = (st.session_state.analytics['satisfaction']['thumbs_up'] + 
                         st.session_state.analytics['satisfaction']['thumbs_down'])
        if total_feedback > 0:
            satisfaction_rate = (st.session_state.analytics['satisfaction']['thumbs_up'] / total_feedback) * 100
        st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Query Categories")
        if st.session_state.analytics['categories']:
            categories_df = pd.DataFrame({
                'Category': list(st.session_state.analytics['categories'].keys()),
                'Count': list(st.session_state.analytics['categories'].values())
            })
            st.bar_chart(categories_df.set_index('Category'))
        else:
            st.info("No category data yet")
    
    with col2:
        st.markdown("### 🌐 Language Distribution")
        if st.session_state.analytics['languages']:
            languages_df = pd.DataFrame({
                'Language': list(st.session_state.analytics['languages'].keys()),
                'Count': list(st.session_state.analytics['languages'].values())
            })
            st.bar_chart(languages_df.set_index('Language'))
        else:
            st.info("No language data yet")
    
    # Recent Activity
    st.markdown("### 📝 Recent Activity")
    if st.session_state.messages:
        recent_activity = []
        for msg in st.session_state.messages[-5:]:  # Last 5 messages
            recent_activity.append({
                'Time': msg.get('timestamp', ''),
                'Query': msg.get('query', '')[:50] + '...',
                'Response': msg.get('response', '')[:50] + '...',
                'Confidence': f"{msg.get('confidence', 0):.0f}%"
            })
        
        if recent_activity:
            st.dataframe(pd.DataFrame(recent_activity))
    else:
        st.info("No recent activity")

def chat_tab():
    """Chat support tab content"""
    st.markdown("# 💬 Chat Support")
    
    # Language selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        language = st.selectbox("🌐 Language", ["English", "Hindi", "Marathi"], key="chat_language")
    with col3:
        business_unit = st.selectbox("🏢 Business Unit", 
                                   ["IT Services", "Real Estate", "E-commerce", "General"], 
                                   key="business_unit")
    
    # Chat container
    chat_container = st.container()
    
    # Initialize AI components
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = KnowledgeBase()
    
    if 'ai_generator' not in st.session_state:
        st.session_state.ai_generator = AIResponseGenerator(st.session_state.knowledge_base)
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong> {message['content']}
                    <small style="color: #6B7280; float: right;">{message.get('timestamp', '')}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = message.get('confidence', 0)
                confidence_color = "🔴" if confidence < 30 else "🟡" if confidence < 70 else "🟢"
                
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>🤖 AI Assistant:</strong> {message['content']}
                    <div class="confidence-bar" style="width: {confidence}%"></div>
                    <small style="color: #6B7280;">
                        {confidence_color} Confidence: {confidence:.0f}% | 
                        Language: {message.get('language', 'English')} | 
                        {message.get('timestamp', '')}
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                if message.get('ticket_created'):
                    st.success(f"🎫 Ticket Created: {message.get('ticket_id')}")
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Type your message...", key="chat_input", 
                                      placeholder="How can I help you today?")
        with col2:
            submit = st.form_submit_button("Send", use_container_width=True)
        
        if submit and user_input:
            # Add user message
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.messages.append({
                'role': 'user',
                'content': user_input,
                'timestamp': timestamp
            })
            
            # Generate AI response
            lang_map = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
            response = st.session_state.ai_generator.generate_response(
                user_input, 
                language=lang_map.get(language, "en")
            )
            
            # Add AI response
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response['response'],
                'confidence': response['confidence'],
                'ticket_created': response['ticket_created'],
                'ticket_id': response['ticket_id'],
                'language': language,
                'timestamp': datetime.now().strftime("%H:%M")
            })
            
            # Update analytics
            st.session_state.analytics['total_queries'] += 1
            if response['confidence'] > 50:
                st.session_state.analytics['resolved'] += 1
            else:
                st.session_state.analytics['escalated'] += 1
                
            # Create ticket if needed
            if response['ticket_created']:
                ticket = {
                    'ticket_id': response['ticket_id'],
                    'query': user_input,
                    'status': 'open',
                    'created_at': datetime.now(),
                    'priority': 'medium',
                    'assigned_to': None
                }
                st.session_state.tickets.append(ticket)
            
            st.rerun()

def knowledge_base_tab():
    """Knowledge base management tab"""
    st.markdown("# 📚 Knowledge Base")
    
    tab1, tab2, tab3 = st.tabs(["📄 Upload Documents", "📝 Manual Entry", "🔍 Search & Manage"])
    
    with tab1:
        st.markdown("### Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx', 'csv'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, DOCX, or CSV files"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file not in st.session_state.uploaded_files:
                    try:
                        # Read file content based on type
                        if uploaded_file.type == "application/pdf":
                            import PyPDF2
                            pdf_reader = PyPDF2.PdfReader(uploaded_file)
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text()
                        elif uploaded_file.type == "text/plain":
                            text = uploaded_file.read().decode("utf-8")
                        else:
                            text = uploaded_file.read().decode("utf-8", errors='ignore')
                        
                        # Add to knowledge base
                        if 'knowledge_base' not in st.session_state:
                            st.session_state.knowledge_base = KnowledgeBase()
                        
                        chunks_added = st.session_state.knowledge_base.add_document(
                            text, 
                            source=uploaded_file.name
                        )
                        
                        st.session_state.uploaded_files.append(uploaded_file)
                        st.success(f"✅ Added {chunks_added} chunks from {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.markdown("### Uploaded Files")
            for file in st.session_state.uploaded_files:
                st.text(f"📄 {file.name}")
    
    with tab2:
        st.markdown("### Add Knowledge Manually")
        
        with st.form("manual_entry_form"):
            category = st.selectbox("Category", 
                                  ["General", "Billing", "Technical", "Product", "Warranty", "Service"])
            
            question = st.text_input("Question/Title")
            answer = st.text_area("Answer/Content", height=200)
            
            submit = st.form_submit_button("Add to Knowledge Base")
            
            if submit and question and answer:
                content = f"Q: {question}\nA: {answer}"
                
                if 'knowledge_base' not in st.session_state:
                    st.session_state.knowledge_base = KnowledgeBase()
                
                chunks_added = st.session_state.knowledge_base.add_document(
                    content,
                    source="manual_entry"
                )
                
                st.success(f"✅ Added knowledge entry with {chunks_added} chunks")
    
    with tab3:
        st.markdown("### Search Knowledge Base")
        
        search_query = st.text_input("Search for information")
        
        if search_query and 'knowledge_base' in st.session_state:
            results = st.session_state.knowledge_base.search(search_query, top_k=5)
            
            if results:
                st.markdown(f"Found {len(results)} relevant results:")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} (Score: {result['score']:.2f})"):
                        st.markdown(f"**Source:** {result['source']}")
                        st.markdown(f"**Category:** {result['category']}")
                        st.markdown(f"**Content:** {result['content']}")
            else:
                st.info("No results found")
        
        # Knowledge base stats
        if 'knowledge_base' in st.session_state:
            st.markdown("### Knowledge Base Stats")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", len(st.session_state.knowledge_base.chunks))
            with col2:
                categories = {}
                for chunk in st.session_state.knowledge_base.chunks:
                    cat = chunk.get('category', 'unknown')
                    categories[cat] = categories.get(cat, 0) + 1
                st.metric("Categories", len(categories))
            with col3:
                sources = set(chunk['source'] for chunk in st.session_state.knowledge_base.chunks)
                st.metric("Sources", len(sources))

def tickets_tab():
    """Ticket management tab"""
    st.markdown("# 🎫 Ticket Management")
    
    # Create new ticket
    with st.expander("➕ Create New Ticket", expanded=False):
        with st.form("new_ticket_form"):
            col1, col2 = st.columns(2)
            with col1:
                customer_name = st.text_input("Customer Name")
                customer_email = st.text_input("Customer Email")
            with col2:
                priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
                category = st.selectbox("Category", 
                                      ["Technical", "Billing", "General", "Complaint", "Service"])
            
            issue_description = st.text_area("Issue Description", height=100)
            
            submit = st.form_submit_button("Create Ticket")
            
            if submit and issue_description:
                ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                new_ticket = {
                    'ticket_id': ticket_id,
                    'customer_name': customer_name,
                    'customer_email': customer_email,
                    'priority': priority,
                    'category': category,
                    'description': issue_description,
                    'status': 'open',
                    'created_at': datetime.now(),
                    'assigned_to': None,
                    'notes': []
                }
                
                st.session_state.tickets.append(new_ticket)
                st.success(f"Ticket {ticket_id} created successfully!")
                st.rerun()
    
    # Filter tickets
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_status = st.selectbox("Filter by Status", 
                                   ["All", "open", "in_progress", "resolved", "closed"])
    with col2:
        filter_priority = st.selectbox("Filter by Priority", 
                                     ["All", "Low", "Medium", "High", "Critical"])
    with col3:
        filter_assigned = st.selectbox("Filter by Assignee", 
                                     ["All", "Unassigned", "Assigned to me"])
    
    # Display tickets
    filtered_tickets = st.session_state.tickets
    
    if filter_status != "All":
        filtered_tickets = [t for t in filtered_tickets if t['status'] == filter_status]
    
    if filter_priority != "All":
        filtered_tickets = [t for t in filtered_tickets if t.get('priority') == filter_priority]
    
    if filter_assigned == "Unassigned":
        filtered_tickets = [t for t in filtered_tickets if not t.get('assigned_to')]
    elif filter_assigned == "Assigned to me":
        filtered_tickets = [t for t in filtered_tickets if t.get('assigned_to') == st.session_state.username]
    
    if filtered_tickets:
        for ticket in filtered_tickets:
            status_color = {
                'open': 'red',
                'in_progress': 'orange',
                'resolved': 'green',
                'closed': 'gray'
            }.get(ticket['status'], 'gray')
            
            priority_color = {
                'Low': 'green',
                'Medium': 'blue',
                'High': 'orange',
                'Critical': 'red'
            }.get(ticket.get('priority', 'Medium'), 'blue')
            
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### {ticket['ticket_id']}")
                    st.markdown(f"**Description:** {ticket.get('description', 'No description')[:100]}...")
                    st.markdown(f"**Customer:** {ticket.get('customer_name', 'N/A')} ({ticket.get('customer_email', 'N/A')})")
                
                with col2:
                    st.markdown(f"**Status:** :{status_color}[{ticket['status'].replace('_', ' ').title()}]")
                    st.markdown(f"**Priority:** :{priority_color}[{ticket.get('priority', 'Medium')}]")
                
                with col3:
                    st.markdown(f"**Created:** {ticket.get('created_at').strftime('%Y-%m-%d')}")
                    st.markdown(f"**Assigned to:** {ticket.get('assigned_to', 'Unassigned')}")
                    
                    # Action buttons
                    if ticket['status'] == 'open':
                        if st.button(f"Take Ticket", key=f"take_{ticket['ticket_id']}"):
                            ticket['assigned_to'] = st.session_state.username
                            ticket['status'] = 'in_progress'
                            st.rerun()
                    
                    if ticket['status'] == 'in_progress' and ticket.get('assigned_to') == st.session_state.username:
                        if st.button(f"Resolve", key=f"resolve_{ticket['ticket_id']}"):
                            ticket['status'] = 'resolved'
                            st.rerun()
                
                st.markdown("---")
    else:
        st.info("No tickets found with the selected filters")

def analytics_tab():
    """Analytics and reporting tab"""
    st.markdown("# 📊 Analytics & Reports")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", 
                                 value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Key metrics
    st.markdown("### 📈 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", st.session_state.analytics['total_queries'])
    
    with col2:
        resolution_rate = 0
        if st.session_state.analytics['total_queries'] > 0:
            resolution_rate = (st.session_state.analytics['resolved'] / 
                             st.session_state.analytics['total_queries']) * 100
        st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
    
    with col3:
        avg_confidence = 0
        if st.session_state.messages:
            ai_messages = [m for m in st.session_state.messages if m['role'] == 'assistant']
            if ai_messages:
                avg_confidence = sum(m.get('confidence', 0) for m in ai_messages) / len(ai_messages)
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    with col4:
        satisfaction_rate = 0
        total_feedback = (st.session_state.analytics['satisfaction']['thumbs_up'] + 
                         st.session_state.analytics['satisfaction']['thumbs_down'])
        if total_feedback > 0:
            satisfaction_rate = (st.session_state.analytics['satisfaction']['thumbs_up'] / total_feedback) * 100
        st.metric("Satisfaction", f"{satisfaction_rate:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Query Categories")
        if st.session_state.analytics['categories']:
            categories_df = pd.DataFrame({
                'Category': list(st.session_state.analytics['categories'].keys()),
                'Count': list(st.session_state.analytics['categories'].values())
            })
            st.bar_chart(categories_df.set_index('Category'))
    
    with col2:
        st.markdown("### 🌐 Language Distribution")
        if st.session_state.analytics['languages']:
            languages_df = pd.DataFrame({
                'Language': list(st.session_state.analytics['languages'].keys()),
                'Count': list(st.session_state.analytics['languages'].values())
            })
            st.bar_chart(languages_df.set_index('Language'))
    
    # Export data
    st.markdown("### 📤 Export Data")
    
    if st.button("📥 Download Analytics Report"):
        # Create a simple report
        report = {
            'generated_at': datetime.now().isoformat(),
            'date_range': f"{start_date} to {end_date}",
            'analytics': st.session_state.analytics,
            'total_tickets': len(st.session_state.tickets),
            'open_tickets': len([t for t in st.session_state.tickets if t['status'] == 'open']),
            'recent_queries': st.session_state.messages[-10:] if st.session_state.messages else []
        }
        
        # Convert to JSON for download
        json_report = json.dumps(report, indent=2, default=str)
        
        st.download_button(
            label="Download JSON Report",
            data=json_report,
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

def settings_tab():
    """Settings tab"""
    st.markdown("# ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["AI Settings", "System Settings", "User Management"])
    
    with tab1:
        st.markdown("### AI Configuration")
        
        confidence_threshold = st.slider(
            "Confidence Threshold for Auto-Resolution",
            min_value=0,
            max_value=100,
            value=50,
            help="Queries with confidence below this threshold will be escalated"
        )
        
        max_response_time = st.slider(
            "Maximum Response Time (seconds)",
            min_value=1,
            max_value=10,
            value=2,
            help="Target maximum response time for AI"
        )
        
        supported_languages = st.multiselect(
            "Supported Languages",
            ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Bengali", "Gujarati"],
            default=["English", "Hindi", "Marathi"]
        )
        
        if st.button("Save AI Settings"):
            st.success("AI settings saved successfully!")
    
    with tab2:
        st.markdown("### System Configuration")
        
        business_units = st.text_area(
            "Business Units (one per line)",
            value="IT Services\nReal Estate\nE-commerce\nGeneral",
            height=100
        )
        
        notification_email = st.text_input("Notification Email")
        
        auto_ticket_assign = st.checkbox("Auto-assign tickets", value=True)
        
        if st.button("Save System Settings"):
            st.success("System settings saved!")
    
    with tab3:
        st.markdown("### User Management")
        
        if st.session_state.username == 'admin':
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Add New User")
                with st.form("add_user_form"):
                    new_username = st.text_input("Username")
                    new_password = st.text_input("Password", type="password")
                    new_role = st.selectbox("Role", ["admin", "support", "client"])
                    
                    if st.form_submit_button("Add User"):
                        st.success(f"User {new_username} added!")
            
            with col2:
                st.markdown("#### Current Users")
                users = [
                    {"username": "admin", "role": "admin"},
                    {"username": "support", "role": "support"},
                    {"username": "client", "role": "client"}
                ]
                
                for user in users:
                    st.text(f"👤 {user['username']} - {user['role']}")
        else:
            st.info("Only administrators can manage users")

# Main application
def main():
    """Main application function"""
    # Load CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Initialize database
    init_database()
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 Customer Support AI Agent</h1>", unsafe_allow_html=True)
    st.markdown("### Multi-channel AI Support System with FAQ, Ticketing & Analytics")
    
    # Check authentication
    if not st.session_state.user_authenticated:
        st.warning("🔒 Please login to access the system")
        login_form()
        return
    
    # Sidebar
    sidebar()
    
    # Main content based on selected tab
    if st.session_state.current_tab == "Dashboard":
        dashboard_tab()
    elif st.session_state.current_tab == "Chat":
        chat_tab()
    elif st.session_state.current_tab == "Knowledge":
        knowledge_base_tab()
    elif st.session_state.current_tab == "Tickets":
        tickets_tab()
    elif st.session_state.current_tab == "Analytics":
        analytics_tab()
    elif st.session_state.current_tab == "Settings":
        settings_tab()

# Run the application
if __name__ == "__main__":
    main()

