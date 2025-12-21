# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="VALORANT Agent",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# CUSTOM CSS - Valorant Theme
# ======================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0F1923 0%, #1a2634 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0F1923;
        border-right: 2px solid #FF4655;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ECE8E1 !important;
    }
    
    /* Valorant red accent */
    .valorant-red {
        color: #FF4655;
        font-weight: bold;
    }
    
    /* Cards */
    .stat-card {
        background: #1F2326;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #FF4655;
        margin: 10px 0;
    }
    
    /* Success card */
    .success-card {
        background: #1F2326;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #3DDC97;
        margin: 10px 0;
    }
    
    /* Warning card */
    .warning-card {
        background: #1F2326;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #FFD700;
        margin: 10px 0;
    }
    
    /* Big number stat */
    .big-stat {
        font-size: 48px;
        font-weight: bold;
        color: #FF4655;
    }
    
    /* Metric label */
    .metric-label {
        color: #7B8084;
        font-size: 14px;
        text-transform: uppercase;
    }
    
    /* Button styling */
    .stButton > button {
        background: #FF4655;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: #BD3944;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #ECE8E1;
    }
    
    /* Select box */
    .stSelectbox > label {
        color: #ECE8E1;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# LOAD DATA
# ======================================================================
@st.cache_data
def load_weapons():
    return pd.read_csv("weapons.csv")

@st.cache_data
def load_strategies():
    return pd.read_csv("valorant_strategies.csv")


# ======================================================================
# DECISION TREE MODEL FOR WEAPONS
# ======================================================================
@st.cache_resource
def train_weapon_model():
    """Train a Decision Tree to predict weapon category based on user preferences"""
    weapons = pd.read_csv("weapons.csv")
    
    # Create training data - simulate user profiles matched to weapons
    np.random.seed(42)
    training_data = []
    
    # For each weapon, create profiles that would prefer it
    for _, weapon in weapons.iterrows():
        if weapon['Weapon'] == 'Knife':
            continue
            
        # Generate 20 synthetic user profiles per weapon
        for _ in range(20):
            # Rank (0-8: Iron to Radiant)
            if weapon['Category'] in ['Rifle', 'Sniper']:
                rank = np.random.randint(4, 9)  # Higher ranks
            elif weapon['Category'] in ['SMG', 'Machine Gun']:
                rank = np.random.randint(0, 6)  # Lower-mid ranks
            else:
                rank = np.random.randint(0, 9)  # Any rank
                
            # Sensitivity (0=low, 1=med, 2=high)
            if weapon['Category'] == 'Sniper':
                sensitivity = np.random.choice([0, 1], p=[0.7, 0.3])
            elif weapon['Category'] == 'Shotgun':
                sensitivity = np.random.choice([1, 2], p=[0.4, 0.6])
            else:
                sensitivity = np.random.randint(0, 3)
                
            # Playstyle (0=aggressive, 1=passive, 2=balanced)
            if weapon['Category'] in ['SMG', 'Shotgun']:
                playstyle = np.random.choice([0, 2], p=[0.7, 0.3])
            elif weapon['Category'] in ['Sniper', 'Rifle']:
                playstyle = np.random.choice([1, 2], p=[0.6, 0.4])
            else:
                playstyle = np.random.randint(0, 3)
                
            # Preferred range (0=close, 1=medium, 2=long)
            if weapon['Category'] in ['Shotgun', 'SMG']:
                pref_range = np.random.choice([0, 1], p=[0.7, 0.3])
            elif weapon['Category'] == 'Sniper':
                pref_range = np.random.choice([1, 2], p=[0.3, 0.7])
            else:
                pref_range = np.random.randint(0, 3)
                
            # Aim style (0=flick, 1=tracking, 2=spray)
            if weapon['Category'] == 'Sniper':
                aim_style = np.random.choice([0, 1], p=[0.8, 0.2])
            elif weapon['Fire Mode'] == 'Full-Auto':
                aim_style = np.random.choice([1, 2], p=[0.4, 0.6])
            else:
                aim_style = np.random.randint(0, 3)
                
            # Headshot confidence (0=low, 1=med, 2=high)
            if weapon['Head Damage (0-30m)'] >= 150:
                hs_conf = np.random.choice([1, 2], p=[0.3, 0.7])
            else:
                hs_conf = np.random.randint(0, 3)
                
            # Budget (0=eco, 1=mid, 2=full)
            cost = weapon['Cost']
            if cost <= 1000:
                budget = np.random.choice([0, 1], p=[0.7, 0.3])
            elif cost <= 2500:
                budget = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
            else:
                budget = np.random.choice([1, 2], p=[0.3, 0.7])
                
            training_data.append([
                rank, sensitivity, playstyle, pref_range, 
                aim_style, hs_conf, budget, weapon['Weapon']
            ])
    
    # Create DataFrame
    df = pd.DataFrame(training_data, columns=[
        'Rank', 'Sensitivity', 'Playstyle', 'PreferredRange',
        'AimStyle', 'HeadshotConf', 'Budget', 'Weapon'
    ])
    
    # Encode target
    label_encoder = LabelEncoder()
    df['WeaponEncoded'] = label_encoder.fit_transform(df['Weapon'])
    
    # Features and target
    feature_names = ['Rank', 'Sensitivity', 'Playstyle', 'PreferredRange', 'AimStyle', 'HeadshotConf', 'Budget']
    X = df[feature_names]
    y = df['WeaponEncoded']
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = DecisionTreeClassifier(max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    
    accuracy = metrics.accuracy_score(y_test, clf.predict(X_test))
    
    return clf, label_encoder, feature_names, accuracy


# ======================================================================
# WEAPON RECOMMENDER
# ======================================================================
def weapon_recommender():
    st.markdown("# 🔫 WEAPON RECOMMENDER")
    st.markdown("*Find your perfect loadout based on your playstyle using Decision Tree AI*")
    st.divider()
    
    weapons = load_weapons()
    clf, label_encoder, feature_names, accuracy = train_weapon_model()
    
    # Show model accuracy
    st.markdown(f"""
    <div class="success-card">
        <span class="metric-label">🤖 Decision Tree Model</span>
        <p>Accuracy: <strong>{accuracy*100:.1f}%</strong> • Trained on simulated player profiles</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Your Profile")
        
        rank = st.selectbox(
            "🎖️ What's your current rank?",
            ['Iron', 'Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Ascendant', 'Immortal', 'Radiant'],
            index=3
        )
        
        sensitivity = st.select_slider(
            "🖱️ Mouse sensitivity style",
            options=['Low (arm aimer)', 'Medium (hybrid)', 'High (wrist aimer)'],
            value='Medium (hybrid)'
        )
        
        playstyle = st.radio(
            "⚔️ How do you like to play?",
            ['Aggressive (entry fragger)', 'Passive (angle holder)', 'Balanced (adaptive)'],
            horizontal=True
        )
        
    with col2:
        st.markdown("### 🎯 Your Aim")
        
        preferred_range = st.radio(
            "📏 Preferred engagement range?",
            ['Close (shotgun range)', 'Medium (site fights)', 'Long (cross-map)'],
            horizontal=True
        )
        
        aim_style = st.radio(
            "🎮 When an enemy appears, you...",
            ['Flick to their head', 'Track their movement', 'Spray and control'],
            horizontal=True
        )
        
        headshot_conf = st.select_slider(
            "🎯 Headshot confidence",
            options=['Low (body aimer)', 'Medium (situational)', 'High (headshot machine)'],
            value='Medium (situational)'
        )
        
        budget = st.radio(
            "💰 Economy style?",
            ['Eco saver', 'Force buyer', 'Full buy only'],
            horizontal=True
        )
    
    st.divider()
    
    if st.button("🔍 FIND MY WEAPONS", use_container_width=True):
        # Convert inputs to numeric values for Decision Tree
        rank_map = {'Iron': 0, 'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4, 
                    'Diamond': 5, 'Ascendant': 6, 'Immortal': 7, 'Radiant': 8}
        sens_map = {'Low (arm aimer)': 0, 'Medium (hybrid)': 1, 'High (wrist aimer)': 2}
        play_map = {'Aggressive (entry fragger)': 0, 'Passive (angle holder)': 1, 'Balanced (adaptive)': 2}
        range_map = {'Close (shotgun range)': 0, 'Medium (site fights)': 1, 'Long (cross-map)': 2}
        aim_map = {'Flick to their head': 0, 'Track their movement': 1, 'Spray and control': 2}
        hs_map = {'Low (body aimer)': 0, 'Medium (situational)': 1, 'High (headshot machine)': 2}
        budget_map = {'Eco saver': 0, 'Force buyer': 1, 'Full buy only': 2}
        
        # Create feature vector
        user_features = np.array([[
            rank_map[rank],
            sens_map[sensitivity],
            play_map[playstyle],
            range_map[preferred_range],
            aim_map[aim_style],
            hs_map[headshot_conf],
            budget_map[budget]
        ]])
        
        # Predict with Decision Tree
        prediction = clf.predict(user_features)[0]
        predicted_weapon = label_encoder.inverse_transform([prediction])[0]
        
        # Get probability scores for all weapons
        proba = clf.predict_proba(user_features)[0]
        weapon_scores = dict(zip(label_encoder.classes_, proba))
        sorted_weapons = sorted(weapon_scores.items(), key=lambda x: x[1], reverse=True)
        
        main_name = predicted_weapon
        main_weapon = weapons[weapons['Weapon'] == main_name].iloc[0]
        main_cat = main_weapon['Category']
        
        # Find secondary from different category
        for name, score in sorted_weapons[1:]:
            if name not in weapons['Weapon'].values:
                continue
            w = weapons[weapons['Weapon'] == name].iloc[0]
            if w['Category'] != main_cat:
                secondary_weapon = w
                break
        else:
            # Fallback
            for name, score in sorted_weapons[1:]:
                if name in weapons['Weapon'].values:
                    secondary_weapon = weapons[weapons['Weapon'] == name].iloc[0]
                    break
        
        # Display results
        st.markdown("## 🎯 YOUR PERFECT LOADOUT")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <span class="metric-label">PRIMARY WEAPON</span>
                <h1 style="color: #FF4655; margin: 10px 0;">{}</h1>
                <p style="color: #7B8084;">{} • {} credits</p>
            </div>
            """.format(main_weapon['Weapon'], main_weapon['Category'], main_weapon['Cost']), 
            unsafe_allow_html=True)
            
            # Weapon stats chart
            fig = go.Figure()
            
            categories = ['Damage', 'Fire Rate', 'Magazine', 'Accuracy']
            main_values = [
                min(main_weapon['Head Damage (0-30m)'] / 255 * 100, 100),
                min(main_weapon['Fire Rate (rounds/sec)'] / 18 * 100, 100),
                min(main_weapon['Magazine Size'] / 100 * 100, 100),
                100 - (main_weapon['Reload Time (sec)'] / 5 * 50)
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=main_values,
                theta=categories,
                fill='toself',
                name=main_weapon['Weapon'],
                line_color='#FF4655'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100]),
                    bgcolor='#1F2326'
                ),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ECE8E1',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 **Tip:** {get_weapon_advice(main_weapon['Weapon'])}")
            
        with col2:
            st.markdown("""
            <div class="success-card">
                <span class="metric-label">SECONDARY OPTION</span>
                <h2 style="color: #3DDC97; margin: 10px 0;">{}</h2>
                <p style="color: #7B8084;">{} • {} credits</p>
            </div>
            """.format(secondary_weapon['Weapon'], secondary_weapon['Category'], secondary_weapon['Cost']), 
            unsafe_allow_html=True)
            
            # Stats comparison
            st.markdown("#### 📊 Quick Stats")
            
            metrics = {
                'Head Damage': (main_weapon['Head Damage (0-30m)'], secondary_weapon['Head Damage (0-30m)']),
                'Body Damage': (main_weapon['Body Damage (0-30m)'], secondary_weapon['Body Damage (0-30m)']),
                'Fire Rate': (main_weapon['Fire Rate (rounds/sec)'], secondary_weapon['Fire Rate (rounds/sec)']),
                'Magazine': (main_weapon['Magazine Size'], secondary_weapon['Magazine Size']),
            }
            
            for stat, (m, s) in metrics.items():
                col_a, col_b, col_c = st.columns([2, 1, 2])
                with col_a:
                    st.metric(stat, f"{m}")
                with col_b:
                    st.markdown("vs")
                with col_c:
                    st.metric("", f"{s}")
            
            st.info(f"💰 **Budget tip:** Use {secondary_weapon['Weapon']} on eco/force rounds when you can't afford {main_weapon['Weapon']}")
        
        # All weapons comparison
        st.divider()
        st.markdown("### 📈 All Weapons Ranked For You (Decision Tree Confidence)")
        
        # Filter to only weapons that exist in our dataset
        top_weapons = [(w, s) for w, s in sorted_weapons[:10] if w in weapons['Weapon'].values]
        
        fig_bar = px.bar(
            x=[w[0] for w in top_weapons],
            y=[w[1] * 100 for w in top_weapons],  # Convert to percentage
            color=[w[1] * 100 for w in top_weapons],
            color_continuous_scale=['#1F2326', '#FF4655'],
            labels={'x': 'Weapon', 'y': 'Match Confidence (%)'}
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ECE8E1',
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Show Decision Tree Visualization
        st.divider()
        st.markdown("### 🌳 Decision Tree Visualization")
        st.markdown("*This is how the AI makes decisions based on your inputs*")
        
        with st.expander("📊 View Decision Tree", expanded=False):
            fig_tree, ax = plt.subplots(figsize=(20, 12))
            plot_tree(clf, 
                     feature_names=feature_names,
                     class_names=label_encoder.classes_,
                     filled=True, 
                     rounded=True,
                     ax=ax,
                     fontsize=8,
                     max_depth=3)  # Limit depth for readability
            plt.title("Weapon Recommendation Decision Tree (Top 3 Levels)", fontsize=14, color='white')
            fig_tree.patch.set_facecolor('#0F1923')
            ax.set_facecolor('#0F1923')
            st.pyplot(fig_tree)
            
            # Also show text representation
            st.markdown("#### 📝 Decision Rules (Text)")
            tree_rules = export_text(clf, feature_names=feature_names, max_depth=4)
            st.code(tree_rules, language='text')


def calculate_weapon_scores(weapons, rank, sensitivity, playstyle, preferred_range, aim_style, headshot_conf, budget):
    scores = {}
    
    for _, weapon in weapons.iterrows():
        name = weapon['Weapon']
        if name == 'Knife':
            continue
            
        score = 50
        
        # Rank scoring
        low_ranks = ['Iron', 'Bronze', 'Silver']
        high_ranks = ['Immortal', 'Radiant', 'Ascendant', 'Diamond']
        
        if rank in low_ranks:
            if weapon['Category'] in ['SMG', 'Machine Gun']:
                score += 15
            if weapon['Magazine Size'] >= 25:
                score += 10
        elif rank in high_ranks:
            if weapon['Category'] in ['Rifle', 'Sniper']:
                score += 15
            if weapon['Head Damage (0-30m)'] >= 150:
                score += 10
                
        # Sensitivity
        if 'Low' in sensitivity:
            if weapon['Category'] == 'Rifle':
                score += 15
        elif 'High' in sensitivity:
            if weapon['Category'] in ['Shotgun', 'Sniper']:
                score += 15
                
        # Playstyle
        if 'Aggressive' in playstyle:
            if weapon['Category'] in ['SMG', 'Shotgun']:
                score += 15
            if weapon['Fire Rate (rounds/sec)'] >= 10:
                score += 10
        elif 'Passive' in playstyle:
            if weapon['Category'] in ['Rifle', 'Sniper']:
                score += 15
                
        # Range
        if 'Close' in preferred_range:
            if weapon['Category'] in ['Shotgun', 'SMG', 'Sidearm']:
                score += 15
        elif 'Long' in preferred_range:
            if weapon['Category'] in ['Sniper', 'Rifle']:
                score += 15
            if weapon['Weapon'] in ['Vandal', 'Guardian']:
                score += 10
                
        # Aim style
        if 'Flick' in aim_style:
            if weapon['Category'] in ['Sniper', 'Shotgun']:
                score += 15
        elif 'Track' in aim_style:
            if weapon['Fire Mode'] == 'Full-Auto':
                score += 15
        else:  # Spray
            if weapon['Category'] == 'Rifle':
                score += 15
            if weapon['Weapon'] in ['Phantom', 'Vandal']:
                score += 10
                
        # Headshot confidence
        if 'High' in headshot_conf:
            if weapon['Head Damage (0-30m)'] >= 150:
                score += 20
        elif 'Low' in headshot_conf:
            if weapon['Magazine Size'] >= 25:
                score += 10
                
        # Budget
        cost = weapon['Cost']
        if 'Eco' in budget:
            if cost <= 1000:
                score += 20
            elif cost <= 2000:
                score += 10
            else:
                score -= 15
        elif 'Force' in budget:
            if 1500 <= cost <= 2500:
                score += 15
        else:
            if cost >= 2500:
                score += 15
                
        scores[name] = score
        
    return scores


def get_weapon_advice(weapon_name):
    advice = {
        'Vandal': "One-tap potential at ANY range. Master the first-bullet accuracy for instant kills.",
        'Phantom': "Silenced with no tracers — perfect for spraying through smokes. Forgiving spray pattern.",
        'Operator': "The ultimate angle holder. Never peek with it — let them come to you.",
        'Sheriff': "Eco round beast. One-tap heads at any range for only 800 credits.",
        'Ghost': "Most reliable pistol. Silenced, accurate, cheap. Your go-to for pistol rounds.",
        'Spectre': "Run and gun king. Great for aggressive plays and close angles.",
        'Marshal': "Budget sniper that rewards aggression. Quick-scope for early picks.",
        'Judge': "Corner monster. Hold tight angles and delete anyone who pushes.",
        'Guardian': "Two-tap body, one-tap head. Great for patient players who hit shots.",
        'Ares': "Spray and pray LMG. Good for holding chokes and wallbanging.",
        'Odin': "100 bullets of suppression. Wallbang common spots to tilt enemies.",
        'Bulldog': "Budget rifle with burst mode. Right-click for medium range precision.",
        'Stinger': "Cheap force buy option. Burst at range, spray up close.",
        'Bucky': "One-shot potential up close. Right-click for medium range slug.",
        'Frenzy': "Full-auto pistol. Great for eco rush plays.",
        'Classic': "Free but deadly. Right-click burst at close range shreds.",
        'Shorty': "150 credit pocket shotgun. Punish careless pushers.",
        'Outlaw': "Double-barrel sniper. Two quick shots for aggressive plays."
    }
    return advice.get(weapon_name, "Solid choice for your playstyle!")


# ======================================================================
# STRATEGY QUIZ
# ======================================================================
def strategy_quiz():
    st.markdown("# 🧠 STRATEGY QUIZ")
    st.markdown("*Test your game sense with tactical scenarios*")
    st.divider()
    
    # Initialize session state
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = []
    if 'current_q' not in st.session_state:
        st.session_state.current_q = 0
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'answered' not in st.session_state:
        st.session_state.answered = False
    if 'show_explanation' not in st.session_state:
        st.session_state.show_explanation = False
    
    strategies = load_strategies()
    
    if not st.session_state.quiz_started:
        # Start screen
        st.markdown("""
        <div class="stat-card">
            <h2>📋 How It Works</h2>
            <ul>
                <li>10 tactical scenarios from real game situations</li>
                <li>4 options per question — only ONE is optimal</li>
                <li>Score based on <strong>accuracy</strong> and <strong>speed</strong></li>
                <li>Wrong answers show explanations to help you learn</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 START QUIZ", use_container_width=True):
            st.session_state.quiz_started = True
            st.session_state.quiz_questions = strategies.sample(n=10).to_dict('records')
            st.session_state.current_q = 0
            st.session_state.score = 0
            st.session_state.start_time = time.time()
            st.session_state.answered = False
            st.session_state.show_explanation = False
            st.rerun()
            
    elif st.session_state.current_q >= len(st.session_state.quiz_questions):
        # Results screen
        show_quiz_results()
        
    else:
        # Quiz question
        show_quiz_question()


def show_quiz_question():
    q = st.session_state.quiz_questions[st.session_state.current_q]
    
    # Progress
    progress = (st.session_state.current_q) / len(st.session_state.quiz_questions)
    st.progress(progress)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Question {st.session_state.current_q + 1} of {len(st.session_state.quiz_questions)}**")
    with col2:
        st.markdown(f"**Score: {st.session_state.score}/{st.session_state.current_q}**")
    
    # Scenario
    side_text = "🔴 ATTACKING" if q['Side'] == 'A' else "🔵 DEFENDING"
    
    st.markdown(f"""
    <div class="stat-card">
        <h3>📍 {q['Map']} — {q['Location']}</h3>
        <h4>{side_text}</h4>
        <h2 style="color: #FFD700;">⚠️ {q['Situation'].upper()}</h2>
        <p style="color: #ECE8E1; font-size: 18px;">What's the optimal play?</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Options
    if not st.session_state.show_explanation:
        options = [q['Option1'], q['Option2'], q['Option3'], q['Option4']]
        
        cols = st.columns(2)
        for i, option in enumerate(options):
            with cols[i % 2]:
                if st.button(f"{chr(65+i)}. {option}", key=f"opt_{i}", use_container_width=True):
                    correct_idx = int(q['CorrectOption'])
                    if i == correct_idx:
                        st.session_state.score += 1
                        st.session_state.current_q += 1
                        st.session_state.answered = False
                        st.rerun()
                    else:
                        st.session_state.show_explanation = True
                        st.session_state.wrong_answer = option
                        st.session_state.correct_answer = options[correct_idx]
                        st.rerun()
    else:
        # Show explanation
        st.error(f"❌ Your answer: {st.session_state.wrong_answer}")
        st.success(f"✅ Correct answer: {st.session_state.correct_answer}")
        
        st.markdown(f"""
        <div class="warning-card">
            <h4>💡 Why?</h4>
            <p style="color: #ECE8E1;">{q['Explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Next Question →", use_container_width=True):
            st.session_state.current_q += 1
            st.session_state.show_explanation = False
            st.rerun()


def show_quiz_results():
    total_time = time.time() - st.session_state.start_time
    accuracy = (st.session_state.score / len(st.session_state.quiz_questions)) * 100
    avg_time = total_time / len(st.session_state.quiz_questions)
    
    # Rank calculation
    if accuracy >= 90:
        rank = "RADIANT 👑"
        rank_color = "#FFD700"
    elif accuracy >= 80:
        rank = "IMMORTAL 💎"
        rank_color = "#FF4655"
    elif accuracy >= 70:
        rank = "DIAMOND 💠"
        rank_color = "#B9F2FF"
    elif accuracy >= 60:
        rank = "PLATINUM 🥈"
        rank_color = "#3DDC97"
    elif accuracy >= 50:
        rank = "GOLD 🥇"
        rank_color = "#FFD700"
    else:
        rank = "KEEP PRACTICING 💪"
        rank_color = "#7B8084"
    
    # Speed rating
    if avg_time < 5:
        speed = "⚡ Lightning Fast"
    elif avg_time < 10:
        speed = "🧠 Quick Thinker"
    elif avg_time < 15:
        speed = "🎯 Steady Player"
    else:
        speed = "🔍 Careful Analyst"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 40px;">
        <h1 style="color: {rank_color}; font-size: 48px;">{rank}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="text-align: center;">
            <span class="metric-label">SCORE</span>
            <div class="big-stat">{st.session_state.score}/{len(st.session_state.quiz_questions)}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="text-align: center;">
            <span class="metric-label">ACCURACY</span>
            <div class="big-stat">{accuracy:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        st.markdown(f"""
        <div class="stat-card" style="text-align: center;">
            <span class="metric-label">TIME</span>
            <div class="big-stat">{minutes}:{seconds:02d}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="success-card" style="text-align: center;">
        <h3>{speed}</h3>
        <p>Average: {avg_time:.1f}s per question</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Accuracy", 'font': {'color': '#ECE8E1'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#ECE8E1'},
            'bar': {'color': '#FF4655'},
            'bgcolor': '#1F2326',
            'bordercolor': '#7B8084',
            'steps': [
                {'range': [0, 50], 'color': '#1F2326'},
                {'range': [50, 70], 'color': '#3D5A80'},
                {'range': [70, 90], 'color': '#3DDC97'},
                {'range': [90, 100], 'color': '#FFD700'}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ECE8E1',
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Try Again", use_container_width=True):
            st.session_state.quiz_started = False
            st.rerun()
    with col2:
        if st.button("🏠 Main Menu", use_container_width=True):
            st.session_state.quiz_started = False
            st.session_state.page = 'home'
            st.rerun()


# ======================================================================
# DECISION TREE VIEWER
# ======================================================================
def decision_tree_viewer():
    st.markdown("# 🌳 DECISION TREE VIEWER")
    st.markdown("*Explore how the AI model makes weapon recommendations*")
    st.divider()
    
    clf, label_encoder, feature_names, accuracy = train_weapon_model()
    
    # Model stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="text-align: center;">
            <span class="metric-label">ACCURACY</span>
            <div class="big-stat">{accuracy*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="text-align: center;">
            <span class="metric-label">TREE DEPTH</span>
            <div class="big-stat">{clf.get_depth()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card" style="text-align: center;">
            <span class="metric-label">LEAF NODES</span>
            <div class="big-stat">{clf.get_n_leaves()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Tree depth selector
    max_depth_display = st.slider("Display Tree Depth", 1, min(clf.get_depth(), 6), 3)
    
    # Full tree visualization
    st.markdown("### 📊 Full Decision Tree")
    
    fig_tree, ax = plt.subplots(figsize=(24, 14))
    plot_tree(clf, 
             feature_names=feature_names,
             class_names=label_encoder.classes_,
             filled=True, 
             rounded=True,
             ax=ax,
             fontsize=7,
             max_depth=max_depth_display)
    plt.title(f"Weapon Recommendation Decision Tree (Depth: {max_depth_display})", fontsize=16, color='white')
    fig_tree.patch.set_facecolor('#0F1923')
    ax.set_facecolor('#0F1923')
    st.pyplot(fig_tree)
    
    # Feature importance
    st.divider()
    st.markdown("### 📈 Feature Importance")
    st.markdown("*Which factors matter most in weapon selection?*")
    
    importance = clf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    feature_labels = {
        'Rank': '🎖️ Rank',
        'Sensitivity': '🖱️ Sensitivity',
        'Playstyle': '⚔️ Playstyle',
        'PreferredRange': '📏 Preferred Range',
        'AimStyle': '🎮 Aim Style',
        'HeadshotConf': '🎯 Headshot Confidence',
        'Budget': '💰 Budget'
    }
    importance_df['Feature'] = importance_df['Feature'].map(feature_labels)
    
    fig_imp = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#3D5A80', '#FF4655']
    )
    fig_imp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ECE8E1',
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Text rules
    st.divider()
    st.markdown("### 📝 Decision Rules (Text Format)")
    
    with st.expander("View Full Decision Rules"):
        tree_rules = export_text(clf, feature_names=feature_names, max_depth=5)
        st.code(tree_rules, language='text')
    
    # Explanation
    st.markdown("""
    <div class="warning-card">
        <h4>💡 How to Read the Tree</h4>
        <ul>
            <li><strong>Each box</strong> = A decision point</li>
            <li><strong>Condition at top</strong> = The question being asked (e.g., "Rank <= 4")</li>
            <li><strong>gini</strong> = How "pure" the split is (0 = perfect, 0.5 = random)</li>
            <li><strong>samples</strong> = How many training profiles reached this node</li>
            <li><strong>value</strong> = Distribution of weapons at this point</li>
            <li><strong>class</strong> = The predicted weapon if you stop here</li>
            <li><strong>Color</strong> = The dominant weapon class (darker = more confident)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ======================================================================
# WEAPON DATABASE
# ======================================================================
def weapon_database():
    st.markdown("# 📊 WEAPON DATABASE")
    st.markdown("*Explore all Valorant weapons and their stats*")
    st.divider()
    
    weapons = load_weapons()
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        category = st.multiselect(
            "Filter by Category",
            weapons['Category'].unique().tolist(),
            default=weapons['Category'].unique().tolist()
        )
    with col2:
        price_range = st.slider(
            "Price Range",
            0, int(weapons['Cost'].max()),
            (0, int(weapons['Cost'].max()))
        )
    
    filtered = weapons[
        (weapons['Category'].isin(category)) & 
        (weapons['Cost'] >= price_range[0]) & 
        (weapons['Cost'] <= price_range[1])
    ]
    
    # Interactive chart
    fig = px.scatter(
        filtered,
        x='Cost',
        y='Head Damage (0-30m)',
        size='Magazine Size',
        color='Category',
        hover_name='Weapon',
        hover_data=['Fire Rate (rounds/sec)', 'Body Damage (0-30m)'],
        title='Weapons: Cost vs Damage',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ECE8E1',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("### 📋 Full Stats")
    st.dataframe(
        filtered.style.background_gradient(subset=['Head Damage (0-30m)', 'Cost'], cmap='Reds'),
        use_container_width=True,
        height=400
    )
    
    # Category breakdown
    st.markdown("### 📈 Category Analysis")
    
    cat_stats = weapons.groupby('Category').agg({
        'Cost': 'mean',
        'Head Damage (0-30m)': 'mean',
        'Fire Rate (rounds/sec)': 'mean'
    }).round(1)
    
    fig_cat = px.bar(
        cat_stats.reset_index(),
        x='Category',
        y='Head Damage (0-30m)',
        color='Cost',
        color_continuous_scale=['#3D5A80', '#FF4655'],
        title='Average Head Damage by Category'
    )
    fig_cat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ECE8E1'
    )
    st.plotly_chart(fig_cat, use_container_width=True)


# ======================================================================
# MAIN APP
# ======================================================================
def main():
    # Sidebar navigation
    with st.sidebar:
        st.markdown("# 🎮 VALORANT")
        st.markdown("### Agent Assistant")
        st.divider()
        
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔫 Weapon Recommender", "🧠 Strategy Quiz", "🌳 Decision Tree", "📊 Weapon Database"],
            label_visibility="collapsed"
        )
        
        st.divider()
        st.markdown("*Made for tactical players*")
    
    # Page routing
    if "Home" in page:
        # Home page
        st.markdown("""
        <div style="text-align: center; padding: 50px 0;">
            <h1 style="color: #FF4655; font-size: 64px; margin-bottom: 0;">VALORANT</h1>
            <h2 style="color: #ECE8E1;">Agent Assistant</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <h3>🔫 Weapon Recommender</h3>
                <p>Answer questions about your playstyle, rank, and preferences to get personalized weapon recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="stat-card">
                <h3>🧠 Strategy Quiz</h3>
                <p>Test your game sense with 10 tactical scenarios. Learn optimal plays for every situation.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        <div class="success-card" style="text-align: center;">
            <h4>👈 Select a mode from the sidebar to get started!</h4>
        </div>
        """, unsafe_allow_html=True)
        
    elif "Weapon Recommender" in page:
        weapon_recommender()
        
    elif "Strategy Quiz" in page:
        strategy_quiz()
        
    elif "Decision Tree" in page:
        decision_tree_viewer()
        
    elif "Weapon Database" in page:
        weapon_database()


if __name__ == "__main__":
    main()
