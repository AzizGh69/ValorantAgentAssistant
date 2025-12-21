# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import random
import time
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ======================================================================
# STYLE - Valorant Theme (Dark Red/Black)
# ======================================================================
class ValorantStyle:
    BG_PRIMARY = "#0F1923"      # Dark background
    BG_SECONDARY = "#1F2326"    # Slightly lighter
    ACCENT = "#FF4655"          # Valorant Red
    ACCENT2 = "#BD3944"         # Darker red
    TEXT_PRIMARY = "#ECE8E1"    # Light text
    TEXT_SECONDARY = "#7B8084"  # Muted text
    BUTTON_BG = "#FF4655"
    BUTTON_FG = "#FFFFFF"
    SUCCESS = "#3DDC97"
    WARNING = "#FFD700"

    @staticmethod
    def configure_style():
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except:
            pass
        style.configure('TFrame', background=ValorantStyle.BG_PRIMARY)
        style.configure('TLabel', background=ValorantStyle.BG_PRIMARY,
                        foreground=ValorantStyle.TEXT_PRIMARY, font=('Segoe UI', 11))
        style.configure('TButton', background=ValorantStyle.BUTTON_BG,
                        foreground=ValorantStyle.BUTTON_FG, font=('Segoe UI', 10, 'bold'))
        style.configure('TRadiobutton', background=ValorantStyle.BG_PRIMARY,
                        foreground=ValorantStyle.TEXT_PRIMARY, font=('Segoe UI', 11))
        return style


# ======================================================================
# DECISION TREE WEAPON MODEL
# ======================================================================
class WeaponDecisionTree:
    def __init__(self, csv_path="weapons.csv"):
        self.weapons = pd.read_csv(csv_path)
        self.clf = None
        self.label_encoder = LabelEncoder()
        self.feature_names = ['Rank', 'Sensitivity', 'Playstyle', 'PreferredRange', 
                              'AimStyle', 'HeadshotConf', 'Budget']
        self.accuracy = 0.0
        self._train_model()
        
    def _train_model(self):
        """Train Decision Tree on simulated player profiles"""
        np.random.seed(42)
        training_data = []
        
        for _, weapon in self.weapons.iterrows():
            if weapon['Weapon'] == 'Knife':
                continue
                
            # Generate 20 synthetic user profiles per weapon
            for _ in range(20):
                # Rank (0-8: Iron to Radiant)
                if weapon['Category'] in ['Rifle', 'Sniper']:
                    rank = np.random.randint(4, 9)
                elif weapon['Category'] in ['SMG', 'Machine Gun']:
                    rank = np.random.randint(0, 6)
                else:
                    rank = np.random.randint(0, 9)
                    
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
        df = pd.DataFrame(training_data, columns=self.feature_names + ['Weapon'])
        
        # Encode target
        df['WeaponEncoded'] = self.label_encoder.fit_transform(df['Weapon'])
        
        # Features and target
        X = df[self.feature_names]
        y = df['WeaponEncoded']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.clf = DecisionTreeClassifier(max_depth=8, random_state=42)
        self.clf.fit(X_train, y_train)
        
        self.accuracy = metrics.accuracy_score(y_test, self.clf.predict(X_test))
        
    def predict(self, answers):
        """Predict weapon based on user answers"""
        # Convert answers to numeric
        rank_map = {'Iron': 0, 'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4, 
                    'Diamond': 5, 'Ascendant': 6, 'Immortal': 7, 'Radiant': 8}
        sens_map = {'low': 0, 'medium': 1, 'high': 2}
        play_map = {'aggressive': 0, 'passive': 1, 'balanced': 2}
        range_map = {'close': 0, 'medium': 1, 'long': 2}
        aim_map = {'flick': 0, 'tracking': 1, 'spray_control': 2}
        hs_map = {'low': 0, 'medium': 1, 'high': 2}
        budget_map = {'eco': 0, 'mid': 1, 'full': 2}
        
        features = np.array([[
            rank_map.get(answers.get('rank', 'Gold'), 3),
            sens_map.get(answers.get('sensitivity', 'medium'), 1),
            play_map.get(answers.get('playstyle', 'balanced'), 2),
            range_map.get(answers.get('preferred_range', 'medium'), 1),
            aim_map.get(answers.get('aim_style', 'spray_control'), 2),
            hs_map.get(answers.get('headshot_confidence', 'medium'), 1),
            budget_map.get(answers.get('budget', 'mid'), 1)
        ]])
        
        # Predict
        prediction = self.clf.predict(features)[0]
        predicted_weapon = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities for all weapons
        proba = self.clf.predict_proba(features)[0]
        weapon_scores = dict(zip(self.label_encoder.classes_, proba))
        sorted_weapons = sorted(weapon_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get main weapon
        main_weapon = self.weapons[self.weapons['Weapon'] == predicted_weapon].iloc[0]
        main_cat = main_weapon['Category']
        
        # Find secondary from different category
        secondary_weapon = None
        for name, score in sorted_weapons[1:]:
            if name not in self.weapons['Weapon'].values:
                continue
            w = self.weapons[self.weapons['Weapon'] == name].iloc[0]
            if w['Category'] != main_cat:
                secondary_weapon = w
                break
        
        if secondary_weapon is None:
            for name, score in sorted_weapons[1:]:
                if name in self.weapons['Weapon'].values:
                    secondary_weapon = self.weapons[self.weapons['Weapon'] == name].iloc[0]
                    break
                    
        return main_weapon, secondary_weapon, sorted_weapons


# ======================================================================
# LEGACY WEAPON RECOMMENDER (for reference)
# ======================================================================
class WeaponRecommender:
    def __init__(self, csv_path="weapons.csv"):
        self.weapons = pd.read_csv(csv_path)
        
    def recommend(self, answers):
        """
        answers dict keys:
        - rank: Iron/Bronze/Silver/Gold/Platinum/Diamond/Ascendant/Immortal/Radiant
        - sensitivity: low/medium/high
        - playstyle: aggressive/passive/balanced
        - aim_style: flick/tracking/spray_control
        - budget: eco/mid/full
        - preferred_range: close/medium/long
        - headshot_confidence: low/medium/high
        """
        scores = {}
        
        for _, weapon in self.weapons.iterrows():
            name = weapon['Weapon']
            if name == 'Knife':
                continue
                
            score = 50  # Base score
            
            # Rank-based scoring
            rank = answers.get('rank', 'Gold')
            low_ranks = ['Iron', 'Bronze', 'Silver']
            high_ranks = ['Immortal', 'Radiant', 'Ascendant', 'Diamond']
            
            if rank in low_ranks:
                # Lower ranks: forgiving weapons
                if weapon['Category'] in ['SMG', 'Machine Gun']:
                    score += 15
                if weapon['Magazine Size'] >= 25:
                    score += 10
            elif rank in high_ranks:
                # High ranks: precision weapons
                if weapon['Category'] in ['Rifle', 'Sniper']:
                    score += 15
                if weapon['Head Damage (0-30m)'] >= 150:
                    score += 10
                    
            # Sensitivity scoring
            sens = answers.get('sensitivity', 'medium')
            if sens == 'low':
                # Low sens = good tracking, prefer rifles
                if weapon['Category'] == 'Rifle':
                    score += 15
                if weapon['Fire Mode'] == 'Full-Auto':
                    score += 5
            elif sens == 'high':
                # High sens = good flicks, prefer shotguns/snipers
                if weapon['Category'] in ['Shotgun', 'Sniper', 'Sidearm']:
                    score += 15
                    
            # Playstyle scoring
            playstyle = answers.get('playstyle', 'balanced')
            if playstyle == 'aggressive':
                if weapon['Category'] in ['SMG', 'Shotgun']:
                    score += 15
                if weapon['Fire Rate (rounds/sec)'] >= 10:
                    score += 10
            elif playstyle == 'passive':
                if weapon['Category'] in ['Rifle', 'Sniper']:
                    score += 15
                if weapon['Head Damage (0-30m)'] >= 140:
                    score += 10
                    
            # Budget scoring
            budget = answers.get('budget', 'mid')
            cost = weapon['Cost']
            if budget == 'eco':
                if cost <= 1000:
                    score += 20
                elif cost <= 2000:
                    score += 10
                else:
                    score -= 15
            elif budget == 'mid':
                if 1500 <= cost <= 2500:
                    score += 15
            else:  # full buy
                if cost >= 2500:
                    score += 15
                    
            # Preferred range scoring
            pref_range = answers.get('preferred_range', 'medium')
            if pref_range == 'close':
                if weapon['Category'] in ['Shotgun', 'SMG', 'Sidearm']:
                    score += 15
            elif pref_range == 'long':
                if weapon['Category'] in ['Sniper', 'Rifle']:
                    score += 15
                if weapon['Weapon'] in ['Vandal', 'Guardian']:
                    score += 10
                    
            # Headshot confidence scoring
            hs_conf = answers.get('headshot_confidence', 'medium')
            if hs_conf == 'high':
                if weapon['Head Damage (0-30m)'] >= 150:
                    score += 20
                if weapon['Weapon'] in ['Vandal', 'Sheriff', 'Guardian']:
                    score += 15
            elif hs_conf == 'low':
                if weapon['Body Damage (0-30m)'] >= 30:
                    score += 10
                if weapon['Magazine Size'] >= 25:
                    score += 10
                    
            # Aim style scoring
            aim_style = answers.get('aim_style', 'spray_control')
            if aim_style == 'flick':
                if weapon['Category'] in ['Sniper', 'Shotgun']:
                    score += 15
                if weapon['Fire Mode'] == 'Semi-Auto':
                    score += 10
            elif aim_style == 'tracking':
                if weapon['Fire Mode'] == 'Full-Auto':
                    score += 15
                if weapon['Fire Rate (rounds/sec)'] >= 10:
                    score += 5
            elif aim_style == 'spray_control':
                if weapon['Category'] == 'Rifle':
                    score += 15
                if weapon['Weapon'] in ['Phantom', 'Vandal']:
                    score += 10
                    
            scores[name] = score
            
        # Sort by score
        sorted_weapons = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get main and secondary (different categories)
        main = sorted_weapons[0]
        main_weapon = self.weapons[self.weapons['Weapon'] == main[0]].iloc[0]
        
        # Secondary from different category
        main_cat = main_weapon['Category']
        for weapon_name, score in sorted_weapons[1:]:
            weapon_data = self.weapons[self.weapons['Weapon'] == weapon_name].iloc[0]
            if weapon_data['Category'] != main_cat:
                secondary = (weapon_name, score)
                secondary_weapon = weapon_data
                break
        else:
            secondary = sorted_weapons[1]
            secondary_weapon = self.weapons[self.weapons['Weapon'] == secondary[0]].iloc[0]
            
        return main_weapon, secondary_weapon


# ======================================================================
# STRATEGY QUIZ
# ======================================================================
class StrategyQuiz:
    def __init__(self, csv_path="valorant_strategies.csv"):
        self.data = pd.read_csv(csv_path)
        self.questions = []
        self.current_idx = 0
        self.score = 0
        self.start_time = None
        self.answers = []  # (correct, time_taken)
        
    def start_session(self, num_questions=10):
        self.questions = self.data.sample(n=min(num_questions, len(self.data))).to_dict('records')
        self.current_idx = 0
        self.score = 0
        self.start_time = time.time()
        self.answers = []
        return self.get_current_question()
        
    def get_current_question(self):
        if self.current_idx >= len(self.questions):
            return None
        return self.questions[self.current_idx]
        
    def answer_question(self, selected_idx):
        q = self.questions[self.current_idx]
        correct_idx = int(q['CorrectOption'])
        is_correct = selected_idx == correct_idx
        time_taken = time.time() - self.start_time
        
        if is_correct:
            self.score += 1
            
        self.answers.append((is_correct, time_taken))
        self.current_idx += 1
        
        return is_correct, q['Explanation'], q[f'Option{correct_idx + 1}']
        
    def get_results(self):
        total_time = time.time() - self.start_time
        accuracy = (self.score / len(self.questions)) * 100 if self.questions else 0
        
        # Speed bonus calculation
        avg_time_per_q = total_time / len(self.questions) if self.questions else 0
        if avg_time_per_q < 5:
            speed_rating = "Lightning Fast ⚡"
        elif avg_time_per_q < 10:
            speed_rating = "Quick Thinker 🧠"
        elif avg_time_per_q < 15:
            speed_rating = "Steady Player 🎯"
        else:
            speed_rating = "Careful Analyst 🔍"
            
        # Rank based on accuracy
        if accuracy >= 90:
            rank = "Radiant 👑"
        elif accuracy >= 80:
            rank = "Immortal 💎"
        elif accuracy >= 70:
            rank = "Diamond 💠"
        elif accuracy >= 60:
            rank = "Platinum 🥈"
        elif accuracy >= 50:
            rank = "Gold 🥇"
        else:
            rank = "Keep Practicing 💪"
            
        return {
            'score': self.score,
            'total': len(self.questions),
            'accuracy': accuracy,
            'total_time': total_time,
            'avg_time': avg_time_per_q,
            'speed_rating': speed_rating,
            'rank': rank
        }


# ======================================================================
# MAIN APPLICATION
# ======================================================================
class ValorantAgent:
    def __init__(self, root):
        self.root = root
        self.root.title("VALORANT Agent — Weapon Advisor & Strategy Trainer")
        self.root.geometry("900x700")
        self.root.configure(bg=ValorantStyle.BG_PRIMARY)
        
        ValorantStyle.configure_style()
        
        # Initialize Decision Tree model
        self.weapon_model = WeaponDecisionTree("weapons.csv")
        self.strategy_quiz = StrategyQuiz("valorant_strategies.csv")
        
        # State variables
        self.current_workflow = None
        self.weapon_answers = {}
        self.question_start_time = None
        
        self._create_main_menu()
        
    def _clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
            
    def _create_main_menu(self):
        self._clear_window()
        
        # Header
        header = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        header.pack(fill=tk.X, pady=30)
        
        tk.Label(header, text="VALORANT", font=("Segoe UI", 36, "bold"),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.ACCENT).pack()
        tk.Label(header, text="Agent Assistant", font=("Segoe UI", 18),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.TEXT_PRIMARY).pack()
                 
        # Menu buttons
        menu = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        menu.pack(expand=True)
        
        tk.Button(menu, text="🔫  WEAPON RECOMMENDER", font=("Segoe UI", 14, "bold"),
                  bg=ValorantStyle.ACCENT, fg="white", width=30, height=2,
                  command=self._start_weapon_workflow).pack(pady=15)
                  
        tk.Label(menu, text="Find your perfect weapon based on your playstyle",
                 font=("Segoe UI", 10), bg=ValorantStyle.BG_PRIMARY,
                 fg=ValorantStyle.TEXT_SECONDARY).pack()
                 
        tk.Button(menu, text="🧠  STRATEGY QUIZ", font=("Segoe UI", 14, "bold"),
                  bg=ValorantStyle.ACCENT2, fg="white", width=30, height=2,
                  command=self._start_quiz_workflow).pack(pady=15)
                  
        tk.Label(menu, text="Test your game sense with 10 tactical scenarios",
                 font=("Segoe UI", 10), bg=ValorantStyle.BG_PRIMARY,
                 fg=ValorantStyle.TEXT_SECONDARY).pack()
                 
        tk.Button(menu, text="🌳  VIEW DECISION TREE", font=("Segoe UI", 14, "bold"),
                  bg=ValorantStyle.SUCCESS, fg="white", width=30, height=2,
                  command=self._show_tree_visualization).pack(pady=15)
                  
        tk.Label(menu, text=f"See how the ML model makes decisions (Accuracy: {self.weapon_model.accuracy:.1%})",
                 font=("Segoe UI", 10), bg=ValorantStyle.BG_PRIMARY,
                 fg=ValorantStyle.TEXT_SECONDARY).pack()
                 
        # Footer
        footer = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        footer.pack(side=tk.BOTTOM, pady=20)
        tk.Label(footer, text="Made for tactical players", font=("Segoe UI", 9),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.TEXT_SECONDARY).pack()

    # ==================================================================
    # WEAPON RECOMMENDER WORKFLOW
    # ==================================================================
    def _start_weapon_workflow(self):
        self._clear_window()
        self.weapon_answers = {}
        self.current_question = 0
        
        self.weapon_questions = [
            {
                'key': 'rank',
                'question': "What's your current rank?",
                'subtext': "Be honest! This helps us match weapon difficulty to your skill.",
                'options': ['Iron', 'Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Ascendant', 'Immortal', 'Radiant']
            },
            {
                'key': 'sensitivity',
                'question': "How would you describe your mouse sensitivity?",
                'subtext': "Low = large mouse movements, High = small wrist flicks",
                'options': ['Low (arm aimer)', 'Medium (hybrid)', 'High (wrist aimer)'],
                'values': ['low', 'medium', 'high']
            },
            {
                'key': 'playstyle',
                'question': "How do you like to play?",
                'subtext': "Think about how you naturally move around the map.",
                'options': ['Aggressive (entry, first in)', 'Passive (hold angles, support)', 'Balanced (adapt to situation)'],
                'values': ['aggressive', 'passive', 'balanced']
            },
            {
                'key': 'preferred_range',
                'question': "What range do most of your kills happen at?",
                'subtext': "Think about where you feel most confident taking fights.",
                'options': ['Close (shotgun range)', 'Medium (site distances)', 'Long (cross-map)'],
                'values': ['close', 'medium', 'long']
            },
            {
                'key': 'aim_style',
                'question': "When an enemy appears, what do you do?",
                'subtext': "This tells us about your natural aim mechanics.",
                'options': ['Snap to their head quickly (flick)', 'Track their movement smoothly (tracking)', 'Aim at body and control spray (spray control)'],
                'values': ['flick', 'tracking', 'spray_control']
            },
            {
                'key': 'headshot_confidence',
                'question': "How often do you hit headshots?",
                'subtext': "Be honest — this affects whether you need a one-tap gun.",
                'options': ['Rarely (I aim for body)', 'Sometimes (when I focus)', 'Often (headshot machine)'],
                'values': ['low', 'medium', 'high']
            },
            {
                'key': 'budget',
                'question': "How do you manage your economy?",
                'subtext': "Your buy habits affect which weapons suit you.",
                'options': ['Often eco/saving', 'Usually force or half-buy', 'Full buy whenever possible'],
                'values': ['eco', 'mid', 'full']
            }
        ]
        
        self._show_weapon_question()
        
    def _show_weapon_question(self):
        self._clear_window()
        
        if self.current_question >= len(self.weapon_questions):
            self._show_weapon_results()
            return
            
        q = self.weapon_questions[self.current_question]
        
        # Progress bar
        progress = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        progress.pack(fill=tk.X, padx=40, pady=20)
        
        progress_pct = ((self.current_question + 1) / len(self.weapon_questions)) * 100
        tk.Label(progress, text=f"Question {self.current_question + 1}/{len(self.weapon_questions)}",
                 font=("Segoe UI", 10), bg=ValorantStyle.BG_PRIMARY,
                 fg=ValorantStyle.TEXT_SECONDARY).pack(anchor='w')
        
        bar_frame = tk.Frame(progress, bg=ValorantStyle.BG_SECONDARY, height=8, width=820)
        bar_frame.pack(fill=tk.X, pady=5)
        bar_frame.pack_propagate(False)
        bar_width = int(820 * progress_pct / 100)
        filled = tk.Frame(bar_frame, bg=ValorantStyle.ACCENT, height=8, width=bar_width)
        filled.place(x=0, y=0)
        
        # Question
        q_frame = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        q_frame.pack(expand=True, fill=tk.BOTH, padx=40)
        
        tk.Label(q_frame, text=q['question'], font=("Segoe UI", 18, "bold"),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.TEXT_PRIMARY,
                 wraplength=700).pack(pady=10)
        tk.Label(q_frame, text=q['subtext'], font=("Segoe UI", 11),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.TEXT_SECONDARY,
                 wraplength=700).pack(pady=5)
        
        # Options
        options_frame = tk.Frame(q_frame, bg=ValorantStyle.BG_PRIMARY)
        options_frame.pack(pady=20)
        
        for i, option in enumerate(q['options']):
            value = q.get('values', q['options'])[i] if 'values' in q else option
            btn = tk.Button(options_frame, text=option, font=("Segoe UI", 12),
                           bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_PRIMARY,
                           width=50, height=2, cursor='hand2',
                           command=lambda v=value, k=q['key']: self._answer_weapon_question(k, v))
            btn.pack(pady=5)
            
        # Back button
        if self.current_question > 0:
            tk.Button(self.root, text="← Back", font=("Segoe UI", 10),
                     bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_SECONDARY,
                     command=self._prev_weapon_question).pack(pady=10)
                     
    def _answer_weapon_question(self, key, value):
        self.weapon_answers[key] = value
        self.current_question += 1
        self._show_weapon_question()
        
    def _prev_weapon_question(self):
        self.current_question -= 1
        self._show_weapon_question()
        
    def _show_weapon_results(self):
        self._clear_window()
        
        main, secondary, sorted_weapons = self.weapon_model.predict(self.weapon_answers)
        
        # Header
        header = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        header.pack(fill=tk.X, pady=20)
        tk.Label(header, text="YOUR PERFECT LOADOUT", font=("Segoe UI", 24, "bold"),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.ACCENT).pack()
        tk.Label(header, text=f"Decision Tree Model Accuracy: {self.weapon_model.accuracy:.1%}",
                 font=("Segoe UI", 10), bg=ValorantStyle.BG_PRIMARY,
                 fg=ValorantStyle.SUCCESS).pack()
        
        # Results frame
        results = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        results.pack(expand=True, fill=tk.BOTH, padx=40)
        
        # Main weapon
        main_frame = tk.Frame(results, bg=ValorantStyle.BG_SECONDARY, padx=20, pady=20)
        main_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(main_frame, text="🎯 PRIMARY WEAPON", font=("Segoe UI", 12, "bold"),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.ACCENT).pack(anchor='w')
        tk.Label(main_frame, text=main['Weapon'], font=("Segoe UI", 28, "bold"),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_PRIMARY).pack(anchor='w')
        tk.Label(main_frame, text=f"{main['Category']} • {main['Cost']} credits",
                 font=("Segoe UI", 12), bg=ValorantStyle.BG_SECONDARY,
                 fg=ValorantStyle.TEXT_SECONDARY).pack(anchor='w')
        
        main_stats = f"Damage: {main['Head Damage (0-30m)']} head / {main['Body Damage (0-30m)']} body • Fire Rate: {main['Fire Rate (rounds/sec)']} • Mag: {main['Magazine Size']}"
        tk.Label(main_frame, text=main_stats, font=("Segoe UI", 10),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_SECONDARY).pack(anchor='w', pady=5)
        tk.Label(main_frame, text=f"💡 {main['Special Notes']}", font=("Segoe UI", 11, "italic"),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.SUCCESS).pack(anchor='w')
        
        # Advice for main
        main_advice = self._get_weapon_advice(main, "main")
        tk.Label(main_frame, text=main_advice, font=("Segoe UI", 10),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_PRIMARY,
                 wraplength=700, justify='left').pack(anchor='w', pady=10)
        
        # Secondary weapon
        sec_frame = tk.Frame(results, bg=ValorantStyle.BG_SECONDARY, padx=20, pady=20)
        sec_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(sec_frame, text="🔄 SECONDARY OPTION", font=("Segoe UI", 12, "bold"),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.WARNING).pack(anchor='w')
        tk.Label(sec_frame, text=secondary['Weapon'], font=("Segoe UI", 22, "bold"),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_PRIMARY).pack(anchor='w')
        tk.Label(sec_frame, text=f"{secondary['Category']} • {secondary['Cost']} credits",
                 font=("Segoe UI", 11), bg=ValorantStyle.BG_SECONDARY,
                 fg=ValorantStyle.TEXT_SECONDARY).pack(anchor='w')
        
        sec_advice = self._get_weapon_advice(secondary, "secondary")
        tk.Label(sec_frame, text=sec_advice, font=("Segoe UI", 10),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_PRIMARY,
                 wraplength=700, justify='left').pack(anchor='w', pady=5)
        
        # Buttons
        btn_frame = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="🌳 Visualize Tree", font=("Segoe UI", 11),
                 bg=ValorantStyle.SUCCESS, fg="white",
                 command=self._show_tree_visualization).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Try Again", font=("Segoe UI", 11),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_PRIMARY,
                 command=self._start_weapon_workflow).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Main Menu", font=("Segoe UI", 11),
                 bg=ValorantStyle.ACCENT, fg="white",
                 command=self._create_main_menu).pack(side=tk.LEFT, padx=10)
                 
    def _get_weapon_advice(self, weapon, wtype):
        name = weapon['Weapon']
        cat = weapon['Category']
        cost = weapon['Cost']
        
        advice_map = {
            'Vandal': "Buy this on full buy rounds when you're confident in your aim. One-tap potential at any range makes it deadly for headshots.",
            'Phantom': "Great for holding angles and spraying. The silencer hides your tracers — perfect for spraying through smokes.",
            'Operator': "The ultimate angle-holder. Buy when your team has economy lead. Don't peek with it — let them come to you.",
            'Sheriff': "Eco round beast. If your headshots are on point, this can win anti-ecos. Otherwise, stick to Ghost.",
            'Ghost': "Most reliable pistol. Silenced, accurate, cheap. Perfect for pistol rounds and light buys.",
            'Spectre': "Run and gun demon. Great for aggressive plays and close angles. Best SMG in the game.",
            'Marshal': "Budget sniper for aggressive peeking. Mobile and deadly. Great for early picks.",
            'Judge': "Corner monster. Hold close angles and punish rushers. Don't use in open spaces.",
            'Stinger': "Cheap force buy option. Right-click burst at medium range, full auto up close.",
            'Bulldog': "Budget rifle with burst ADS. Good for half-buys when you can't afford Phantom/Vandal.",
            'Guardian': "DMR for the patient player. Two-tap body kills, one-tap heads. Great for wallbangs.",
            'Ares': "Spray and pray LMG. Good for holding chokes and wallbanging. Accurate while moving.",
            'Odin': "The wallbang machine. 100 bullets of suppressive fire. Great for spamming common spots.",
            'Bucky': "One-shot potential up close. Right-click for medium range slug shot.",
            'Classic': "Free pistol, surprisingly strong. Right-click burst at close range is deadly.",
            'Frenzy': "Full-auto pistol. Great for rushing on eco rounds.",
            'Shorty': "Pocket shotgun. 150 credits for a close-range assassin. Punish careless pushers."
        }
        
        base = advice_map.get(name, f"A solid {cat.lower()} choice for your playstyle.")
        
        if wtype == "main":
            return f"📌 {base}"
        else:
            if cost < 1500:
                return f"💰 Use this on eco/force buy rounds when you can't afford your primary. {base}"
            else:
                return f"🔀 Good alternative when your primary isn't working or you want to mix it up. {base}"
    
    def _show_tree_visualization(self):
        """Display the Decision Tree in a new window"""
        tree_window = tk.Toplevel(self.root)
        tree_window.title("Decision Tree Visualization - Weapon Recommender")
        tree_window.geometry("1200x800")
        tree_window.configure(bg=ValorantStyle.BG_PRIMARY)
        
        # Header
        tk.Label(tree_window, text="🌳 Decision Tree Visualization", 
                 font=("Segoe UI", 18, "bold"),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.ACCENT).pack(pady=10)
        
        tk.Label(tree_window, text=f"Model Accuracy: {self.weapon_model.accuracy:.1%}",
                 font=("Segoe UI", 12),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.SUCCESS).pack()
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('#0F1923')
        ax.set_facecolor('#0F1923')
        
        # Plot tree
        plot_tree(self.weapon_model.clf, 
                  feature_names=self.weapon_model.feature_names,
                  class_names=self.weapon_model.label_encoder.classes_,
                  filled=True,
                  rounded=True,
                  max_depth=4,
                  fontsize=7,
                  ax=ax)
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=tree_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button frame
        btn_frame = tk.Frame(tree_window, bg=ValorantStyle.BG_PRIMARY)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Show Full Rules", font=("Segoe UI", 10),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_PRIMARY,
                 command=lambda: self._show_tree_rules(tree_window)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Close", font=("Segoe UI", 10),
                 bg=ValorantStyle.ACCENT, fg="white",
                 command=tree_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def _show_tree_rules(self, parent):
        """Show decision tree rules as text"""
        rules_window = tk.Toplevel(parent)
        rules_window.title("Decision Tree Rules")
        rules_window.geometry("800x600")
        rules_window.configure(bg=ValorantStyle.BG_PRIMARY)
        
        tk.Label(rules_window, text="📋 Decision Tree Rules", 
                 font=("Segoe UI", 16, "bold"),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.ACCENT).pack(pady=10)
        
        # Get rules as text
        rules_text = export_text(self.weapon_model.clf, 
                                  feature_names=self.weapon_model.feature_names,
                                  max_depth=6)
        
        # Create scrolled text widget
        text_widget = scrolledtext.ScrolledText(rules_window, 
                                                 font=("Consolas", 10),
                                                 bg=ValorantStyle.BG_SECONDARY,
                                                 fg=ValorantStyle.TEXT_PRIMARY,
                                                 width=90, height=30)
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, rules_text)
        text_widget.configure(state='disabled')
        
        tk.Button(rules_window, text="Close", font=("Segoe UI", 10),
                 bg=ValorantStyle.ACCENT, fg="white",
                 command=rules_window.destroy).pack(pady=10)

    # ==================================================================
    # STRATEGY QUIZ WORKFLOW
    # ==================================================================
    def _start_quiz_workflow(self):
        self._clear_window()
        
        # Start screen
        start_frame = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        start_frame.pack(expand=True)
        
        tk.Label(start_frame, text="🧠 STRATEGY QUIZ", font=("Segoe UI", 28, "bold"),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.ACCENT).pack(pady=20)
        tk.Label(start_frame, text="Test your tactical knowledge with 10 in-game scenarios",
                 font=("Segoe UI", 12), bg=ValorantStyle.BG_PRIMARY,
                 fg=ValorantStyle.TEXT_SECONDARY).pack()
                 
        rules = """
        📋 RULES:
        • 10 questions about real game situations
        • Each question has 4 possible actions
        • Only ONE is the optimal play
        • Your score is based on accuracy AND speed
        • Wrong answers will show an explanation
        """
        tk.Label(start_frame, text=rules, font=("Segoe UI", 11),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.TEXT_PRIMARY,
                 justify='left').pack(pady=20)
                 
        tk.Button(start_frame, text="START QUIZ", font=("Segoe UI", 14, "bold"),
                 bg=ValorantStyle.ACCENT, fg="white", width=20, height=2,
                 command=self._begin_quiz).pack(pady=20)
                 
        tk.Button(start_frame, text="← Back to Menu", font=("Segoe UI", 10),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_SECONDARY,
                 command=self._create_main_menu).pack()
                 
    def _begin_quiz(self):
        self.strategy_quiz.start_session(10)
        self.question_start_time = time.time()
        self._show_quiz_question()
        
    def _show_quiz_question(self):
        self._clear_window()
        
        question = self.strategy_quiz.get_current_question()
        if question is None:
            self._show_quiz_results()
            return
            
        # Progress
        progress = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        progress.pack(fill=tk.X, padx=40, pady=20)
        
        q_num = self.strategy_quiz.current_idx + 1
        total = len(self.strategy_quiz.questions)
        
        tk.Label(progress, text=f"Question {q_num}/{total}",
                 font=("Segoe UI", 12, "bold"), bg=ValorantStyle.BG_PRIMARY,
                 fg=ValorantStyle.TEXT_PRIMARY).pack(anchor='w')
        
        # Score display
        tk.Label(progress, text=f"Score: {self.strategy_quiz.score}/{self.strategy_quiz.current_idx}",
                 font=("Segoe UI", 10), bg=ValorantStyle.BG_PRIMARY,
                 fg=ValorantStyle.SUCCESS).pack(anchor='e')
        
        # Scenario description
        scenario_frame = tk.Frame(self.root, bg=ValorantStyle.BG_SECONDARY, padx=20, pady=20)
        scenario_frame.pack(fill=tk.X, padx=40, pady=10)
        
        side_text = "ATTACKING" if question['Side'] == 'A' else "DEFENDING"
        side_color = ValorantStyle.ACCENT if question['Side'] == 'A' else ValorantStyle.SUCCESS
        
        tk.Label(scenario_frame, text=f"📍 {question['Map']} — {question['Location']}",
                 font=("Segoe UI", 14, "bold"), bg=ValorantStyle.BG_SECONDARY,
                 fg=ValorantStyle.TEXT_PRIMARY).pack(anchor='w')
        tk.Label(scenario_frame, text=f"You are {side_text}",
                 font=("Segoe UI", 11, "bold"), bg=ValorantStyle.BG_SECONDARY,
                 fg=side_color).pack(anchor='w')
                 
        tk.Label(scenario_frame, text=f"Situation: {question['Situation'].upper()}",
                 font=("Segoe UI", 16, "bold"), bg=ValorantStyle.BG_SECONDARY,
                 fg=ValorantStyle.WARNING).pack(pady=15)
                 
        tk.Label(scenario_frame, text="What's the optimal play?",
                 font=("Segoe UI", 12), bg=ValorantStyle.BG_SECONDARY,
                 fg=ValorantStyle.TEXT_SECONDARY).pack()
        
        # Options
        options_frame = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        options_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=10)
        
        for i in range(4):
            option_text = question[f'Option{i+1}']
            btn = tk.Button(options_frame, text=f"{chr(65+i)}.  {option_text}",
                           font=("Segoe UI", 12), bg=ValorantStyle.BG_SECONDARY,
                           fg=ValorantStyle.TEXT_PRIMARY, width=60, height=2,
                           anchor='w', padx=20, cursor='hand2',
                           command=lambda idx=i: self._answer_quiz(idx))
            btn.pack(pady=5)
            
        self.question_start_time = time.time()
        
    def _answer_quiz(self, selected_idx):
        is_correct, explanation, correct_answer = self.strategy_quiz.answer_question(selected_idx)
        
        if is_correct:
            self._show_quiz_question()
        else:
            self._show_wrong_answer(explanation, correct_answer)
            
    def _show_wrong_answer(self, explanation, correct_answer):
        self._clear_window()
        
        frame = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        frame.pack(expand=True)
        
        tk.Label(frame, text="❌ INCORRECT", font=("Segoe UI", 24, "bold"),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.ACCENT).pack(pady=20)
                 
        tk.Label(frame, text=f"Correct answer: {correct_answer}",
                 font=("Segoe UI", 14, "bold"), bg=ValorantStyle.BG_PRIMARY,
                 fg=ValorantStyle.SUCCESS).pack(pady=10)
                 
        exp_frame = tk.Frame(frame, bg=ValorantStyle.BG_SECONDARY, padx=20, pady=20)
        exp_frame.pack(fill=tk.X, padx=40, pady=10)
        
        tk.Label(exp_frame, text="💡 Why?", font=("Segoe UI", 12, "bold"),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.WARNING).pack(anchor='w')
        tk.Label(exp_frame, text=explanation, font=("Segoe UI", 11),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_PRIMARY,
                 wraplength=600, justify='left').pack(pady=10)
                 
        tk.Button(frame, text="Next Question →", font=("Segoe UI", 12, "bold"),
                 bg=ValorantStyle.ACCENT, fg="white", width=20,
                 command=self._show_quiz_question).pack(pady=20)
                 
    def _show_quiz_results(self):
        self._clear_window()
        
        results = self.strategy_quiz.get_results()
        
        # Header
        tk.Label(self.root, text="📊 QUIZ COMPLETE", font=("Segoe UI", 28, "bold"),
                 bg=ValorantStyle.BG_PRIMARY, fg=ValorantStyle.ACCENT).pack(pady=20)
        
        # Results card
        results_frame = tk.Frame(self.root, bg=ValorantStyle.BG_SECONDARY, padx=40, pady=30)
        results_frame.pack(padx=40, pady=10)
        
        tk.Label(results_frame, text=results['rank'], font=("Segoe UI", 32, "bold"),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.WARNING).pack()
                 
        # Stats grid
        stats = tk.Frame(results_frame, bg=ValorantStyle.BG_SECONDARY)
        stats.pack(pady=20)
        
        # Score
        tk.Label(stats, text="SCORE", font=("Segoe UI", 10),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_SECONDARY).grid(row=0, column=0, padx=30)
        tk.Label(stats, text=f"{results['score']}/{results['total']}", font=("Segoe UI", 24, "bold"),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_PRIMARY).grid(row=1, column=0, padx=30)
        
        # Accuracy
        tk.Label(stats, text="ACCURACY", font=("Segoe UI", 10),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_SECONDARY).grid(row=0, column=1, padx=30)
        acc_color = ValorantStyle.SUCCESS if results['accuracy'] >= 70 else ValorantStyle.WARNING if results['accuracy'] >= 50 else ValorantStyle.ACCENT
        tk.Label(stats, text=f"{results['accuracy']:.0f}%", font=("Segoe UI", 24, "bold"),
                 bg=ValorantStyle.BG_SECONDARY, fg=acc_color).grid(row=1, column=1, padx=30)
        
        # Time
        tk.Label(stats, text="TOTAL TIME", font=("Segoe UI", 10),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_SECONDARY).grid(row=0, column=2, padx=30)
        minutes = int(results['total_time'] // 60)
        seconds = int(results['total_time'] % 60)
        tk.Label(stats, text=f"{minutes}:{seconds:02d}", font=("Segoe UI", 24, "bold"),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_PRIMARY).grid(row=1, column=2, padx=30)
        
        # Speed rating
        tk.Label(results_frame, text=results['speed_rating'], font=("Segoe UI", 14),
                 bg=ValorantStyle.BG_SECONDARY, fg=ValorantStyle.TEXT_SECONDARY).pack(pady=10)
        
        # Avg time per question
        tk.Label(results_frame, text=f"Average: {results['avg_time']:.1f}s per question",
                 font=("Segoe UI", 11), bg=ValorantStyle.BG_SECONDARY,
                 fg=ValorantStyle.TEXT_SECONDARY).pack()
        
        # Buttons
        btn_frame = tk.Frame(self.root, bg=ValorantStyle.BG_PRIMARY)
        btn_frame.pack(pady=30)
        
        tk.Button(btn_frame, text="Try Again", font=("Segoe UI", 12),
                 bg=ValorantStyle.ACCENT2, fg="white", width=15,
                 command=self._start_quiz_workflow).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Main Menu", font=("Segoe UI", 12),
                 bg=ValorantStyle.ACCENT, fg="white", width=15,
                 command=self._create_main_menu).pack(side=tk.LEFT, padx=10)


# ======================================================================
# LAUNCH
# ======================================================================
def main():
    root = tk.Tk()
    app = ValorantAgent(root)
    root.mainloop()

if __name__ == "__main__":
    main()
