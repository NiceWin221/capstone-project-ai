# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import time
from collections import deque
import heapq
from sklearn.linear_model import LinearRegression

# Set Page Config
st.set_page_config(
    page_title="Smart Agent Navigation & Strategy Game",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F60 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #31333F;
        border-left: 4px solid #FF4B4B;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .grid-container {
        display: grid;
        grid-template-columns: repeat(5, 60px);
        grid-gap: 8px;
        justify-content: center;
        margin: 20px auto;
        background: #F0F2F6;
        padding: 15px;
        border-radius: 12px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        width: 332px;
    }
    
    .grid-cell {
        width: 60px;
        height: 60px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .cell-normal { background-color: #FFFFFF; border: 1px solid #E0E2E6; }
    .cell-start { background-color: #4CAF50; color: white; }
    .cell-goal { background-color: #E91E63; color: white; }
    .cell-obstacle { background-color: #9E9E9E; color: white; }
    .cell-obstacle-wet { background-color: #3F51B5; color: white; }
    .cell-path { background-color: #FFEB3B; border: 2px solid #FFC107; transform: scale(1.05); }
    .cell-visited { background-color: #E3F2FD; border: 1px dashed #2196F3; }
    
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 4px solid #FF4B4B;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: #FF4B4B;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #757575;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions & Classes -----------------

# 1. State Space Definition Formulation (for documentation)
STATE_SPACE_DOC = """
### Definisi Formal Ruang Keadaan (State Space) 5 Komponen:
1. **Initial State (Keadaan Awal)**:
   Posisi awal agen di grid $5 \\times 5$, direpresentasikan sebagai koordinat $(x, y) = (x_{start}, y_{start})$ (Gudang).
2. **Actions (Aksi)**:
   Kumpulan pergerakan yang valid dari state $s = (x, y)$:
   $$\\text{Actions}(s) = \\{ \\text{Up, Down, Left, Right} \\}$$
   dengan batasan bahwa pergerakan tidak keluar dari batas grid $[0, 4] \\times [0, 4]$.
3. **Transition Model (Model Transisi)**:
   Fungsi hasil aksi yang memindahkan koordinat:
   - $\\text{Up}: (x, y) \\to (x-1, y)$
   - $\\text{Down}: (x, y) \\to (x+1, y)$
   - $\\text{Left}: (x, y) \\to (x, y-1)$
   - $\\text{Right}: (x, y) \\to (x, y+1)$
4. **Goal Test (Uji Tujuan)**:
   Kondisi pengecekan apakah state saat ini $s$ sama dengan posisi Konsumen:
   $$\\text{GoalTest}(s) \\iff s == (x_{goal}, y_{goal})$$
5. **Path Cost (Biaya Jalur)**:
   Total akumulasi biaya langkah sepanjang rute. Setiap pergerakan dasar bernilai $1$, namun sel dengan rintangan/kemacetan mendapatkan penalti tambahan yang dipengaruhi oleh probabilitas cuaca (Bayesian Cost).
"""

# Bayesian Weather Probability Calculations
def get_move_cost(is_obstacle, weather):
    """
    Menghitung biaya gerakan berdasarkan status rintangan dan cuaca menggunakan konsep probabilitas Bayes sederhana.
    P(Macet | Cuaca, Rintangan)
    - Jika rintangan & Hujan: Probabilitas macet sangat tinggi (0.9), Penalti biaya = 5
    - Jika rintangan & Cerah: Probabilitas macet sedang (0.5), Penalti biaya = 2.5
    - Jika normal & Hujan: Probabilitas macet rendah (0.2), Penalti biaya = 1.2
    - Jika normal & Cerah: Probabilitas macet sangat rendah (0.05), Penalti biaya = 1.0
    """
    if is_obstacle:
        p_macet = 0.9 if weather == "Hujan" else 0.5
        penalty = 5.0 * p_macet
    else:
        p_macet = 0.2 if weather == "Hujan" else 0.05
        penalty = 2.0 * p_macet
    return 1.0 + penalty

# Pathfinding Algorithms
def bfs_search(start, goal, obstacles, grid_size=5):
    queue = deque([[start]])
    visited = set([start])
    explored_nodes = 0
    
    while queue:
        path = queue.popleft()
        node = path[-1]
        explored_nodes += 1
        
        if node == goal:
            return path, explored_nodes, visited
            
        x, y = node
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                # BFS standard ignores obstacle cost but cannot pass hard walls if we treat obstacles as blockades, 
                # or it can pass with high cost. Let's assume obstacles are passable but high cost, 
                # or blocked entirely. Let's make it passable but high cost, allowing comparison.
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(path + [(nx, ny)])
    return [], explored_nodes, visited

def a_star_search(start, goal, obstacles, weather, grid_size=5):
    # priority queue stores: (f_score, g_score, path)
    # h = manhattan distance
    h = lambda pos: abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    pq = [(h(start), 0, [start])]
    visited = {}
    explored_nodes = 0
    all_visited = set()
    
    while pq:
        f, g, path = heapq.heappop(pq)
        node = path[-1]
        explored_nodes += 1
        all_visited.add(node)
        
        if node == goal:
            return path, explored_nodes, all_visited
            
        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        
        x, y = node
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                is_obstacle = (nx, ny) in obstacles
                step_cost = get_move_cost(is_obstacle, weather)
                next_g = g + step_cost
                next_f = next_g + h((nx, ny))
                heapq.heappush(pq, (next_f, next_g, path + [(nx, ny)]))
                
    return [], explored_nodes, all_visited

# Machine Learning Battery Predictor Model
@st.cache_resource
def train_battery_model():
    # Synthetic dataset creation
    # Features: [Path Length, Weather_Code (0=Cerah, 1=Hujan), Congestion_Penalty]
    # Target: Battery Consumption (percentage)
    np.random.seed(42)
    X = np.random.rand(100, 3) * [10, 1, 5]  # Path length up to 10, weather 0 or 1, penalty up to 5
    # True relationship: Battery_Consumed = 4 * Path_Length + 3 * Weather + 2 * Congestion + noise
    y = 4 * X[:, 0] + 3 * X[:, 1] + 2 * X[:, 2] + np.random.randn(100) * 1.5
    model = LinearRegression()
    model.fit(X, y)
    return model

# Tic-Tac-Toe Minimax with Alpha-Beta Pruning
def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != "":
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != "":
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != "":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != "":
        return board[0][2]
    if all(cell != "" for row in board for cell in row):
        return "Tie"
    return None

def minimax(board, depth, is_maximizing, alpha, beta):
    winner = check_winner(board)
    if winner == "O": # AI
        return 10 - depth
    if winner == "X": # Human
        return depth - 10
    if winner == "Tie":
        return 0
        
    if is_maximizing:
        best_score = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = "O"
                    score = minimax(board, depth + 1, False, alpha, beta)
                    board[i][j] = ""
                    best_score = max(score, best_score)
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break # Beta pruning
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = "X"
                    score = minimax(board, depth + 1, True, alpha, beta)
                    board[i][j] = ""
                    best_score = min(score, best_score)
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break # Alpha pruning
        return best_score

def find_best_move(board):
    best_score = -float('inf')
    best_move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == "":
                board[i][j] = "O"
                score = minimax(board, 0, False, -float('inf'), float('inf'))
                board[i][j] = ""
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
    return best_move

# ----------------- App Layout & Control Flow -----------------

st.markdown('<div class="main-title">Aplikasi Web Navigasi Agen Cerdas & Mini-Game Strategi</div>', unsafe_allow_html=True)

# Initialize Session States
if "ttt_board" not in st.session_state:
    st.session_state.ttt_board = [["" for _ in range(3)] for _ in range(3)]
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "agent_arrived" not in st.session_state:
    st.session_state.agent_arrived = False

# Sidebar Controls
st.sidebar.markdown("### Pengaturan Simulasi")
weather = st.sidebar.selectbox("Simulasi Cuaca", ["Cerah", "Hujan"])

st.sidebar.markdown("### Posisi Grid (0-4)")
start_x = st.sidebar.slider("Gudang (Start) X", 0, 4, 0)
start_y = st.sidebar.slider("Gudang (Start) Y", 0, 4, 0)
goal_x = st.sidebar.slider("Konsumen (Goal) X", 0, 4, 4)
goal_y = st.sidebar.slider("Konsumen (Goal) Y", 0, 4, 4)

obstacles = [(1, 2), (2, 2), (3, 2), (2, 1), (2, 3)]

tab1, tab2 = st.tabs(["🗺️ Navigasi Agen & Prediksi", "🎮 Mini-Game Tic-Tac-Toe"])

# Prepare calculations
start_node = (start_x, start_y)
goal_node = (goal_x, goal_y)

# Run algorithms
bfs_path, bfs_explored, bfs_visited = bfs_search(start_node, goal_node, obstacles)
astar_path, astar_explored, astar_visited = a_star_search(start_node, goal_node, obstacles, weather)

# Train ML battery model
battery_model = train_battery_model()

with tab1:
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown('<div class="sub-title">Visualisasi Grid 5x5 (A* Path)</div>', unsafe_allow_html=True)
        
        # Calculate visual grid state
        grid_html = '<div class="grid-container">'
        for r in range(5):
            for c in range(5):
                cell_class = "cell-normal"
                cell_char = "•"
                
                if (r, c) == start_node:
                    cell_class = "cell-start"
                    cell_char = "🏠"
                elif (r, c) == goal_node:
                    cell_class = "cell-goal"
                    cell_char = "📦"
                elif (r, c) in astar_path:
                    cell_class = "cell-path"
                    cell_char = "🤖"
                elif (r, c) in obstacles:
                    cell_class = "cell-obstacle-wet" if weather == "Hujan" else "cell-obstacle"
                    cell_char = "⚠️"
                elif (r, c) in astar_visited:
                    cell_class = "cell-visited"
                    cell_char = "◦"
                    
                grid_html += f'<div class="grid-cell {cell_class}">{cell_char}</div>'
        grid_html += '</div>'
        
        st.markdown(grid_html, unsafe_allow_html=True)
        
        # Trigger mini-game if path found
        if astar_path:
            st.session_state.agent_arrived = True
            st.success("Agen Cerdas telah sampai di Konsumen! Modul Mini-Game diaktifkan di Tab sebelah.")
        else:
            st.error("Jalur tidak ditemukan!")
            
    with col2:
        st.markdown('<div class="sub-title">Perbandingan Performansi & Evaluasi</div>', unsafe_allow_html=True)
        
        # Comparative Table
        astar_cost = sum(get_move_cost((r, c) in obstacles, weather) for r, c in astar_path) if astar_path else 0
        perf_data = {
            "Metrik": ["Node Dieksplorasi", "Panjang Jalur (Langkah)", "Estimasi Total Biaya"],
            "BFS (Pencarian Buta)": [
                f"{int(bfs_explored)}",
                f"{int(len(bfs_path))}",
                f"{float(len(bfs_path)):.1f}"
            ],
            "A* Search (Berpetunjuk)": [
                f"{int(astar_explored)}", 
                f"{int(len(astar_path))}", 
                f"{float(astar_cost):.1f}"
            ]
        }
        st.table(pd.DataFrame(perf_data))
        
        # Bayesian Weather Probability info box
        st.info(f"**Bayesian Decision:** Cuaca saat ini **{weather}**. Rintangan (⚠️) memiliki penalty cost sebesar **{get_move_cost(True, weather)}** per langkah sedangkan jalur normal memiliki cost **{get_move_cost(False, weather)}**.")
        
        # Machine Learning Battery Prediction
        st.markdown('<div class="sub-title">Prediksi Energi Agen (Machine Learning)</div>', unsafe_allow_html=True)
        if astar_path:
            path_len = len(astar_path)
            weather_code = 1 if weather == "Hujan" else 0
            avg_congestion = sum(get_move_cost((r, c) in obstacles, weather) for r, c in astar_path) / path_len
            
            # Predict consumption
            features = np.array([[path_len, weather_code, avg_congestion]])
            predicted_consumption = battery_model.predict(features)[0]
            remaining_battery = max(0.0, min(100.0, 100.0 - predicted_consumption))
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{predicted_consumption:.1f}%</div><div class="stat-label">Konsumsi Energi</div></div>', unsafe_allow_html=True)
            with col_b2:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{remaining_battery:.1f}%</div><div class="stat-label">Sisa Baterai Agen</div></div>', unsafe_allow_html=True)
        else:
            st.warning("Tidak ada jalur A* untuk memprediksi baterai.")

    st.markdown(STATE_SPACE_DOC)

with tab2:
    st.markdown('<div class="sub-title">Tic-Tac-Toe vs AI Bot (Minimax + Alpha-Beta)</div>', unsafe_allow_html=True)
    
    if not st.session_state.agent_arrived:
        st.warning("Simulasi pergerakan agen ke tujuan (Goal) di Tab 1 untuk membuka mini-game ini!")
    else:
        # Centering the board and limiting its width to make it compact
        col_left, col_mid, col_right = st.columns([1.5, 1, 1.5])
        
        with col_mid:
            # Custom style for Tic-Tac-Toe cells
            st.markdown("""
                <style>
                /* Style only the cells of the game board */
                div[data-testid="stHorizontalBlock"] div[data-testid="element-container"] button[key^="cell_"] {
                    height: 90px !important;
                    font-size: 2.2rem !important;
                    font-weight: 800 !important;
                    margin: 5px 0 !important;
                }
                /* Alternate styling for O and X if possible or just standard */
                </style>
            """, unsafe_allow_html=True)
            
            # Reset game button
            if st.button("Reset Game", use_container_width=True):
                st.session_state.ttt_board = [["" for _ in range(3)] for _ in range(3)]
                st.session_state.game_over = False
                st.rerun()
            
            st.write("") # Spacer
            
            # UI for game board
            cols = st.columns(3)
            for i in range(3):
                for j in range(3):
                    cell_value = st.session_state.ttt_board[i][j]
                    btn_label = cell_value if cell_value != "" else " "
                    
                    # Highlight colors depending on X or O
                    # Check for action
                    if cols[j].button(
                        f"{btn_label}", 
                        key=f"cell_{i}_{j}", 
                        disabled=st.session_state.game_over or cell_value != "",
                        use_container_width=True
                    ):
                        # Player Move
                        st.session_state.ttt_board[i][j] = "X"
                        winner = check_winner(st.session_state.ttt_board)
                        
                        if winner:
                            st.session_state.game_over = True
                        else:
                            # AI Move using Minimax with Alpha-Beta
                            ai_move = find_best_move(st.session_state.ttt_board)
                            if ai_move:
                                st.session_state.ttt_board[ai_move[0]][ai_move[1]] = "O"
                                winner = check_winner(st.session_state.ttt_board)
                                if winner:
                                    st.session_state.game_over = True
                        st.rerun()
            
            st.write("") # Spacer
            
            # Display outcome
            winner = check_winner(st.session_state.ttt_board)
            if winner:
                if winner == "Tie":
                    st.info("Hasil seri!")
                elif winner == "X":
                    st.success("Hebat! Anda menang!")
                else:
                    st.error("AI memenangkan game!")
                
        # Theoretical Pruning Analysis
        st.markdown("""
        ### Analisis Alpha-Beta Pruning pada Game Tic-Tac-Toe
        Pemangkasan komputasi (pruning) terjadi pada cabang-cabang pohon pencarian ketika AI mendeteksi bahwa langkah alternatif di masa depan dijamin bernilai lebih buruk dibandingkan dengan opsi yang sudah dievaluasi sebelumnya. Sebagai contoh, ketika AI mengevaluasi langkah meminimalkan (giliran pemain manusia) dan menemukan skenario di mana manusia dapat langsung menang (skor negatif), nilai beta dari cabang tersebut langsung mengecil. Jika nilai beta ini kurang dari atau sama dengan nilai alpha dari langkah memaksimalkan (giliran AI) di level atasnya, cabang evaluasi selanjutnya di bawah node tersebut langsung dihentikan/dipotong (pruned). Hal ini menghemat operasi rekursif pencarian game tree secara masif terutama di langkah-langkah awal game.
        """)
