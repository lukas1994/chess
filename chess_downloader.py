import requests
import sqlite3
from datetime import datetime
import time
import os
import chess
import chess.engine
import chess.pgn
import io
from openai import OpenAI
import json

class ChessComDownloader:
    def __init__(self, username):
        self.username = username
        self.base_url = "https://api.chess.com/pub"
        self.db_path = "chess_games.db"
        # Add headers for API requests
        self.headers = {
            'User-Agent': 'Chess.com Game Downloader (Python/requests)',
            'Accept': 'application/json'
        }
        # Use Homebrew-installed Stockfish
        self.engine_path = "/opt/homebrew/bin/stockfish"  # Path to Homebrew Stockfish
        
        # Create database and tables immediately
        self.setup_database()

    def setup_database(self):
        """Create the database and tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_url TEXT UNIQUE,
                white_player TEXT,
                black_player TEXT,
                result TEXT,
                timestamp INTEGER,
                time_control TEXT,
                white_rating INTEGER,
                black_rating INTEGER,
                pgn TEXT,
                time_class TEXT,
                rules TEXT,
                white_accuracy REAL,
                black_accuracy REAL,
                initial_setup TEXT,
                fen TEXT,
                tcn TEXT,
                uuid TEXT,
                rated BOOLEAN,
                white_result TEXT,
                black_result TEXT,
                termination TEXT,
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add table for engine analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS engine_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                move_number INTEGER,
                ply INTEGER,
                move_san TEXT,
                evaluation INTEGER,  -- in centipawns
                mate INTEGER,        -- number of moves to mate (if exists)
                FOREIGN KEY(game_id) REFERENCES games(id),
                UNIQUE(game_id, ply)
            )
        ''')
        
        # Add table for LLM analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                opening TEXT,
                blunders TEXT,  -- Store as JSON array
                common_mistakes TEXT,  -- Store as JSON array
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(game_id) REFERENCES games(id),
                UNIQUE(game_id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def get_monthly_archives(self):
        """Get list of monthly archive URLs for the user"""
        archives_url = f"{self.base_url}/player/{self.username}/games/archives"
        print(f"Fetching archives from: {archives_url}")
        response = requests.get(archives_url, headers=self.headers)
        if response.status_code == 200:
            archives = response.json()['archives']
            print(f"API Response: {response.json()}")
            return archives
        else:
            print(f"Error fetching archives. Status code: {response.status_code}")
            print(f"Response: {response.text}")
        return []

    def download_monthly_games(self, archive_url):
        """Download games from a monthly archive"""
        print(f"Fetching games from: {archive_url}")
        response = requests.get(archive_url, headers=self.headers)
        if response.status_code == 200:
            games = response.json()['games']
            print(f"Found {len(games)} games in archive")
            print(games[0])
            return games
        else:
            print(f"Error downloading games. Status code: {response.status_code}")
            print(f"Response: {response.text}")
        return []

    def map_result(self, game):
        """Map Chess.com result codes to standard notation"""
        white_result = game['white']['result']
        
        # Standard win/loss
        if white_result == 'win':
            return '1-0'
        elif game['black']['result'] == 'win':
            return '0-1'
            
        # Various draw conditions
        if white_result in ['agreed', 'repetition', 'stalemate', '50move', 'insufficient', 'timevsinsufficient']:
            return '1/2-1/2'
            
        # Special cases where we need to determine winner
        if white_result in ['timeout', 'resigned', 'abandoned', 'checkmated', 'lose', 'kingofthehill', 'threecheck', 'bughousepartnerlose']:
            return '0-1'  # Black wins
        elif game['black']['result'] in ['timeout', 'resigned', 'abandoned', 'checkmated', 'lose', 'kingofthehill', 'threecheck', 'bughousepartnerlose']:
            return '1-0'  # White wins
            
        # If we get here, it's an unknown result code
        print(f"Unknown result code: {white_result}")
        return white_result

    def save_games_to_db(self, games):
        """Save games to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for game in games:
            try:
                result = self.map_result(game)

                # Extract termination from PGN
                termination = None
                for line in game['pgn'].split('\n'):
                    if '[Termination' in line:
                        termination = line.split('"')[1]
                        break

                cursor.execute('''
                    INSERT OR IGNORE INTO games 
                    (game_url, white_player, black_player, result, timestamp,
                     time_control, white_rating, black_rating, pgn,
                     time_class, rules, white_accuracy, black_accuracy,
                     initial_setup, fen, tcn, uuid, rated,
                     white_result, black_result, termination)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    game['url'],
                    game['white']['username'],
                    game['black']['username'],
                    result,
                    game['end_time'],
                    game['time_control'],
                    game['white']['rating'],
                    game['black']['rating'],
                    game['pgn'],
                    game.get('time_class'),
                    game.get('rules'),
                    game.get('accuracies', {}).get('white'),
                    game.get('accuracies', {}).get('black'),
                    game.get('initial_setup'),
                    game.get('fen'),
                    game.get('tcn'),
                    game.get('uuid'),
                    game.get('rated', True),
                    game['white']['result'],
                    game['black']['result'],
                    termination
                ))
                print(f"Added game to DB: {game['white']['username']} vs {game['black']['username']} ({result})")

            except KeyError as e:
                print(f"Skipping game due to missing data: {e}")
                print(f"Game data: {game}")
                continue
        
        conn.commit()
        conn.close()

    def download_all_games(self, limit=None):
        """Main function to download games"""
        print(f"Starting download for user: {self.username}" + 
              f" (limit: {limit if limit else 'unlimited'} games)")
        
        archives = self.get_monthly_archives()
        total_archives = len(archives)
        
        if total_archives == 0:
            print("No archives found. Please check if the username is correct.")
            return
            
        print(f"Found {total_archives} monthly archives")
        
        total_games = 0
        # Process archives in reverse order to get most recent games first
        for i, archive_url in enumerate(reversed(archives), 1):
            if limit and total_games >= limit:
                break
                
            print(f"\nProcessing archive {i}: {archive_url}")
            games = self.download_monthly_games(archive_url)
            
            if games:
                # Calculate how many more games we need if there's a limit
                if limit:
                    games_needed = limit - total_games
                    games_to_save = games[:games_needed]
                else:
                    games_to_save = games
                
                self.save_games_to_db(games_to_save)
                total_games += len(games_to_save)
                print(f"Saved {len(games_to_save)} games from archive")
            else:
                print("No games found in this archive")
            
            # Be nice to the API - add a small delay between requests
            time.sleep(1)
        
        print(f"\nDownload completed! Total games downloaded: {total_games}")

    def print_recent_games(self, limit=5):
        """Print the most recent games from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT white_player, black_player, result, 
                   datetime(timestamp, 'unixepoch') as game_date,
                   white_rating, black_rating, time_control
            FROM games 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        games = cursor.fetchall()
        
        print("\nMost recent games:")
        print("-" * 80)
        for game in games:
            white, black, result, date, w_rating, b_rating, time_control = game
            print(f"Date: {date}")
            print(f"White: {white} ({w_rating}) vs Black: {black} ({b_rating})")
            print(f"Result: {result}")
            print(f"Time Control: {time_control}")
            print("-" * 80)
        
        conn.close()

    def delete_database(self):
        """Delete the database file if it exists"""
        try:
            # Close any existing connections first
            try:
                conn = sqlite3.connect(self.db_path)
                conn.close()
            except:
                pass
            os.remove(self.db_path)
            print(f"Successfully deleted database: {self.db_path}")
        except FileNotFoundError:
            print(f"Database {self.db_path} does not exist")
        # Recreate the database and tables after deletion
        self.setup_database()

    def count_total_games(self):
        """Return the total number of games in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM games')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def analyze_game(self, game_id, pgn_text, depth=20, time_per_move=0.2):
        """Analyze a game with Stockfish
        
        Args:
            game_id (int): ID of the game to analyze
            pgn_text (str): PGN text of the game
            depth (int): Analysis depth for Stockfish
            time_per_move (float): Time in seconds to spend analyzing each move
        """
        engine = None
        conn = None
        try:
            print(f"\nStarting analysis of game {game_id}")
            engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            engine.configure({"Threads": 2, "Hash": 128})
            print(f"Engine initialized: {self.engine_path}")

            game = chess.pgn.read_game(io.StringIO(pgn_text))
            if not game:
                print(f"Could not parse PGN for game {game_id}")
                return

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Log game headers
            print(f"Game: {game.headers.get('White')} vs {game.headers.get('Black')}")
            print(f"Opening: {game.headers.get('ECO', 'Unknown')} - {game.headers.get('Opening', 'Unknown')}")
            print("\nAnalyzing positions...")

            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                move_san = board.san(move)
                board.push(move)
                
                try:
                    print(f"\nAnalyzing move {i+1} ({move_san}):")
                    info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_per_move))
                    
                    # Get score from White's perspective
                    score = info["score"].white()
                    mate = None
                    eval_cp = None
                    
                    if score.is_mate():
                        mate = score.mate()
                        print(f"  Mate in {mate}")
                    else:
                        eval_cp = score.score()
                        print(f"  Evaluation: {eval_cp/100:.2f}")

                    # Log position details
                    print(f"  Move number: {(i // 2) + 1}")
                    print(f"  Color to move: {'Black' if i % 2 else 'White'}")
                    print(f"  FEN: {board.fen()}")

                    cursor.execute('''
                        INSERT OR REPLACE INTO engine_analysis
                        (game_id, move_number, ply, move_san, evaluation, mate)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        game_id,
                        (i // 2) + 1,  # move number
                        i + 1,         # ply
                        move.uci(),
                        eval_cp,
                        mate
                    ))
                    
                    conn.commit()
                    
                except Exception as e:
                    print(f"Error analyzing move {i+1}: {str(e)}")
                    continue

            print(f"\nCompleted analysis of game {game_id}")

        except Exception as e:
            print(f"Error analyzing game {game_id}: {str(e)}")
            
        finally:
            if engine:
                try:
                    engine.quit()
                    print("Engine closed")
                except:
                    pass
            if conn:
                try:
                    conn.close()
                    print("Database connection closed")
                except:
                    pass

    def analyze_all_games(self, depth=20, time_per_move=1.0, limit=None):
        """Analyze recent games in the database that haven't been analyzed yet
        
        Args:
            depth (int): Analysis depth for Stockfish
            time_per_move (float): Time in seconds to spend analyzing each move
            limit (int, optional): Maximum number of games to analyze. If None, analyze all games.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unanalyzed games from the last 30 days
        cursor.execute('''
            SELECT DISTINCT g.id, g.pgn 
            FROM games g
            LEFT JOIN engine_analysis ea ON g.id = ea.game_id
            WHERE ea.id IS NULL  -- Only get games without any analysis
            AND g.timestamp >= strftime('%s', 'now', '-30 days')
            AND g.rules = 'chess'
            AND g.rated = 1
            AND NOT EXISTS (  -- Ensure no moves have been analyzed for this game
                SELECT 1 FROM engine_analysis 
                WHERE game_id = g.id
            )
            ORDER BY g.timestamp DESC
            LIMIT ?
        ''', (limit if limit else -1,))
        
        games = cursor.fetchall()
        conn.close()
        
        print(f"Found {len(games)} unanalyzed recent games" + 
              f" (limit: {limit if limit else 'unlimited'})")
        
        for game_id, pgn in games:
            print(f"\nAnalyzing game {game_id}...")
            self.analyze_game(game_id, pgn, depth, time_per_move)

    def print_game_with_analysis(self, game_id):
        """Return a game with move by move evaluations and timings as a string"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get game details and analysis
        cursor.execute('''
            SELECT g.white_player, g.black_player, g.pgn,
                   ea.move_number, ea.move_san, ea.evaluation, ea.mate, ea.ply
            FROM games g
            LEFT JOIN engine_analysis ea ON g.id = ea.game_id
            WHERE g.id = ?
            ORDER BY ea.ply
        ''', (game_id,))
        
        rows = cursor.fetchall()
        if not rows:
            return
            
        # Build output string
        output = []
        
        # Add game header
        white_player = rows[0][0]
        black_player = rows[0][1]
        output.append(f"\n{white_player} vs {black_player}\n")
        
        # Parse the PGN to get moves in standard algebraic notation and timings
        game = chess.pgn.read_game(io.StringIO(rows[0][2]))
        if not game:
            return
            
        # Create a board to track position
        board = game.board()
        moves = list(game.mainline_moves())
        node = game
        
        # Group moves by move number
        current_move = 1
        move_pair = []
        
        for i, row in enumerate(rows):
            _, _, _, move_number, uci_move, eval_cp, mate, _ = row
            
            # Convert UCI move to SAN
            move = moves[i]
            san_move = board.san(move)
            
            # Get timing info from PGN
            node = node.variation(0)
            clock = node.clock() if node.clock() is not None else "?"
            
            # Format evaluation and clock
            if mate is not None:
                eval_str = f"M{mate}"
            elif eval_cp is not None:
                eval_str = f"{eval_cp/100:.1f}"
            else:
                eval_str = "?"
                
            move_pair.append(f"{san_move} ({eval_str}) [{clock}s]")
            board.push(move)
            
            # Add completed move pairs to output
            if len(move_pair) == 2:
                output.append(f"{move_number}. {move_pair[0]} {move_pair[1]}")
                move_pair = []
                current_move += 1
                
        # Add final unpaired move if any
        if move_pair:
            output.append(f"{current_move}. {move_pair[0]}")
            
        conn.close()
        
        return "\n".join(output)

    def get_game_analysis_for_llm(self, game_id):
        """Get LLM analysis of a game"""
        # Get the game with analysis
        game_text = self.print_game_with_analysis(game_id)
        if not game_text:
            return None
            
        # Get game headers and player info
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT pgn, white_player, black_player, game_url, 
                   white_rating, black_rating
            FROM games 
            WHERE id = ?
        ''', (game_id,))
        
        result = cursor.fetchone()
        if not result:
            print(f"Game {game_id} not found or not a standard rated chess game")
            return None
            
        pgn_text, white_player, black_player, game_url, white_rating, black_rating = result
        conn.close()
        
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        
        # Determine which side we're analyzing
        analyzing_color = "White" if white_player == self.username else "Black"
        
        # Format prompt for LLM
        prompt = f"""Analyze this chess game from {analyzing_color}'s perspective.

Game Information:
Analyzing: {analyzing_color}
White Rating: {white_rating}
Black Rating: {black_rating}
ECO Code: {game.headers.get('ECO', 'Unknown')}
Time Control: {game.headers.get('TimeControl', 'Unknown')}
Game URL: {game_url}

Game moves with evaluations and time remaining:
{game_text}

Please analyze {analyzing_color}'s play and provide a response in this exact JSON format:
{{
    "opening": "<translate the ECO code to standard opening name>",
    "blunders": [<list of move numbers where {analyzing_color} made major mistakes>],
    "common_mistakes": [<list of mistake categories from the predefined list below, max 3>]
}}

Choose from these mistake categories only:
1. "poor_time_management" - Using too much time early or getting into time trouble
2. "weak_pawn_structure" - Creating permanent pawn weaknesses
3. "piece_coordination" - Poor piece placement and lack of coordination
4. "tactical_oversight" - Missing tactical opportunities or falling for tactics
5. "positional_weakness" - Allowing opponent to gain lasting positional advantage
6. "endgame_technique" - Poor technique in converting advantages or defending endgames
7. "opening_principles" - Violating basic opening principles
8. "king_safety" - Neglecting king safety or poor king positioning
9. "missed_opportunities" - Not capitalizing on advantageous positions
10. "defensive_technique" - Poor defense or missing defensive resources

Focus on:
1. Major evaluation swings (>1.5 pawns) that indicate mistakes by {analyzing_color}
2. {analyzing_color}'s time management patterns
3. Tactical or positional mistakes in {analyzing_color}'s play
4. Common themes in {analyzing_color}'s mistakes

Provide the response as valid JSON only, no additional text. The common_mistakes array must only contain items from the predefined list."""

        return prompt

    def analyze_game_with_llm(self, game_id):
        """Get LLM analysis of a game using OpenAI and store in database"""
        prompt = self.get_game_analysis_for_llm(game_id)
        if not prompt:
            print(f"Could not analyze game {game_id}")
            return
            
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are a chess analysis assistant that provides analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO llm_analysis
                (game_id, opening, blunders, common_mistakes)
                VALUES (?, ?, ?, ?)
            ''', (
                game_id,
                analysis['opening'],
                json.dumps(analysis['blunders']),
                json.dumps(analysis['common_mistakes'])
            ))
            
            conn.commit()
            conn.close()
            
            print("\nAnalysis:")
            print(json.dumps(analysis, indent=2))
            return analysis
            
        except Exception as e:
            print(f"Error with LLM analysis: {str(e)}")
            return None

    def clear_analysis(self):
        """Clear all engine analysis from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM engine_analysis')
            deleted_count = cursor.rowcount
            conn.commit()
            print(f"Cleared {deleted_count} analysis records from database")
        except Exception as e:
            print(f"Error clearing analysis: {str(e)}")
        finally:
            conn.close()

    def analyze_all_games_with_llm(self, game_id=None):
        """Analyze all games with engine analysis using LLM
        
        Args:
            game_id (int, optional): If provided, analyze only this game.
                                   If None, analyze all games with engine analysis but no LLM analysis.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if game_id:
            games_to_analyze = [(game_id,)]
        else:
            # Get games that have engine analysis but no LLM analysis
            cursor.execute('''
                SELECT DISTINCT g.id
                FROM games g
                JOIN engine_analysis ea ON g.id = ea.game_id
                LEFT JOIN llm_analysis la ON g.id = la.game_id
                WHERE g.rules = 'chess' 
                AND g.rated = 1
                AND la.id IS NULL
                ORDER BY g.timestamp DESC
            ''')
            games_to_analyze = cursor.fetchall()
        
        conn.close()
        
        print(f"Found {len(games_to_analyze)} games to analyze with LLM")
        
        for (game_id,) in games_to_analyze:
            print(f"\nAnalyzing game {game_id}...")
            self.analyze_game_with_llm(game_id)
            # Add a small delay to avoid rate limits
            time.sleep(1)

    def generate_html_report(self):
        """Generate an HTML report with statistics and visualizations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all the statistics we need
        # Overall record
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN (white_player = ? AND result = '1-0') OR 
                         (black_player = ? AND result = '0-1') THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = '1/2-1/2' THEN 1 ELSE 0 END) as draws,
                SUM(CASE WHEN (white_player = ? AND result = '0-1') OR 
                         (black_player = ? AND result = '1-0') THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN white_player = ? THEN white_rating ELSE black_rating END) as avg_rating
            FROM games 
            WHERE rules = 'chess' AND rated = 1
        ''', (self.username,) * 5)
        
        total, wins, draws, losses, avg_rating = cursor.fetchone()
        
        # Record by color
        cursor.execute('''
            SELECT 
                'White' as color,
                COUNT(*) as total,
                SUM(CASE WHEN result = '1-0' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = '1/2-1/2' THEN 1 ELSE 0 END) as draws,
                SUM(CASE WHEN result = '0-1' THEN 1 ELSE 0 END) as losses
            FROM games 
            WHERE white_player = ? AND rules = 'chess' AND rated = 1
            UNION ALL
            SELECT 
                'Black' as color,
                COUNT(*) as total,
                SUM(CASE WHEN result = '0-1' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = '1/2-1/2' THEN 1 ELSE 0 END) as draws,
                SUM(CASE WHEN result = '1-0' THEN 1 ELSE 0 END) as losses
            FROM games 
            WHERE black_player = ? AND rules = 'chess' AND rated = 1
        ''', (self.username, self.username))
        
        color_stats = cursor.fetchall()
        
        # Most common openings
        cursor.execute('''
            SELECT opening, COUNT(*) as count
            FROM llm_analysis
            GROUP BY opening
            ORDER BY count DESC
            LIMIT 5
        ''')
        openings = cursor.fetchall()
        
        # Get mistake statistics
        cursor.execute('SELECT common_mistakes FROM llm_analysis')
        mistake_counts = {}
        for (mistakes_json,) in cursor.fetchall():
            if mistakes_json:
                mistakes = json.loads(mistakes_json)
                for mistake in mistakes:
                    mistake_counts[mistake] = mistake_counts.get(mistake, 0) + 1
        
        # Blunder timing
        cursor.execute('SELECT blunders FROM llm_analysis')
        move_counts = {}
        for (blunders_json,) in cursor.fetchall():
            if blunders_json:
                blunders = json.loads(blunders_json)
                for move in blunders:
                    move_range = f"Moves {(move//10)*10+1}-{(move//10)*10+10}"
                    move_counts[move_range] = move_counts.get(move_range, 0) + 1
        
        # Get time control distribution data
        cursor.execute('''
            SELECT time_class, COUNT(*) as count
            FROM games
            WHERE rules = 'chess' AND rated = 1
            GROUP BY time_class
            ORDER BY count DESC
        ''')
        time_control_stats = cursor.fetchall()

        # Build chart data
        time_control_data = {
            'labels': [tc[0].title() for tc in time_control_stats],
            'data': [tc[1] for tc in time_control_stats]
        }

        # Add highest rating query
        cursor.execute('''
            SELECT 
                time_class,
                MAX(CASE WHEN white_player = ? THEN white_rating 
                        WHEN black_player = ? THEN black_rating END) as peak_rating
            FROM games
            WHERE rules = 'chess' AND rated = 1
            GROUP BY time_class
        ''', (self.username, self.username))
        
        peak_ratings = cursor.fetchall()
        
        # Get rating history by time control
        cursor.execute('''
            SELECT 
                time_class,
                timestamp,
                CASE WHEN white_player = ? THEN white_rating 
                     WHEN black_player = ? THEN black_rating END as rating
            FROM games
            WHERE rules = 'chess' AND rated = 1
            ORDER BY timestamp
        ''', (self.username, self.username))
        
        rating_history = cursor.fetchall()
        
        # Process rating history into time control series
        rating_series = {}
        for time_class, timestamp, rating in rating_history:
            if time_class not in rating_series:
                rating_series[time_class] = {'timestamps': [], 'ratings': []}
            rating_series[time_class]['timestamps'].append(timestamp * 1000)  # Convert to milliseconds for JS
            rating_series[time_class]['ratings'].append(rating)

        # Time Management Performance
        cursor.execute('''
            SELECT 
                time_class,
                AVG(CASE WHEN white_player = ? THEN white_accuracy ELSE black_accuracy END) as avg_accuracy,
                COUNT(*) as games
            FROM games
            WHERE rules = 'chess' AND rated = 1 AND white_accuracy IS NOT NULL
            GROUP BY time_class
        ''', (self.username,))
        time_management = cursor.fetchall()

        # Performance by Rating Difference
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN rating_diff <= -200 THEN 'Much Weaker'
                    WHEN rating_diff <= -100 THEN 'Weaker'
                    WHEN rating_diff < 100 THEN 'Equal'
                    WHEN rating_diff < 200 THEN 'Stronger'
                    ELSE 'Much Stronger'
                END as opponent_strength,
                COUNT(*) as games,
                SUM(CASE WHEN won THEN 1 ELSE 0 END)*100.0/COUNT(*) as win_rate
            FROM (
                SELECT 
                    CASE WHEN white_player = ? THEN black_rating - white_rating
                         ELSE white_rating - black_rating
                    END as rating_diff,
                    CASE WHEN (white_player = ? AND result = '1-0') OR 
                              (black_player = ? AND result = '0-1') THEN 1
                         ELSE 0
                    END as won
                FROM games
                WHERE rules = 'chess' AND rated = 1
            )
            GROUP BY opponent_strength
            ORDER BY MIN(rating_diff)
        ''', (self.username, self.username, self.username))
        rating_performance = cursor.fetchall()

        # Time of Day Analysis
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN strftime('%H', datetime(timestamp, 'unixepoch')) BETWEEN '06' AND '11' THEN 'Morning'
                    WHEN strftime('%H', datetime(timestamp, 'unixepoch')) BETWEEN '12' AND '17' THEN 'Afternoon'
                    WHEN strftime('%H', datetime(timestamp, 'unixepoch')) BETWEEN '18' AND '23' THEN 'Evening'
                    ELSE 'Night'
                END as time_of_day,
                COUNT(*) as games,
                SUM(CASE WHEN (white_player = ? AND result = '1-0') OR 
                             (black_player = ? AND result = '0-1') THEN 1 ELSE 0 END)*100.0/COUNT(*) as win_rate
            FROM games
            WHERE rules = 'chess' AND rated = 1
            GROUP BY time_of_day
            ORDER BY 
                CASE time_of_day 
                    WHEN 'Morning' THEN 1 
                    WHEN 'Afternoon' THEN 2 
                    WHEN 'Evening' THEN 3 
                    ELSE 4 
                END
        ''', (self.username, self.username))
        time_of_day = cursor.fetchall()

        # Opening Win Rates
        cursor.execute('''
            SELECT 
                la.opening,
                COUNT(*) as games,
                SUM(CASE WHEN (g.white_player = ? AND g.result = '1-0') OR 
                         (g.black_player = ? AND g.result = '0-1') THEN 1 ELSE 0 END)*100.0/COUNT(*) as win_rate
            FROM llm_analysis la
            JOIN games g ON la.game_id = g.id
            WHERE la.opening IS NOT NULL  -- Only include games with analysis
            AND g.rules = 'chess' 
            AND g.rated = 1
            GROUP BY la.opening
            HAVING COUNT(*) >= 2  -- Changed from 5 to 2
            ORDER BY games DESC
            LIMIT 10
        ''', (self.username, self.username))
        opening_stats = cursor.fetchall()
        
        # Define charts section
        charts = f"""
    <div class="stats-grid">
        <div class="card">
            <h2>Rating History</h2>
            <div class="chart-container">
                <canvas id="ratingChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>Performance by Color</h2>
            <div class="chart-container">
                <canvas id="colorChart"></canvas>
            </div>
        </div>
    </div>

    <div class="stats-grid">
        <div class="card">
            <h2>Most Common Openings</h2>
            <div class="chart-container">
                <canvas id="openingsChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>Time Control Distribution</h2>
            <div class="chart-container">
                <canvas id="timeControlChart"></canvas>
            </div>
        </div>
    </div>

    <div class="stats-grid">
        <div class="card">
            <h2>Time Management Performance</h2>
            <div class="chart-container">
                <canvas id="timeManagementChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>Performance by Rating Difference</h2>
            <div class="chart-container">
                <canvas id="ratingDiffChart"></canvas>
            </div>
        </div>
    </div>

    <div class="stats-grid">
        <div class="card">
            <h2>Common Mistakes</h2>
            <div class="chart-container">
                <canvas id="mistakesChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>Blunder Timing</h2>
            <div class="chart-container">
                <canvas id="blunderChart"></canvas>
            </div>
        </div>
    </div>

    <div class="stats-grid">
        <div class="card">
            <h2>Time of Day Performance</h2>
            <div class="chart-container">
                <canvas id="timeOfDayChart"></canvas>
            </div>
        </div>
        <div class="card">
            <h2>Opening Win Rates</h2>
            <div class="chart-container">
                <canvas id="openingWinRatesChart"></canvas>
            </div>
        </div>
    </div>
"""

        # Create rating history datasets separately
        rating_datasets = []
        time_controls = ['bullet', 'blitz', 'rapid', 'daily', 'classical']
        colors = {
            'bullet': '#FF6B6B',
            'blitz': '#4ECDC4',
            'rapid': '#45B7D1',
            'daily': '#96CEB4',
            'classical': '#FFEEAD'
        }
        
        for tc in time_controls:
            if tc in rating_series:
                data_points = []
                for t, r in zip(rating_series[tc]['timestamps'], rating_series[tc]['ratings']):
                    data_points.append(f"{{x: {t}, y: {r}}}")
                
                dataset = f"""{{
                    label: '{tc.title()}',
                    data: [{','.join(data_points)}],
                    borderColor: '{colors[tc]}',
                    tension: 0.1,
                    fill: false
                }}"""
                rating_datasets.append(dataset)

        # Define header with CSS and chart.js includes
        header = f"""<!DOCTYPE html>
<html>
<head>
    <title>Chess Analysis Report - {self.username}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600&display=swap">
    <style>
        body {{ font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .subtitle {{ color: #666; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .chart-container {{ position: relative; height: 300px; margin: 20px 0; }}
        .stat-value {{ font-size: 24px; font-weight: 500; margin: 10px 0 5px; }}
        .stat-label {{ color: #666; margin: 0; }}
        .peak-ratings {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 20px; }}
        .peak-rating-item {{ text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Chess Analysis Report</h1>
        <p class="subtitle">{self.username}'s Performance Overview</p>
    </div>
"""

        # Define peak ratings section
        peak_ratings_html = """
        <div class="card">
            <h2>Peak Ratings</h2>
            <div class="peak-ratings">
"""
        for time_class, rating in peak_ratings:
            peak_ratings_html += f"""
                <div class="peak-rating-item">
                    <div class="stat-value">{rating}</div>
                    <p class="stat-label">{time_class.title()}</p>
                </div>
"""
        peak_ratings_html += """
            </div>
    </div>
"""
        # Define overall stats section
        overall_stats = f"""
    <div class="stats-grid">
        <div class="card">
            <h2>Overall Record</h2>
            <div class="chart-container">
                <canvas id="overallChart"></canvas>
            </div>
            <div class="stat-value">{total}</div>
            <p class="stat-label">Total Games Played</p>
            <div class="stat-value">{int(avg_rating)}</div>
            <p class="stat-label">Average Rating</p>
        </div>
        {peak_ratings_html}
    </div>
"""


        scripts = "".join([f"""
    <script>
        Chart.defaults.font.family = "'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif";
        Chart.defaults.font.size = 14;
        """, f"""
        // Overall Record Chart
        new Chart(document.getElementById('overallChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Wins', 'Draws', 'Losses'],
                datasets: [{{
                    data: [{wins}, {draws}, {losses}],
                    backgroundColor: ['#2ecc71', '#f1c40f', '#e74c3c'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                cutout: '70%',
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});""", f"""
        
        // Color Performance Chart
        new Chart(document.getElementById('colorChart'), {{
            type: 'bar',
            data: {{
                labels: ['White', 'Black'],
                datasets: [{{
                    label: 'Wins',
                    data: [{color_stats[0][2]}, {color_stats[1][2]}],
                    backgroundColor: '#2ecc71'
                }},
                {{
                    label: 'Draws',
                    data: [{color_stats[0][3]}, {color_stats[1][3]}],
                    backgroundColor: '#f1c40f'
                }},
                {{
                    label: 'Losses',
                    data: [{color_stats[0][4]}, {color_stats[1][4]}],
                    backgroundColor: '#e74c3c'
                }}]
            }},
            options: {{
                scales: {{
                    x: {{ stacked: true }},
                    y: {{ stacked: true }}
                }}
            }}
        }});""", f"""
        
        // Openings Chart
        new Chart(document.getElementById('openingsChart'), {{
            type: 'doughnut',
            data: {{
                labels: {[opening for opening, _ in openings]},
                datasets: [{{
                    data: {[count for _, count in openings]},
                    backgroundColor: ['#FF9800', '#2196F3', '#9C27B0', '#00BCD4', '#795548']
                }}]
            }}
        }});""", f"""
        
        // Mistakes Chart
        new Chart(document.getElementById('mistakesChart'), {{
            type: 'bar',
            data: {{
                labels: {list(mistake_counts.keys())},
                datasets: [{{
                    data: {list(mistake_counts.values())},
                    backgroundColor: '#2196F3'
                }}]
            }},
            options: {{
                indexAxis: 'y'
            }}
        }});""", f"""
        
        // Blunder Timing Chart
        new Chart(document.getElementById('blunderChart'), {{
            type: 'line',
            data: {{
                labels: {list(move_counts.keys())},
                datasets: [{{
                    label: 'Blunders',
                    data: {list(move_counts.values())},
                    borderColor: '#F44336',
                    tension: 0.1
                }}]
            }}
        }});""", f"""
        
        // Time Control Chart
        new Chart(document.getElementById('timeControlChart'), {{
            type: 'doughnut',
            data: {{
                labels: {time_control_data['labels']},
                datasets: [{{
                    data: {time_control_data['data']},
                    backgroundColor: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                cutout: '60%',
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{
                            padding: 20,
                            font: {{
                                size: 14
                            }}
                        }}
                    }},
                    title: {{
                        display: true,
                        text: 'Games by Time Control',
                        font: {{
                            size: 16,
                            weight: 'normal'
                        }}
                    }}
                }},
                layout: {{
                    padding: 20
                }}
            }}
        }});""", f"""
        
        // Rating History Chart
        new Chart(document.getElementById('ratingChart'), {{
            type: 'line',
            data: {{
                datasets: [{','.join(rating_datasets)}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        type: 'time',
                        time: {{
                            unit: 'month'
                        }},
                        title: {{
                            display: true,
                            text: 'Date'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Rating'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});""", f"""

        // Time Management Performance Chart
        new Chart(document.getElementById('timeManagementChart'), {{
            type: 'bar',
            data: {{
                labels: {[tc[0].title() for tc in time_management]},
                datasets: [{{
                    label: 'Average Accuracy',
                    data: {[tc[1] for tc in time_management]},
                    backgroundColor: '#4CAF50',
                    yAxisID: 'y'
                }}, {{
                    label: 'Games Played',
                    data: {[tc[2] for tc in time_management]},
                    backgroundColor: '#2196F3',
                    yAxisID: 'y1'
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        type: 'linear',
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Accuracy %'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Games'
                        }},
                        grid: {{
                            drawOnChartArea: false
                        }}
                    }}
                }}
            }}
        }});""", f"""

        // Rating Difference Performance Chart
        new Chart(document.getElementById('ratingDiffChart'), {{
            type: 'bar',
            data: {{
                labels: {[rp[0] for rp in rating_performance]},
                datasets: [{{
                    label: 'Win Rate %',
                    data: {[rp[2] for rp in rating_performance]},
                    backgroundColor: '#FF9800'
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        title: {{
                            display: true,
                            text: 'Win Rate %'
                        }}
                    }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const labels = {{
                                    'Much Weaker': 'Opponent rated 200+ points lower',
                                    'Weaker': 'Opponent rated 100-200 points lower',
                                    'Equal': 'Rating difference within 100 points',
                                    'Stronger': 'Opponent rated 100-200 points higher',
                                    'Much Stronger': 'Opponent rated 200+ points higher'
                                }};
                                const label = context.label;
                                const value = context.raw.toFixed(1) + '% win rate';
                                return [value, labels[label]];
                            }}
                        }}
                    }}
                }}
            }}
        }});""", f"""

        // Time of Day Performance Chart
        new Chart(document.getElementById('timeOfDayChart'), {{
            type: 'bar',
            data: {{
                labels: {[tod[0] for tod in time_of_day]},
                datasets: [{{
                    label: 'Win Rate %',
                    data: {[tod[2] for tod in time_of_day]},
                    backgroundColor: '#2ecc71',
                    borderWidth: 0,
                    yAxisID: 'y'
                }}, {{
                    label: 'Games Played',
                    data: {[tod[1] for tod in time_of_day]},
                    backgroundColor: '#3498db',
                    borderWidth: 0,
                    yAxisID: 'y1'
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        title: {{
                            display: true,
                            text: 'Win Rate %'
                        }},
                        beginAtZero: true,
                        max: 100
                    }},
                    y1: {{
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Games'
                        }},
                        grid: {{
                            drawOnChartArea: false
                        }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});""", f"""

        // Opening Win Rates Chart
        new Chart(document.getElementById('openingWinRatesChart'), {{
            type: 'bar',
            data: {{
                labels: {[op[0] for op in opening_stats]},
                datasets: [{{
                    label: 'Win Rate %',
                    data: {[op[2] for op in opening_stats]},
                    backgroundColor: '#2ecc71',
                    yAxisID: 'y'
                }}, {{
                    label: 'Games Played',
                    data: {[op[1] for op in opening_stats]},
                    backgroundColor: '#3498db',
                    yAxisID: 'y1'
                }}]
            }},
            options: {{
                indexAxis: 'y',  // Make it horizontal
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Opening'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Win Rate %'
                        }}
                    }},
                    y1: {{
                        position: 'right',
                        beginAtZero: true,
                        grid: {{
                            drawOnChartArea: false
                        }}
                    }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                if (context.dataset.label === 'Win Rate %') {{
                                    return `Win Rate: ${{context.raw.toFixed(1)}}%`;
                                }}
                                return `Games: ${{context.raw}}`;
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""])

        # Combine all parts
        html = header + overall_stats + charts + scripts

        # Write to file
        with open('results.html', 'w') as f:
            f.write(html)
        
        print("Report generated: results.html")

if __name__ == "__main__":
    downloader = ChessComDownloader("lukas19940000")
    # print(f"\nTotal games in database: {downloader.count_total_games()}")
    downloader.clear_analysis()
    downloader.analyze_all_games(depth=20)
    # downloader.delete_database()
    # downloader.download_all_games(limit=None)
    downloader.analyze_all_games_with_llm()
    # downloader.analyze_game_with_llm(45) 
    # downloader.generate_html_report()
    # downloader.generate_statistics() 