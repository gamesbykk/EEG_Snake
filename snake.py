import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import lfilter, lfilter_zi
from mne.filter import create_filter
import pandas as pd
from collections import deque
from datetime import datetime
import pygame
import threading
import time
import sys
import random

class SnakeGame:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.cell_size = 20
        self.grid_width = width // self.cell_size
        self.grid_height = height // self.cell_size
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        
        # Game state
        self.reset_game()
        
        # Font for score display
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        
    def reset_game(self):
        """Reset the game to initial state"""
        self.snake = [(self.grid_width//2, self.grid_height//2)]
        self.direction = (1, 0)  # Moving right initially
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        
    def generate_food(self):
        """Generate food at random position not occupied by snake"""
        while True:
            food_pos = (random.randint(0, self.grid_width-1), 
                       random.randint(0, self.grid_height-1))
            if food_pos not in self.snake:
                return food_pos
    
    def update_direction(self, new_direction):
        """Update snake direction, preventing immediate reversal"""
        # Prevent snake from going into itself
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction
    
    def update_game(self):
        """Update game state for one frame"""
        if self.game_over:
            return
            
        # Move snake head
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or 
            new_head[1] < 0 or new_head[1] >= self.grid_height):
            self.game_over = True
            return
            
        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            return
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check food collision
        if new_head == self.food:
            self.score += 10
            self.food = self.generate_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
    
    def draw(self, screen):
        """Draw the game on the screen"""
        # Clear screen
        screen.fill(self.BLACK)
        
        # Draw snake
        for segment in self.snake:
            rect = pygame.Rect(segment[0] * self.cell_size, 
                             segment[1] * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(screen, self.GREEN, rect)
            pygame.draw.rect(screen, self.WHITE, rect, 1)  # Border
        
        # Draw food
        food_rect = pygame.Rect(self.food[0] * self.cell_size,
                               self.food[1] * self.cell_size,
                               self.cell_size, self.cell_size)
        pygame.draw.rect(screen, self.RED, food_rect)
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        screen.blit(score_text, (10, 10))
        
        # Draw game over message
        if self.game_over:
            game_over_text = self.font.render("GAME OVER - Press R to restart", True, self.RED)
            text_rect = game_over_text.get_rect(center=(self.width//2, self.height//2))
            screen.blit(game_over_text, text_rect)


class EEGSnakeRecorder:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("EEG Snake Game Recorder")
        self.clock = pygame.time.Clock()
        
        # Initialize snake game
        self.snake_game = SnakeGame()
        
        # LSL configuration
        self.LSL_SCAN_TIMEOUT = 5
        self.LSL_EEG_CHUNK = 8
        
        # Data recording variables
        self.recording = False
        self.recorded_data = deque()
        self.current_action = 'no_action'
        self.sfreq = None
        self.n_chans = None
        
        # Filter parameters
        self.filt = True
        self.af = [1.0]
        self.bf = None
        self.filt_state = None
        
        # Key mapping (pygame key constants)
        self.key_mapping = {
            pygame.K_UP: 'up',
            pygame.K_DOWN: 'down',
            pygame.K_LEFT: 'left',
            pygame.K_RIGHT: 'right'
        }
        
        # Direction mapping for snake
        self.direction_mapping = {
            pygame.K_UP: (0, -1),
            pygame.K_DOWN: (0, 1),
            pygame.K_LEFT: (-1, 0),
            pygame.K_RIGHT: (1, 0)
        }
        
        self.active_keys = set()
        
        # Game timing
        self.game_speed = 10  # FPS for snake game
        self.last_move_time = 0
        self.move_delay = 100  # milliseconds between moves
        
        # Initialize LSL stream
        self.initialize_stream()
        
    def initialize_stream(self):
        print("Looking for an EEG stream...")
        streams = resolve_byprop('type', 'EEG', timeout=self.LSL_SCAN_TIMEOUT)
        
        if len(streams) == 0:
            raise RuntimeError("Can't find EEG stream.")
        
        print("EEG stream found. Preparing to record...")
        self.inlet = StreamInlet(streams[0], max_chunklen=self.LSL_EEG_CHUNK)
        info = self.inlet.info()
        self.sfreq = info.nominal_srate()
        self.n_chans = info.channel_count()
        
        # Initialize filter
        dummy_data = np.zeros((100, self.n_chans))
        self.bf = create_filter(dummy_data.T, self.sfreq, 3, 40., method='fir')
        zi = lfilter_zi(self.bf, self.af)
        self.filt_state = np.tile(zi, (self.n_chans, 1)).transpose()
    
    def handle_keyboard_events(self):
        """Process pygame keyboard events to update current action and control snake"""
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.recording = False
                pygame.quit()
                return
            
            if event.type == pygame.KEYDOWN:
                if event.key in self.key_mapping:
                    self.active_keys.add(event.key)
                    # Update snake direction immediately on key press
                    if event.key in self.direction_mapping:
                        self.snake_game.update_direction(self.direction_mapping[event.key])
                elif event.key == pygame.K_ESCAPE:
                    self.recording = False
                    return
                elif event.key == pygame.K_r and self.snake_game.game_over:
                    # Restart game
                    self.snake_game.reset_game()
                elif event.key == pygame.K_SPACE:
                    # Toggle recording
                    if self.recording:
                        self.recording = False
                    else:
                        self.start_recording()
            
            if event.type == pygame.KEYUP:
                if event.key in self.key_mapping and event.key in self.active_keys:
                    self.active_keys.remove(event.key)
        
        # Determine current action based on active keys
        if not self.active_keys:
            self.current_action = 'no_action'
        else:
            # For multiple keys, we'll just use the first one we find
            for key in self.active_keys:
                if key in self.key_mapping:
                    self.current_action = self.key_mapping[key]
                    break
        
        # Update snake game at controlled intervals
        if current_time - self.last_move_time >= self.move_delay:
            self.snake_game.update_game()
            self.last_move_time = current_time
    
    def start_recording(self):
        """Begin recording EEG data with action labels"""
        if self.recording:
            print("Already recording")
            return
            
        self.recording = True
        self.recorded_data.clear()
        print("Recording started. Play snake with arrow keys. Press ESC to stop, SPACE to pause/resume recording.")
        
        # Start data collection thread
        self.collection_thread = threading.Thread(target=self._collect_data)
        self.collection_thread.start()
    
    def _collect_data(self):
        """Thread function to continuously collect data"""
        try:
            while self.recording:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.1, 
                                                          max_samples=self.LSL_EEG_CHUNK)
                if samples:
                    samples = np.array(samples)
                    
                    # Apply filtering if enabled
                    if self.filt:
                        samples, self.filt_state = lfilter(self.bf, self.af, samples,
                                                         axis=0, zi=self.filt_state)
                    
                    # Store data with current action label and game state
                    for sample in samples:
                        sample_dict = {
                            'timestamp': datetime.now().isoformat(timespec='milliseconds'),
                            'action': self.current_action,
                            'score': self.snake_game.score,
                            'game_over': self.snake_game.game_over,
                            'snake_length': len(self.snake_game.snake),
                            **{f'channel_{i}': float(val) for i, val in enumerate(sample)}
                        }
                        self.recorded_data.append(sample_dict)
                        
        except Exception as e:
            print(f"Error in data collection: {e}")
            self.recording = False
    def _collect_sample(self):
        samples, timestamps = self.inlet.pull_chunk(timeout=0.1, 
                                                      max_samples=self.LSL_EEG_CHUNK)
        data=[]
        if samples:
            samples = np.array(samples)
            
            # Apply filtering if enabled
            if self.filt:
                samples, self.filt_state = lfilter(self.bf, self.af, samples,
                                                    axis=0, zi=self.filt_state)
            for sample in samples:
                data.append([float(val) for i, val in enumerate(sample) if i!=4])
            return data
    def run_game(self):
        """Main game loop"""
        print("EEG Snake Game Recorder")
        print("Controls:")
        print("- Arrow keys: Control snake")
        print("- SPACE: Start/Stop recording")
        print("- R: Restart game (when game over)")
        print("- ESC: Exit")
        print("\nPress SPACE to start recording, then play!")
        
        running = True
        while running:
            # Handle events
            self.handle_keyboard_events()
            
            if not self.recording and pygame.key.get_pressed()[pygame.K_ESCAPE]:
                running = False
            
            # Draw everything
            self.snake_game.draw(self.screen)
            
            # Draw recording status
            if self.recording:
                rec_text = pygame.font.Font(None, 24).render("RECORDING", True, (255, 0, 0))
                self.screen.blit(rec_text, (10, 50))
            else:
                rec_text = pygame.font.Font(None, 24).render("Press SPACE to record", True, (255, 255, 255))
                self.screen.blit(rec_text, (10, 50))
            
            # Draw current action
            action_text = pygame.font.Font(None, 24).render(f"Action: {self.current_action}", True, (255, 255, 255))
            self.screen.blit(action_text, (10, 80))
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS for smooth display
        
        # Clean up
        self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and return collected data as DataFrame"""
        if not self.recording:
            return None
            
        self.recording = False
        
        # Wait for collection thread to finish
        if hasattr(self, 'collection_thread') and self.collection_thread.is_alive():
            self.collection_thread.join()
        
        if not self.recorded_data:
            print("No data recorded")
            return None
        
        df = self._create_dataframe()
        print(f"Recording stopped. Collected {len(df)} samples.")
        return df
    
    def _create_dataframe(self):
        """Convert recorded data to pandas DataFrame"""
        df = pd.DataFrame(self.recorded_data)
        
        # Reorder columns to have metadata first
        metadata_cols = ['timestamp', 'action', 'score', 'game_over', 'snake_length']
        channel_cols = [f'channel_{i}' for i in range(self.n_chans)]
        cols = metadata_cols + channel_cols
        return df[cols]
    
    def save_to_csv(self, filename=None):
        """Save recorded data to CSV file"""
        if not self.recorded_data:
            print("No data to save")
            return None
        
        df = self._create_dataframe()
        if filename is None:
            filename = f"eeg_snake_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        print(f"Saving data to {filename}...")
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        
        # Print summary statistics
        print("\nData Summary:")
        print(f"Total samples: {len(df)}")
        print(f"Recording duration: {len(df) / self.sfreq:.2f} seconds")
        print(f"Actions distribution:\n{df['action'].value_counts()}")
        print(f"Max score reached: {df['score'].max()}")
        print(f"Games completed: {df['game_over'].sum()}")
        print("\nFirst few rows:")
        print(df.head())
        
        return filename


if __name__ == '__main__':
    try:
        recorder = EEGSnakeRecorder()
        
        # Run the game (this will block until user exits)
        recorder.run_game()
        
        # Save any recorded data
        if recorder.recorded_data:
            recorder.save_to_csv()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()
        sys.exit()
