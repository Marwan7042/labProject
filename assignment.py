# --- STEP 1: IMPORTING THE TOOLS ---
# Think of these as "toolboxes" we bring in to help us.
import numpy as np                 # Used for advanced math and handling lists of numbers (signals).
import matplotlib.pyplot as plt    # Used for drawing the graphs and charts.
from scipy.io import wavfile       # Used specifically to open and read .wav audio files.
from scipy import signal           # Used for engineering tasks like filtering and frequency analysis.
import tkinter as tk               # Used to create the windows, buttons, and text boxes (the GUI).
from tkinter import filedialog, messagebox # Tools to show "Open File" windows and "Alert" messages.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Connects the graphs to the window.

# This "class" is like a blueprint for our entire application.
class AudioFDMSystem:
    def __init__(self, root):
        """ This function runs the moment the program starts to set things up. """
        self.root = root
        self.root.title("EEC2220: Audio Filtering & FDM Modulation") # Sets the title of the window.
        self.root.geometry("1000x800") # Sets the starting size of the window.

        # These are empty "folders" where we will store our audio data once we load it.
        self.fs = 0 # This will store the 'Sample Rate' (how many snapshots of sound per second).
        self.channels = [] # This will hold the raw sound from the left/right speakers.
        self.filtered_channels = [] # This will hold the "cleaned up" sound after filtering.
        
        # We call this function to actually draw the buttons and screen layout.
        self.create_widgets()

    def create_widgets(self):
        """ This builds the 'look' of the application. """
        # We create a 'frame' (a container) at the top for our buttons.
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Button to trigger the file loading process.
        tk.Button(top_frame, text="Load Audio Files (Select 2)", command=self.load_audio).pack(side=tk.LEFT, padx=10)
        
        # A label and a text box where the user can type the order (e.g., 2,1,4,3).
        tk.Label(top_frame, text="Define Order (e.g., 3,1,4,2):").pack(side=tk.LEFT, padx=5)
        self.order_entry = tk.Entry(top_frame, width=15)
        self.order_entry.insert(0, "1,2,3,4") # Default text inside the box.
        self.order_entry.pack(side=tk.LEFT, padx=5)

        # The main button that triggers the math and the plotting.
        tk.Button(top_frame, text="Process & Visualize", command=self.process_signals).pack(side=tk.LEFT, padx=10)

        # Here we create a "canvas" where our 3 graphs will live.
        # Subplots(3,1) means 3 rows of graphs in 1 column.
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 10))
        self.fig.tight_layout(pad=4.0) # Adds some space so labels don't overlap.
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def load_audio(self):
        """ This handles opening the files and preparing the sound data. """
        # Opens a window for you to pick 2 files.
        file_paths = filedialog.askopenfilenames(title="Select Two Stereo .wav Files", filetypes=[("WAV files", "*.wav")])
        
        if len(file_paths) != 2:
            messagebox.showwarning("Input Error", "Please select exactly two stereo WAV files.")
            return

        try:
            temp_channels = []
            sample_rates = []

            for path in file_paths:
                # Read the file: fs is the speed, data is the actual sound waves.
                fs, data = wavfile.read(path)
                
                # Check if it's stereo (2 channels: left and right).
                if len(data.shape) != 2 or data.shape[1] != 2:
                    messagebox.showerror("Format Error", f"{path} is not a stereo file.")
                    return
                
                # Audio files can have very different volumes. We 'normalize' them 
                # to be between -1 and 1 so they don't drown each other out.
                data = data / np.max(np.abs(data))
                
                # Split the stereo file into two separate lists of numbers.
                temp_channels.append(data[:, 0]) # Left channel.
                temp_channels.append(data[:, 1]) # Right channel.
                sample_rates.append(fs)

            # We need all files to run at the same speed. We pick the highest speed found.
            self.fs = max(sample_rates)
            
            # If the files are different lengths, we add 'silence' (zeros) to the shorter ones
            # so they all start and end at the exact same time.
            max_len = max(len(ch) for ch in temp_channels)
            self.channels = []
            for ch in temp_channels:
                padded = np.pad(ch, (0, max_len - len(ch)), 'constant')
                self.channels.append(padded)

            messagebox.showinfo("Success", "Files loaded and synchronized.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load files: {e}")

    def apply_low_pass(self, data, cutoff=5000):
        """ This is like a 'sieve' for sound. It lets low sounds through and blocks high ones. """
        # We need to tell the math what the 'Nyquist' limit is (half our sampling speed).
        nyquist = self.fs / 2
        # Normalizing the cutoff frequency so the computer understands it.
        norm_cutoff = cutoff / nyquist
        # Designing a 'Butterworth' filter - a very smooth type of filter used in electronics.
        b, a = signal.butter(5, norm_cutoff, btype='low')
        # Applying the filter to our list of sound numbers.
        return signal.lfilter(b, a, data)

    def process_signals(self):
        """ This is the core logic: Filtering -> Modulation -> Mixing -> Demodulation. """
        if not self.channels:
            messagebox.showwarning("Warning", "Please load audio files first.")
            return

        # --- 1. GETTING THE USER'S PREFERRED ORDER ---
        try:
            # We take the text "1,2,3,4", split it, and turn it into computer numbers.
            order = [int(x.strip()) - 1 for x in self.order_entry.get().split(',')]
            if len(order) != 4 or any(o < 0 or o > 3 for o in order):
                raise ValueError
        except:
            messagebox.showerror("Input Error", "Enter order as four numbers 1-4 separated by commas.")
            return

        # --- 2. INDIVIDUAL FILTERING ---
        # We limit the bandwidth of each channel to 7.5kHz. 
        # This makes sure they don't "spill" into each other when we put them on the radio spectrum.
        self.filtered_channels = [self.apply_low_pass(ch, cutoff=7500) for ch in self.channels]

        # --- 3. MODULATION (FDM) ---
        # Frequency Division Multiplexing: putting multiple signals on different 'radio stations'.
        # We space our 'stations' 20kHz apart.
        spacing = 20000
        carriers = [spacing * (i + 1) for i in range(4)] # 20kHz, 40kHz, 60kHz, 80kHz.
        
        # Create a timeline (t) for our math (e.g., 0.001s, 0.002s...).
        t = np.arange(len(self.channels[0])) / self.fs
        # The 'composite' signal starts as silence, and we add modulated signals to it.
        composite_signal = np.zeros(len(t))
        
        # For each channel in the user's order:
        for i, ch_idx in enumerate(order):
            fc = carriers[i] # Pick the station frequency.
            # MATH: Multiply the sound by a high-speed Cosine wave (the Carrier).
            # This shifts the sound from low frequency to high frequency.
            modulated = self.filtered_channels[ch_idx] * np.cos(2 * np.pi * fc * t)
            # Add this 'station' to our single big wire (the composite signal).
            composite_signal += modulated

        # --- 4. DEMODULATION (RECOVERY) ---
        # Now we try to get the original sounds back out of the composite signal.
        recovered_channels = [None] * 4
        for i, ch_idx in enumerate(order):
            fc = carriers[i]
            # MATH: To recover, we multiply by the same Carrier wave again.
            mixed = composite_signal * np.cos(2 * np.pi * fc * t)
            # This creates the original sound + a very high-pitched noise.
            # We use the Low-Pass Filter to throw away the high noise and keep the sound.
            recovered_channels[ch_idx] = self.apply_low_pass(mixed, cutoff=8000) * 2

        # --- 5. VISUALIZATION ---
        self.update_plots(order, composite_signal, recovered_channels)

    def update_plots(self, order, composite, recovered):
        """ This draws the results on the screen. """
        for ax in self.axs: ax.clear() # Clear old graphs.

        # PLOT 1: Show the 4 signals sitting at baseband (0 to 10kHz).
        for i in range(4):
            # 'welch' converts time data into a frequency map (Spectrum).
            freqs, psd = signal.welch(self.filtered_channels[i], self.fs, nperseg=1024)
            self.axs[0].semilogy(freqs/1000, psd, label=f"Ch {i+1}")
        self.axs[0].set_title("1. Frequency Spectrum Before Modulation (Filtered)")
        self.axs[0].set_xlabel("Frequency (kHz)")
        self.axs[0].legend(loc='upper right', fontsize='small')

        # PLOT 2: Show the 'Composite' signal. You will see 4 peaks at 20, 40, 60, and 80kHz.
        freqs_c, psd_c = signal.welch(composite, self.fs, nperseg=2048)
        self.axs[1].semilogy(freqs_c/1000, psd_c, color='purple')
        self.axs[1].set_title(f"2. Composite Spectrum (Order: {[o+1 for o in order]})")
        self.axs[1].set_xlabel("Frequency (kHz)")

        # PLOT 3: Show the signals we recovered. They should look almost identical to Plot 1.
        for i in range(4):
            freqs_r, psd_r = signal.welch(recovered[i], self.fs, nperseg=1024)
            self.axs[2].semilogy(freqs_r/1000, psd_r, label=f"Recov Ch {i+1}", linestyle='--')
        self.axs[2].set_title("3. Spectrum of Recovered Demodulated Signals")
        self.axs[2].set_xlabel("Frequency (kHz)")
        self.axs[2].legend(loc='upper right', fontsize='small')

        self.fig.tight_layout()
        self.canvas.draw() # Refresh the screen with new drawings.

# --- START THE PROGRAM ---
if __name__ == "__main__":
    root = tk.Tk() # Create the main window.
    app = AudioFDMSystem(root) # Initialize our logic.
    root.mainloop() # Keep the window open and wait for clicks.