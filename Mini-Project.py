import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualisasi Spektrum Frekuensi Citra")
        self.root.geometry("1000x600")

        self.citra_asli = None
        self.spektrum_magnitudo = None
        self.current_path = ""

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(expand=True, fill='both')

        control_frame = ttk.Frame(main_frame, width=200)
        control_frame.pack(side='left', fill='y', padx=(0, 10))

        ttk.Label(control_frame, text="Kontrol Proyek", font=("Helvetica", 14, "bold")).pack(pady=(0, 20))

        self.load_button = ttk.Button(control_frame, text="Muat Gambar", command=self.muat_gambar)
        self.load_button.pack(fill='x', pady=5)

        self.analyze_button = ttk.Button(control_frame, text="Analisis Frekuensi", command=self.mulai_analisis, state=tk.DISABLED)
        self.analyze_button.pack(fill='x', pady=5)

        self.reset_button = ttk.Button(control_frame, text="Reset", command=self.reset_tampilan)
        self.reset_button.pack(fill='x', pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        self.path_label = ttk.Label(control_frame, text="File: Belum ada", wraplength=180)
        self.path_label.pack(fill='x')

        visualization_frame = ttk.Frame(main_frame)
        visualization_frame.pack(side='right', expand=True, fill='both')

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=visualization_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)

        self.ax1.set_title("Citra Asli")
        self.ax1.axis('off')
        self.ax2.set_title("Spektrum Frekuensi")
        self.ax2.axis('off')
        self.fig.tight_layout()

        self.status_bar = ttk.Label(root, text="Siap", relief=tk.SUNKEN, anchor='w')
        self.status_bar.pack(side='bottom', fill='x')

    def muat_gambar(self):
        """Fungsi untuk memuat gambar dari file."""
        path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if path:
            self.citra_asli = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.citra_asli is None:
                messagebox.showerror("Error", "Gagal memuat gambar.")
                self.status_bar.config(text="Gagal memuat gambar")
                return
            
            self.citra_asli = self.citra_asli.astype(np.float32) / 255.0
            
            self.current_path = path
            self.path_label.config(text=f"File: {path.split('/')[-1]}")
            self.tampilkan_citra_asli()
            self.analyze_button.config(state=tk.NORMAL)
            self.status_bar.config(text="Gambar berhasil dimuat. Siap untuk analisis.")

    def tampilkan_citra_asli(self):
        """Fungsi untuk menampilkan citra asli di plot pertama."""
        self.ax1.clear()
        self.ax1.imshow(self.citra_asli, cmap='gray')
        self.ax1.set_title("Citra Asli (Domain Spasial)")
        self.ax1.axis('off')
        self.canvas.draw()

    def mulai_analisis(self):
        """Fungsi untuk melakukan FFT dan menampilkan spektrum frekuensi."""
        if self.citra_asli is None:
            messagebox.showerror("Error", "Mohon muat gambar terlebih dahulu.")
            return

        self.status_bar.config(text="Menganalisis frekuensi...")

        fft_citra = np.fft.fft2(self.citra_asli)
        fft_citra_geser = np.fft.fftshift(fft_citra)
        
        epsilon = 1e-8
        self.spektrum_magnitudo = 20 * np.log(np.abs(fft_citra_geser) + epsilon)
        
        self.tampilkan_spektrum()
        self.status_bar.config(text="Analisis frekuensi selesai")

    def tampilkan_spektrum(self):
        """Fungsi untuk menampilkan spektrum frekuensi di plot kedua."""
        self.ax2.clear()
        self.ax2.imshow(self.spektrum_magnitudo, cmap='gray', aspect='auto', origin='lower')
        self.ax2.set_title("Spektrum Frekuensi (FFT)")
        self.ax2.set_xlabel("Frekuensi Horizontal")
        self.ax2.set_ylabel("Frekuensi Vertikal")
        self.ax2.axis('off')
        self.canvas.draw()

    def reset_tampilan(self):
        """Fungsi untuk mereset semua tampilan."""
        self.citra_asli = None
        self.spektrum_magnitudo = None
        self.current_path = ""
        
        self.ax1.clear()
        self.ax1.set_title("Citra Asli")
        self.ax1.axis('off')
        
        self.ax2.clear()
        self.ax2.set_title("Spektrum Frekuensi")
        self.ax2.axis('off')
        
        self.canvas.draw()
        
        self.analyze_button.config(state=tk.DISABLED)
        self.path_label.config(text="File: Belum ada")
        self.status_bar.config(text="Tampilan direset")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()