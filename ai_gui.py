import os
import numpy as np
from PIL import Image
import tensorflow as tf
import customtkinter as ctk

# --- CLINICAL DARK THEME STYLING ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class MedicalAIGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Automated Cardiac Anomaly Detection Engine")
        self.geometry("850x520")
        self.resizable(False, False)

        # --- MODEL LOADING ---
        self.model_path = 'heartbeat_model.h5'
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print("\n========================================================")
            print("Model loaded successfully.")
            print(f"Model Input Shape: {self.model.input_shape}")
            # Detect if model has internal Rescaling layer
            self.model_has_rescaling = isinstance(
                self.model.layers[0], tf.keras.layers.Rescaling
            )
            print(f"Internal Rescaling Layer Detected: {self.model_has_rescaling}")
            print("========================================================\n")
        else:
            self.model = None
            self.model_has_rescaling = False
            print("Error: 'heartbeat_model.h5' not found.")

        # --- GRID LAYOUT ---
        self.grid_columnconfigure(0, weight=1, minsize=380)
        self.grid_columnconfigure(1, weight=1, minsize=470)
        self.grid_rowconfigure(0, weight=1)

        # =========================================================
        # LEFT PANEL
        # =========================================================
        self.left_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#1A1C1E")
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.title_label = ctk.CTkLabel(
            self.left_frame, text="Neural Evaluation Unit",
            font=ctk.CTkFont(family="Arial", size=18, weight="bold"), text_color="#E2E8F0"
        )
        self.title_label.pack(anchor="w", padx=25, pady=(25, 5))

        self.subtitle_label = ctk.CTkLabel(
            self.left_frame, text="Binary Deep Learning Feature Classifier",
            font=ctk.CTkFont(family="Arial", size=12), text_color="#718096"
        )
        self.subtitle_label.pack(anchor="w", padx=25, pady=(0, 20))

        self.upload_btn = ctk.CTkButton(
            self.left_frame, text="Upload Heartbeat Image", command=self.upload_and_predict,
            font=ctk.CTkFont(size=13, weight="bold"), height=42, corner_radius=8,
            fg_color="#3B82F6", hover_color="#2563EB"
        )
        self.upload_btn.pack(fill="x", padx=25, pady=(0, 25))

        self.image_preview = ctk.CTkLabel(
            self.left_frame, text="Awaiting Image Input",
            font=ctk.CTkFont(size=12), fg_color="#111315", corner_radius=10,
            width=330, height=260
        )
        self.image_preview.pack(padx=25, pady=(0, 25), expand=True, fill="both")

        # =========================================================
        # RIGHT PANEL
        # =========================================================
        self.right_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#111315")
        self.right_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.dash_label = ctk.CTkLabel(
            self.right_frame, text="DIAGNOSTIC TELEMETRY",
            font=ctk.CTkFont(family="Arial", size=11, weight="bold"), text_color="#A0AEC0"
        )
        self.dash_label.pack(anchor="w", padx=30, pady=(30, 0))

        self.status_box = ctk.CTkLabel(
            self.right_frame, text="SYSTEM IDLE\nUpload an image to begin",
            font=ctk.CTkFont(family="Arial", size=18, weight="bold"), text_color="#A0AEC0",
            height=160, fg_color="#1A1C1E", corner_radius=12
        )
        self.status_box.pack(fill="x", padx=30, pady=(25, 20))

        self.confidence_title = ctk.CTkLabel(
            self.right_frame, text="Classification Confidence",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#718096"
        )
        self.confidence_title.pack(anchor="w", padx=30, pady=(15, 5))

        self.progress_bar = ctk.CTkProgressBar(self.right_frame, height=18, corner_radius=9)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=30, pady=(0, 10))

        self.confidence_percentage = ctk.CTkLabel(
            self.right_frame, text="Confidence Level: --%",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#A0AEC0"
        )
        self.confidence_percentage.pack(anchor="e", padx=30)

        self.spec_label = ctk.CTkLabel(
            self.right_frame,
            text="Model Architecture: 3-Layer Convolutional Neural Network\n"
                 "Validation Accuracy: 95.90% | Decision Threshold: 0.50\n"
                 "Classes: anomaly=0, normal=1 (Keras alphabetical order)",
            font=ctk.CTkFont(size=10), text_color="#4A5568", justify="left"
        )
        self.spec_label.pack(side="bottom", anchor="w", padx=30, pady=25)

    # --- PREDICTION PIPELINE ---
    def upload_and_predict(self):
        if self.model is None:
            self.status_box.configure(
                text="CRITICAL ERROR\n'heartbeat_model.h5' Missing!",
                text_color="#EF4444"
            )
            return

        file_path = ctk.filedialog.askopenfilename(
            title="Select Heartbeat GAF Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )

        if not file_path:
            return

        # --- STEP 1: Show preview ---
        native_img = Image.open(file_path)
        scaled_ctk_img = ctk.CTkImage(
            light_image=native_img, dark_image=native_img, size=(300, 240)
        )
        self.image_preview.configure(image=scaled_ctk_img, text="")

        # --- STEP 2: Preprocess to match training exactly ---
        # Training used: grayscale, 64x64, raw uint8 fed to model
        # Model's first layer is Rescaling(1./255) — it handles normalization internally
        # So we must NOT divide by 255 here. Feed raw pixel values (0–255).

        img_gray = native_img.convert('L')            # Grayscale (1 channel)
        img_resized = img_gray.resize((64, 64))        # Match training size
        img_array = np.array(img_resized, dtype='float32')  # Keep as 0–255 range

        # ✅ DO NOT divide by 255 — the model's internal Rescaling layer does this.
        # Dividing here caused DOUBLE normalization: values became ~0.004 (all black),
        # which is why the model always output the same fixed number.

        # Shape: (1, 64, 64, 1) — batch of 1, grayscale
        img_tensor = img_array[np.newaxis, :, :, np.newaxis]

        # --- STEP 3: Inference ---
        raw_prediction = self.model.predict(img_tensor, verbose=0)[0][0]

        print(f"\n[DEBUG] File     : {os.path.basename(file_path)}")
        print(f"[DEBUG] Raw Score: {raw_prediction:.6f}")
        print(f"[DEBUG] Pixel min/max in tensor: {img_tensor.min():.1f} / {img_tensor.max():.1f}")

        # --- STEP 4: Interpret result ---
        # Keras sorted folders alphabetically: anomaly=0, normal=1
        # So: output close to 1.0 = Normal, output close to 0.0 = Anomaly
        THRESHOLD = 0.5

        if raw_prediction >= THRESHOLD:
            confidence = raw_prediction * 100
            self.status_box.configure(
                text="NORMAL DIAGNOSIS\nHealthy Signal Profile Detected",
                text_color="#10B981", fg_color="#192D20"
            )
            self.progress_bar.configure(progress_color="#10B981")
            self.progress_bar.set(float(raw_prediction))
            self.confidence_percentage.configure(
                text=f"Normal Confidence: {confidence:.2f}%"
            )
            print(f"[DEBUG] Result   : NORMAL ({confidence:.2f}% confidence)")
        else:
            confidence = (1.0 - raw_prediction) * 100
            self.status_box.configure(
                text="ANOMALY DETECTED\nCardiac Irregularity Identified",
                text_color="#EF4444", fg_color="#2D1919"
            )
            self.progress_bar.configure(progress_color="#EF4444")
            self.progress_bar.set(float(1.0 - raw_prediction))
            self.confidence_percentage.configure(
                text=f"Anomaly Confidence: {confidence:.2f}%"
            )
            print(f"[DEBUG] Result   : ANOMALY ({confidence:.2f}% confidence)")

if __name__ == "__main__":
    app = MedicalAIGUI()
    app.mainloop()