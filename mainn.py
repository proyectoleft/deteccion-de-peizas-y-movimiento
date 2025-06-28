import cv2
import numpy as np
import time
import chess
import chess.svg
from screeninfo import get_monitors
from cairosvg import svg2png
from PIL import Image
import io
from ultralytics import YOLO
import json
import os
from collections import Counter
import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk
from datetime import datetime


# === CONFIGURACIÓN AJUSTABLE ===
CONFIG = {
    'MODEL_PATH': r"C:/Users/BRANDON/Downloads/proye/proyecto2 - copia//best.pt",  # Ruta al modelo YOLO
    'CAMERA_INDEX': 1 , # <- este debe ser el que corresponde a Camo
    'VIDEO_PATH': r"C:/Users/BRANDON/Downloads/proye/proyecto2 - copia/video3.mp4",  # Ruta al video
    'CORNERS_FILE': r"C:/Users/BRANDON/Downloads/proye/proyecto2 - copia/board_corners.json",  # Archivo de esquinas
    'CONFIDENCE_THRESHOLD': 0.5,  # Umbral de confianza para YOLO 
                                  # Opciones: 0.2, 0.25, 0.3, 0.4
    'DETECTION_INTERVAL': 0.5,  # Intervalo entre detecciones en segundos (1.0: equilibrado, 0.5: más frecuente)
                                # Opciones: 0.5, 1.0, 1.5, 2.0
    'INPUT_RESOLUTION': (2560, 1440),  # Resolución para YOLO (320x320: rápido, 640x480: preciso)
                                     # Opciones: (320, 320), (416, 416), (640, 480)
    'BOARD_SIZE': 400,  # Tamaño del tablero digital en píxeles (400: equilibrado)
                        # Opciones: 300, 400, 500
    'FRAME_SUBSAMPLE': 1.0,  # Procesar cada N frames (1: todos, 2: cada segundo frame)
                           # Opciones: 1, 2, 3
    'STABILITY_THRESHOLD': 1,  # Detecciones consecutivas para confirmar una pieza (3: estable, 2: más rápido)

    'MOVIMIENTOS_PATH': r"C:/Users/BRANDON/Downloads/proye/proyecto2 - copia/t3_movimientos.txt",
                               # Opciones: 2, 3, 4
}



def detectar_movimiento(before, after, guardar_fn=None):
    movidos_desde = []
    movidos_hacia = []

    for square in before:
        if square not in after or before[square] != after[square]:
            movidos_desde.append((square, before[square]))

    for square in after:
        if square not in before or before.get(square) != after[square]:
            movidos_hacia.append((square, after[square]))

    print("Desde:", movidos_desde)
    print("Hacia:", movidos_hacia)

    if len(movidos_desde) == 1 and len(movidos_hacia) == 1:
        origen = movidos_desde[0][0]
        destino = movidos_hacia[0][0]

        if origen == destino:
            print("Movimiento inválido (origen igual a destino):", origen)
            return

        move_uci = origen + destino
        print("Movimiento detectado:", move_uci)

        if guardar_fn:
            guardar_fn(move_uci)
    else:
        print("No se detectó un solo movimiento claro.")

class AgrupadorTiempoReal:
    def __init__(self, archivo_salida="C:/Users/BRANDON/Downloads/proye/proyecto2 - copia/t3.txt"):
        self.grupo_actual = []
        self.repetidos = []
        self.ultimo_agregado = None
        self.contador = Counter()
        self.esperando_reset = False
        self.archivo_salida = archivo_salida
        self.archivo_movimientos = archivo_salida.replace(".txt", "_movimientos.txt")

        with open(self.archivo_salida, "w") as f:
            f.write("")
        with open(self.archivo_movimientos, "w") as f:
            f.write("")

    def procesar_diccionario(self, dic):
        if dic == {}:
            self.grupo_actual = []
            self.contador = Counter()
            self.esperando_reset = False
            print("Grupo reiniciado.")
        elif not self.esperando_reset:
            dic_str = json.dumps(dic, sort_keys=True)

            if self.contador[dic_str] >= 3:
                return

            self.grupo_actual.append(dic)
            self.contador[dic_str] += 1

            if self.contador[dic_str] == 3:
                dic_json = json.loads(dic_str)
                if dic_json == self.ultimo_agregado:
                    print("Repetido consecutivo:", dic_json)
                else:
                    print("Repetido detectado (3 veces):", dic_json)

                if self.ultimo_agregado is not None:
                    detectar_movimiento(self.ultimo_agregado, dic_json, guardar_fn=self._guardar_movimiento_txt)

                self.repetidos.append(dic_json)
                self._guardar_en_txt(dic_json)
                self.ultimo_agregado = dic_json
                self.esperando_reset = True
        else:
            print("Esperando nuevo grupo... Ignorando:", dic)

    def _guardar_en_txt(self, diccionario):
        with open(self.archivo_salida, "a") as f:
            f.write(json.dumps(diccionario, ensure_ascii=False) + "\n")

    def _guardar_movimiento_txt(self, movimiento_uci):

        hora_actual = datetime.now().strftime("%H:%M:%S")
        texto = f"[{hora_actual}] movimiento: {movimiento_uci}"
        with open(self.archivo_movimientos, "a") as f:
            f.write(texto + "\n")

    def obtener_repetidos(self):
        return self.repetidos
# === OBTENER RESOLUCIÓN DE PANTALLA ===
try:
    monitor = get_monitors()[0]
    screen_w, screen_h = monitor.width, monitor.height
except Exception:
    print("Error al obtener resolución de pantalla. Usando valores por defecto.")
    screen_w, screen_h = 1920, 1080

# === SEGMENTACIÓN DEL TABLERO ===

def guardar_diccionario(diccionario, archivo='C:/Users/BRANDON/Downloads/proye/proyecto2 - copia/t1.txt'):
    with open(archivo, 'a') as f:
        f.write(str(diccionario) + '\n')

def guardar_diccionario2(diccionario, archivo='C:/Users/BRANDON/Downloads/proye/proyecto2 - copia/t2.txt'):
    with open(archivo, 'a') as f:
        f.write(str(diccionario) + '\n')


def select_corners(event, x, y, flags, param):
    corners, frame, scale = param
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x / scale, y / scale))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Seleccionar Esquinas", frame)

def segment_board(frame):
    corners = []
    if os.path.exists(CONFIG['CORNERS_FILE']):
        with open(CONFIG['CORNERS_FILE'], 'r') as f:
            corners = json.load(f)
        print("✅ Esquinas cargadas desde", CONFIG['CORNERS_FILE'])
    else:
        frame_copy = frame.copy()
        resized_frame, scale = resize_to_fit_screen(frame_copy)
        cv2.namedWindow("Seleccionar Esquinas")
        cv2.setMouseCallback("Seleccionar Esquinas", select_corners, [corners, resized_frame, scale])
        print("Seleccione las 4 esquinas en orden: superior-izquierda, superior-derecha, inferior-derecha, inferior-izquierda")
        while len(corners) < 4:
            cv2.imshow("Seleccionar Esquinas", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt("Selección cancelada.")
        cv2.destroyWindow("Seleccionar Esquinas")
        with open(CONFIG['CORNERS_FILE'], 'w') as f:
            json.dump(corners, f)
        print("✅ Esquinas guardadas en", CONFIG['CORNERS_FILE'])
    
    src_pts = np.float32(corners)
    board_size = CONFIG['BOARD_SIZE']
    dst_pts = np.float32([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return matrix, corners

# === PROCESAMIENTO DE DETECCIONES YOLO ===
def map_piece_label(label, square):
    piece_map = {
        "white-pawn": "P", "black-pawn": "p",
        "white-rook": "R", "black-rook": "r",
        "white-knight": "N", "black-knight": "n",
        "white-bishop": "B", "black-bishop": "b",
        "white-queen": "Q", "black-queen": "q",
        "white-king": "K", "black-king": "k"
    }
    return piece_map.get(label.lower(), "")

def process_yolo_detections(frame, model, matrix, video_w, video_h):
    frame_resized = cv2.resize(frame, CONFIG['INPUT_RESOLUTION'], interpolation=cv2.INTER_NEAREST)
    results = model.predict(frame_resized, conf=CONFIG['CONFIDENCE_THRESHOLD'])
    
    scale_x = video_w / CONFIG['INPUT_RESOLUTION'][0]
    scale_y = video_h / CONFIG['INPUT_RESOLUTION'][1]
    board_size = CONFIG['BOARD_SIZE']
    positions = {}
    detections = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        conf = float(box.conf[0])
        label = model.names[int(box.cls[0])]

        # Calcular centro del bounding box en píxeles reales
        center = np.array([[[ (x1 + x2) / 2, (y1 + y2) / 2 - (y1 - y2) / 4 ]]], dtype=np.float32)

        # Aplicar transformación de perspectiva
        center_warped = cv2.perspectiveTransform(center, matrix)[0][0]

        # Mostrar resultados
        print(f"Etiqueta: {label}, Coordenadas transformadas: x={center_warped[0]:.1f}, y={center_warped[1]:.1f}")
        
        #center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
        #center_warped = cv2.perspectiveTransform(center[None, :, :], matrix)[0][0]
        #print(f"Etiqueta: {label}, Coordenadas: {center_warped}")
        
        if 0 <= center_warped[0] < board_size and 0 <= center_warped[1] < board_size:
            col = int(center_warped[0] * 8 / board_size)
            row = int(center_warped[1] * 8 / board_size)
            square = f"{chr(97 + col)}{8 - row}"
            piece = map_piece_label(label, square)
            if piece:
                positions[square] = piece
                detections.append((x1, y1, x2, y2, conf, label, square))
    
    return positions, detections

# === VISUALIZACIÓN CON PYTHON-CHESS ===
def positions_to_board(positions):
    board = chess.Board(None)
    for square, piece in positions.items():
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1])
        square_idx = row * 8 + col
        piece_map = {'P': chess.PAWN, 'p': chess.PAWN, 'R': chess.ROOK, 'r': chess.ROOK,
                     'N': chess.KNIGHT, 'n': chess.KNIGHT, 'B': chess.BISHOP, 'b': chess.BISHOP,
                     'Q': chess.QUEEN, 'q': chess.QUEEN, 'K': chess.KING, 'k': chess.KING}
        if piece in piece_map:
            color = chess.WHITE if piece.isupper() else chess.BLACK
            board.set_piece_at(square_idx, chess.Piece(piece_map[piece], color))
    return board

def draw_digital_board(positions, last_positions, last_board_img):
    # Si positions está vacío pero hay un último tablero válido, reutilízalo
    if not positions and last_positions and last_board_img is not None:
        return last_board_img, last_positions

    # Si las posiciones no han cambiado, reutiliza la imagen anterior
    if positions == last_positions and last_board_img is not None:
        return last_board_img, last_positions

    # Crear el tablero nuevo a partir de las posiciones actuales
    board = positions_to_board(positions)
    svg_data = chess.svg.board(board, size=CONFIG['BOARD_SIZE'])
    png_buffer = io.BytesIO()
    svg2png(bytestring=svg_data, write_to=png_buffer)
    img = Image.open(png_buffer)
    img_array = np.array(img)

    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    return img_array, positions

# === FILTRO DE ESTABILIDAD ===
def stabilize_positions(new_positions, prev_positions, counter):
    stabilized = {}
    for square, piece in new_positions.items():
        if square in prev_positions and prev_positions[square] == piece:
            counter[square] = counter.get(square, 0) + 1
        else:
            counter[square] = 1
        if counter.get(square, 0) >= CONFIG['STABILITY_THRESHOLD']:
            stabilized[square] = piece
    return stabilized, counter

# === UTILIDADES ===
def resize_to_fit_screen(frame):
    h, w = frame.shape[:2]
    scale = min(screen_w * 0.9 / w, screen_h * 0.9 / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized_frame, scale

def validar_diccionario(diccionario):
    if isinstance(diccionario, dict) and 30 < len(diccionario) < 34:
        return diccionario
    return {}



# === CLASE DE INTERFAZ ===

import numpy as np

class ChessDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de Ajedrez")
        
        self.video_label = tk.Label(root)
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        self.board_label = tk.Label(root)
        self.board_label.grid(row=0, column=1, padx=10, pady=10)

        # Historial de movimientos
        self.history_label = tk.Label(root, text="Historial de Movimientos")
        self.history_label.grid(row=2, column=0, columnspan=2)

        self.history_text = tk.Text(root, height=10, width=50, state="disabled")
        self.history_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        self.start_button = ttk.Button(root, text="▶️ Iniciar", command=self.start_detection)
        self.start_button.grid(row=1, column=0, pady=5)

        self.stop_button = ttk.Button(root, text="⏹️ Detener", command=self.stop_detection)
        self.stop_button.grid(row=1, column=1, pady=5)

        self.running = False
        self.thread = None

    def actualizar_historial(self):
        try:
            with open(CONFIG['MOVIMIENTOS_PATH'], 'r') as f:
                contenido = f.read()
            self.history_text.config(state="normal")
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(tk.END, contenido)
            self.history_text.config(state="disabled")
        except FileNotFoundError:
            pass

    def start_detection(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_detection)
            self.thread.start()

    def stop_detection(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def run_detection(self):
        try:
            model = YOLO(CONFIG['MODEL_PATH'])
            print("✅ Modelo cargado")
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            return
        
        cap = cv2.VideoCapture(CONFIG['CAMERA_INDEX'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el primer frame.")
            return


    
        matrix, corners = segment_board(frame)
        
        
        last_processed_time = 0
        prev_positions = {}
        counter = {}
        last_board_img = None
        last_positions = {}
        fps_start_time = time.time()
        frame_count = 0
        frame_idx = 0
        fps = 0.0
        agrupador = AgrupadorTiempoReal()

        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

        
        
            current_time = time.time()
            detections = []
            positions = {}
            prev_positions = {}

 
            if current_time - last_processed_time >= CONFIG['DETECTION_INTERVAL']:
                last_processed_time = current_time
                positions, detections = process_yolo_detections(frame, model, matrix, frame.shape[1], frame.shape[0])
                #stabilized_positions, counter = stabilize_positions(positions, prev_positions, counter)
                #prev_positions = positions
                #positions = stabilized_positions
                #print(f"\nFrame {cap.get(cv2.CAP_PROP_POS_FRAMES)} - Posiciones: {positions}")
                guardar_diccionario(positions)
                a=validar_diccionario(positions)
                agrupador.procesar_diccionario(a)

                self.root.after(0, self.actualizar_historial)


                guardar_diccionario2(a)
                positions=a
                 
            
            for det in detections:
                x1, y1, x2, y2, _, label, square = det
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} ({square})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            for corner in corners:
                cx, cy = map(int, corner)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            
            # Mostrar video
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Tablero digital
            digital_board, last_positions = draw_digital_board(positions, last_positions, last_board_img)
            last_board_img = digital_board
            board_img = Image.fromarray(cv2.cvtColor(digital_board, cv2.COLOR_BGR2RGB))
            board_imgtk = ImageTk.PhotoImage(image=board_img)
            self.board_label.imgtk = board_imgtk
            self.board_label.configure(image=board_imgtk)

        cap.release()

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    root = tk.Tk()
    app = ChessDetectorApp(root)
    root.mainloop()