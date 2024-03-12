import sys
import cv2
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Создание виджета для отображения видео
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Создание макета для размещения виджета
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)

        # Создание виджета для размещения макета
        container = QWidget()
        container.setLayout(layout)

        # Установка виджета в качестве центрального виджета окна
        self.setCentralWidget(container)

        # Загрузка модели и инициализация переменных
        self.model = YOLO('best (2).onnx')
        self.cap = cv2.VideoCapture(0)
        self.frame_counter = 0
        self.last_update_time = time.time()
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.subtitles = []
        self.current_x = 10
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.font_scale = 2
        self.font_thickness = 2
        self.color = (255, 255, 255)
        self.max_subtitles_per_line = 22
        self.min_display_time = 2
        self.letter_timings = {}

        # Таймер для обновления видео и субтитров
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / 30))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return


        # Обработка кадра и обновление субтитров
        results = self.model(frame, show=True, conf=0.3, save=False, stream=True)
        self.frame_counter += 1
        current_time = time.time()
        if current_time - self.last_update_time >= 1.0:
            fps = self.frame_counter / (current_time - self.last_update_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.frame_counter = 0
            self.last_update_time = current_time

        for r in results:
            boxes = r.boxes
            class_indices = boxes.cls.tolist()
            for i, detection in enumerate(boxes.xyxy):
                class_index = int(class_indices[i])
                letter_mapping = {0: 'А', 1: 'И', 2: 'Л', 3: 'М', 4: 'Н', 5: 'О', 6: 'П', 7: 'Р', 8: 'С', 9: 'Б', 10: 'Т', 11: 'У', 12: 'Ф', 13: 'Х', 14: 'Ч', 15: 'Ш', 16: 'Ы', 17: 'В', 18: 'Э', 19: 'Ю', 20: 'Я', 21: 'Г', 22: 'Е', 23: 'Ж'}
                letter = letter_mapping.get(class_index, "")

                if letter:
                    text_size, _ = cv2.getTextSize(letter, self.font, self.font_scale, self.font_thickness)
                    y = frame.shape[0] - text_size[1] - 10

                    if letter in self.letter_timings:
                        elapsed_time = time.time() - self.letter_timings[letter]
                        if elapsed_time >= self.min_display_time:
                            self.subtitles.append(((self.current_x, y), letter))
                            self.current_x += text_size[0] + 10
                            if len(self.subtitles) >= self.max_subtitles_per_line:
                                self.subtitles = self.subtitles[self.max_subtitles_per_line:]
                                self.current_x = 10
                            self.letter_timings[letter] = time.time()
                    else:
                        self.letter_timings[letter] = time.time()

        for (x, y), letter in self.subtitles:
            cv2.putText(frame, letter, (x, y), self.font, self.font_scale, self.color, self.font_thickness)

        # Конвертация кадра из BGR в RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Преобразование кадра в QImage и отображение его на QLabel
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())