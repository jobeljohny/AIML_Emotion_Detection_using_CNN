from PIL import Image, ImageTk
import tkinter as tk
import cv2
from predictor import predict

class Application:
    def __init__(self):
      
        self.vs = cv2.VideoCapture(0) 
       
        self.current_image = None 

        self.root = tk.Tk()  
        self.root.title("Emotion Detector")  
      
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root) 
        self.panel.pack(padx=10, pady=10)

        self.emotionVariable = tk.StringVar()
        self.emotionVariable.set('loading...')
        result_Label = tk.Label(self.root, textvariable = self.emotionVariable)
        result_Label.pack(fill="both", expand=True, padx=10, pady=10)

       
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()  # read frame from video stream
        emotion=predict(frame)
        self.emotionVariable.set(emotion)
        if ok: 
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)  
            self.panel.imgtk = imgtk  
            self.panel.config(image=imgtk)  
        self.root.after(30, self.video_loop)

    def take_snapshot(self):
       pass

    def destructor(self):
        self.root.destroy()
        self.vs.release()  
        cv2.destroyAllWindows()  



# start the app
print("starting...")
pba = Application()
pba.root.mainloop()