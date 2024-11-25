import threading
import tkinter as tk
from PIL import Image as PilImage
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.graphics import Color, Rectangle
from kivy.uix.popup import Popup
from kivy.core.window import Window
from torchvision import models
from torch import nn
import torch
import normalizar as nom
import manejo_archivos_pdf as mapdf
from plyer import filechooser
from kivy.config import Config
import os

# Deshabilitar WM_PenProvider
Config.set('input', 'wm_pen', 'ignore')
Config.set('input', 'wm_touch', 'ignore')

# Obtener la ruta absoluta de la carpeta del script actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Establecer la ruta actual como la base
os.chdir(BASE_DIR)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Cargar el modelo preentrenado ResNet50
        self.image_features_ = models.resnet50(pretrained=True)

        # Eliminar la última capa de clasificación (fully connected) de ResNet50
        self.image_features_ = nn.Sequential(*list(self.image_features_.children())[:-1])

        # Procesamiento de las características numéricas (4 características)
        self.numeric_features_ = nn.Sequential(
            nn.Linear(4, 64),  # Aquí usas 4 datos numéricos
            nn.GELU(),  # GELU en lugar de ReLU
            nn.Dropout(),
            nn.Linear(64, 64*3),
            nn.GELU(),  # GELU
            nn.Dropout(),
            nn.Linear(64*3, 64*3*3),
            nn.GELU(),  # GELU
        )

        # Capa final que combina las características visuales y numéricas
        self.combined_features_ = nn.Sequential(
            nn.Linear(2048 + 64*3*3, 64*3*3*2*2),  # 2048 provienen de ResNet50 + numéricas
            nn.GELU(),  # GELU
            nn.Dropout(),
            nn.Linear(64*3*3*2*2, 64*3*3*2),
            nn.GELU(),  # GELU
            nn.Linear(64*3*3*2, 64),
            nn.Linear(64, 1),  # Predicción final
        )

    def forward(self, x, y):
        # Pasar las imágenes por ResNet50 para obtener las características visuales
        x = self.image_features_(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        # Pasar las características numéricas por la red densa
        y = self.numeric_features_(y)

        # Combinar características visuales y numéricas
        z = torch.cat((x, y), dim=1)
        # print(z.shape)

        # Pasar las características combinadas por las capas finales
        z = self.combined_features_(z)

        return z.squeeze(1)
    
# Ruta y tamaño deseado para la imagen
def ajustar_imagen(ruta_original, ruta_ajustada, tamano):
    ruta_original = os.path.join(BASE_DIR, ruta_original)
    ruta_ajustada = os.path.join(BASE_DIR, ruta_ajustada)
    with PilImage.open(ruta_original) as img:
        img = img.resize(tamano)
        img.save(ruta_ajustada)
    
# Diccionario de preguntas y respuestas
RESPUESTAS = {
    "¿Es necesario toda la información?": "Sí, es necesario para un avalúo preciso.",
    "¿Es inmediato?": "El proceso es rápido, pero puede tomar unos minutos.",
    "¿Es gratuito?": "Sí, este servicio es gratuito.",
    "¿Cómo funciona?": "Nuestro sistema utiliza inteligencia artificial para evaluar tu inmueble."
}

# Variable global para controlar la ventana de ayuda
ventana_ayuda_abierta = False

# Función para iniciar la ventana de chat en un hilo separado
def abrir_chat():
    global ventana_ayuda_abierta
    if not ventana_ayuda_abierta:
        ventana_ayuda_abierta = True
        threading.Thread(target=mostrar_chat).start()

# Función para mostrar la pregunta y la respuesta en el chat
def mostrar_respuesta_en_chat(text_widget, pregunta):
    respuesta = RESPUESTAS.get(pregunta, "Lo siento, no tengo una respuesta para esa pregunta.")
    
    # Insertar la pregunta en el Text widget
    text_widget.insert(tk.END, "Pregunta: " + pregunta + "\n")
    
    # Insertar la respuesta en el Text widget
    text_widget.insert(tk.END, "Respuesta: " + respuesta + "\n\n")
    
    # Hacer scroll al final del Text widget
    text_widget.yview(tk.END)

# Función para mostrar el centro de ayuda en Tkinter
def mostrar_chat():
    global ventana_ayuda_abierta

    # Crear una ventana oculta de Tkinter para inicializar el entorno
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter

    # Crear ventana de chat
    chat = tk.Toplevel(root)
    chat.title("Centro de Ayuda")
    chat.geometry("350x530")

    # Al cerrar la ventana, actualizar la variable de control y destruir el entorno de Tkinter
    def cerrar_chat():
        global ventana_ayuda_abierta
        ventana_ayuda_abierta = False
        chat.destroy()
        root.quit()  # Cierra el entorno de Tkinter por completo

    chat.protocol("WM_DELETE_WINDOW", cerrar_chat)

    # Título
    titulo_label = tk.Label(chat, text="Centro de ayuda", font=("Helvetica", 16), bg="#d3d3d3")
    titulo_label.place(x=0, y=0, relwidth=1, height=30)

    # Sección de preguntas comunes
    preguntas_label = tk.Label(chat, text="Preguntas comunes", font=("Helvetica", 12))
    preguntas_label.place(x=10, y=40)

    # Crear un frame para las preguntas comunes
    preguntas_frame = tk.Frame(chat)
    preguntas_frame.place(x=10, y=70, relwidth=0.95)

    # Botones para preguntas comunes
    for idx, pregunta in enumerate(RESPUESTAS.keys()):
        btn_pregunta = tk.Button(preguntas_frame, text=pregunta, font=("Helvetica", 10), relief="ridge", 
                                 command=lambda p=pregunta: mostrar_respuesta_en_chat(text_widget, p))
        btn_pregunta.grid(row=idx, column=0, padx=10, pady=5, sticky="w")
    
    # Sección de chat
    chat_label = tk.Label(chat, text="Chat", font=("Helvetica", 12))
    chat_label.place(x=10, y=240)

    # Crear un Text widget para mostrar el chat (con ajuste de texto)
    text_widget = tk.Text(chat, font=("Helvetica", 10), wrap=tk.WORD)
    text_widget.place(x=10, y=270, relwidth=0.9, relheight=0.4)

    # Crear un Scrollbar y asociarlo al Text widget
    scrollbar = tk.Scrollbar(chat, orient="vertical", command=text_widget.yview)
    scrollbar.place(x=315, y=270, relheight=0.4)
    text_widget.config(yscrollcommand=scrollbar.set)

    # Insertar un mensaje inicial en el Text widget
    text_widget.insert(tk.END, "¡Hola! Ingresa tu pregunta y verificaré si tengo una respuesta a tu pregunta o a una pregunta similar.\n\n")

    # Botón para cerrar
    cerrar_btn = tk.Button(chat, text="Cerrar", command=cerrar_chat)
    cerrar_btn.place(x=150, y=490)

    # Ejecutar el bucle de eventos de Tkinter
    root.mainloop()

class ColoredBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(1, 0.6, 0, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self._update_rect, pos=self._update_rect)

    def _update_rect(self, instance, value):
        # Actualizar el tamaño y la posición del rectángulo de fondo
        self.rect.size = instance.size
        self.rect.pos = instance.pos

class AvaluoApp(App):
    def build(self):
        #self.modelo = nom.obtener_modelo()
        self.title = "Avalúo de Inmuebles"
        self.root = BoxLayout(orientation='vertical')
        self.crear_interfaz_inicio()
        return self.root

    def crear_interfaz_inicio(self):
        self.root.clear_widgets()

        # Layout superior que contendrá dos secciones
        layout_superior = ColoredBoxLayout(orientation='horizontal', size_hint=(1, 0.65))

        # Sección izquierda-arriba con la imagen de la casa
        layout_1 = BoxLayout(size_hint=(0.6, 1), orientation='vertical')
        self.imagen_1 = Image(source='assets/imagen_principal.jpg', allow_stretch=True, keep_ratio=True)
        layout_1.add_widget(self.imagen_1)

        # Sección derecha-arriba con el texto y el botón
        layout_2 = ColoredBoxLayout(size_hint=(0.4, 1), orientation='vertical')

        texto_inicio = Label(text="[b]Bienvenido a 'Price4House',[/b]\nesta aplicación te ayuda a\notener el avalúo de tu\ninmueble mediante\ninteligencia artificial.\n\nSolicita el avalúo de tu\ninmueble de forma inmediata.",
                             font_size=24, markup=True, halign='center', valign='middle')
        texto_inicio.bind(size=texto_inicio.setter('text_size'))
        layout_2.add_widget(texto_inicio)

        btnEA = Button(text="Empezar avalúo", background_color=(0, 0, 1, 1), color=(1, 1, 1, 1), size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        btnEA.bind(on_press=self.empezar_avaluo)
        layout_2.add_widget(btnEA)

        layout_2.add_widget(Label(size_hint=(None, 0.1)))

        # Layout inferior que contendrá las otras dos secciones
        layout_inferior = BoxLayout(orientation='horizontal', size_hint=(1, 0.35))

        # Sección izquierda-abajo con la descripción
        layout_3 = ColoredBoxLayout(size_hint=(0.6, 1), orientation='vertical')

        texto_desc = Label(text="[b]¿Qué es un avalúo?[/b]\nUn avalúo es un documento técnico que permite\nestimar el valor de un inmueble a partir de sus\ncaracterísticas físicas, de ubicación, de uso, y del\nanálisis del mercado inmobiliario.",
                            font_size=24, markup=True, halign='center', valign='top')
        texto_desc.bind(size=texto_desc.setter('text_size'))
        layout_3.add_widget(texto_desc)

        layout_btn_ayuda = BoxLayout(orientation='horizontal', spacing=10, size_hint=(None, 0.25), pos_hint={'center_x': 0.45})

        ajustar_imagen("assets/chat.png", "assets/chat_ajustado.png", (30, 30))
        btn_ayuda = Button(background_normal="assets/chat_ajustado.png", size_hint=(None, None), size=(30, 30))
        btn_ayuda.bind(on_release=lambda instance: abrir_chat())

        label_btn_ayuda = Label(text="Centro de ayuda", font_size=22, size_hint=(None, None), size=(150, 40))

        layout_btn_ayuda.add_widget(btn_ayuda)
        layout_btn_ayuda.add_widget(label_btn_ayuda)

        layout_3.add_widget(layout_btn_ayuda)

        layout_3.add_widget(Label(size_hint=(None, 0.1)))

        layout_btn_manual = BoxLayout(orientation='horizontal', spacing=10, size_hint=(None, 0.25), pos_hint={'center_x': 0.44})

        ajustar_imagen("assets/manual.png", "assets/manual_ajustado.png", (30, 30))
        btn_manual = Button(background_normal="assets/manual_ajustado.png", size_hint=(None, None), size=(30, 30))
        btn_manual.bind(on_press=self.mover_manual)

        label_btn_manual = Label(text="Manual de Usuario", font_size=22, size_hint=(None, None), size=(170, 40))

        layout_btn_manual.add_widget(btn_manual)
        layout_btn_manual.add_widget(label_btn_manual)

        layout_3.add_widget(layout_btn_manual)

        layout_3.add_widget(Label(size_hint=(None, 0.1)))

        # Sección derecha-abajo con la imagen
        layout_4 = ColoredBoxLayout(size_hint=(0.4, 1), orientation='vertical')
        self.imagen_2 = Image(source='assets/imagen_secundaria.jpg', allow_stretch=True, keep_ratio=True)
        layout_4.add_widget(self.imagen_2)

        # Añadir las secciones superiores al layout principal
        layout_superior.add_widget(layout_1)
        layout_superior.add_widget(layout_2)

        # Añadir las secciones inferiores al layout principal
        layout_inferior.add_widget(layout_3)
        layout_inferior.add_widget(layout_4)

        # Añadir los layouts superior e inferior al root
        self.root.add_widget(layout_superior)
        self.root.add_widget(layout_inferior)

        self.limpiar_form(0)

        ancho, alto = Window.size

        # Ajustar la imagen según el tamaño de la ventana
        ruta_original = 'assets/imagen_principal.jpg'
        ruta_ajustada = 'assets/imagen_principal_modificada.jpeg'
        ajustar_imagen(ruta_original, ruta_ajustada, (int(ancho * 0.65), int(alto * 0.65)))

        # Actualizar la fuente y recargar la imagen
        self.imagen_1.source = ruta_ajustada
        self.imagen_1.reload()

        ruta_original = 'assets/imagen_secundaria.jpg'
        ruta_ajustada = 'assets/imagen_secundaria_modificada.jpg'
        ajustar_imagen(ruta_original, ruta_ajustada, (int(ancho * 0.35), int(alto * 0.35)))

        self.imagen_2.source = ruta_ajustada
        self.imagen_2.reload()

        # Vincula el cambio de tamaño de la ventana a una función
        Window.bind(size=self.actualizar_tamano_inicio)

    def actualizar_tamano_inicio(self, instancia, valor):
        # Tamaño de la ventana
        ancho, alto = Window.size
        #print(f"Nueva resolución: {ancho}x{alto}")

        # Ajustar la imagen según el tamaño de la ventana
        ruta_original = 'assets/imagen_principal.jpg'
        ruta_ajustada = 'assets/imagen_principal_modificada.jpeg'
        ajustar_imagen(ruta_original, ruta_ajustada, (int(ancho * 0.65), int(alto * 0.65)))

        # Actualizar la fuente y recargar la imagen
        self.imagen_1.source = ruta_ajustada
        self.imagen_1.reload()

        ruta_original = 'assets/imagen_secundaria.jpg'
        ruta_ajustada = 'assets/imagen_secundaria_modificada.jpg'
        ajustar_imagen(ruta_original, ruta_ajustada, (int(ancho * 0.35), int(alto * 0.35)))

        self.imagen_2.source = ruta_ajustada
        self.imagen_2.reload()

    def mover_manual(self, *args): 
        mapdf.colocar_manual_descargas(BASE_DIR)

        # Crear el contenido del popup
        popup_content = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Agregar un Label con el mensaje
        label = Label(text="Manual de usuario descargado correctamente.",
                    size_hint=(1, None), height=50,
                    halign='center', valign='middle')
        popup_content.add_widget(label)

        # Crear el Popup
        popup = Popup(title="Notificación",
                    content=popup_content,
                    size_hint=(None, None),
                    size=(500, 150),
                    auto_dismiss=True) 

        popup.open()

    def empezar_avaluo(self, *args):
        self.root.clear_widgets()
        layout = ColoredBoxLayout(orientation='vertical', padding=10, spacing=10)

        btn_regresar = Button(text="Regresar", background_color=(1, 0, 0, 1), color=(1, 1, 1, 1), size_hint=(None, None), size=(150, 40), pos_hint={'x': 0.01})
        btn_regresar.bind(on_release=self.pantalla_inicio)
        layout.add_widget(btn_regresar)

        instrucciones = Label(text="Proporciona la información solicitada\npara generar el avalúo de tu inmueble.", font_size=28, halign="center", height=70)
        layout.add_widget(instrucciones)

        fila_text_1 = BoxLayout(orientation='horizontal', size_hint=(1, 0.5), spacing=10)

        num_habitaciones_label = Label(
            text="Número de habitaciones:",
            size_hint=(0.5, 1)
        )

        num_banos_label = Label(
            text="Número de baños:",
            size_hint=(0.5, 1)
        )

        fila_text_1.add_widget(num_habitaciones_label)
        fila_text_1.add_widget(num_banos_label)
        layout.add_widget(fila_text_1)

        fila_1 = BoxLayout(orientation='horizontal', size_hint=(1, 0.5), spacing=10)

        self.num_habitaciones_in = Spinner(
            background_color=(1, 0.2, 0.5, 1),
            text=self.num_habitaciones,
            values=["1", "2", "3", "4", "5", "6", "7", "8"],
            size_hint=(0.5, 1)
        )

        if(self.num_habitaciones == ""):
            self.num_habitaciones_in.text = "1"
        else:
            self.num_habitaciones_in.text = self.num_habitaciones
        
        self.num_banos_in = Spinner(
            background_color=(1, 0.2, 0.5, 1),
            text=self.num_habitaciones,
            values=["0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5"],
            size_hint=(0.5, 1)
        )

        if(self.num_banos == ''):
            self.num_banos_in.text = "0.5"
        else:
            self.num_banos_in.text = self.num_banos

        fila_1.add_widget(self.num_habitaciones_in)
        fila_1.add_widget(self.num_banos_in)
        layout.add_widget(fila_1)

        fila_text_2 = BoxLayout(orientation='horizontal', size_hint=(1, 0.5), spacing=10)

        num_habitaciones_label = Label(
            text="Área en metros cuadrados:",
            size_hint=(0.5, 1)
        )

        num_banos_label = Label(
            text="Código postal::",
            size_hint=(0.5, 1)
        )

        fila_text_2.add_widget(num_habitaciones_label)
        fila_text_2.add_widget(num_banos_label)
        layout.add_widget(fila_text_2)

        fila_2 = BoxLayout(orientation='horizontal', size_hint=(1, 0.5), spacing=10)

        if(self.metros_cuadrados == ''):
            self.metros_cuadrados_in = TextInput(hint_text="Ingresa el área en metros cuadrados", multiline=False, halign="center", size_hint=(0.5, 1))
        else:
            self.metros_cuadrados_in = TextInput(text=self.metros_cuadrados, multiline=False, halign="center", size_hint=(0.5, 1))

        if(self.codigo_postal == ''):
            self.codigo_postal_in = TextInput(hint_text="Ingresa el código postal", multiline=False, halign="center", size_hint=(0.5, 1))
        else:
            self.codigo_postal_in = TextInput(text=self.codigo_postal, multiline=False, halign="center", size_hint=(0.5, 1))
            
        fila_2.add_widget(self.metros_cuadrados_in)
        fila_2.add_widget(self.codigo_postal_in)
        layout.add_widget(fila_2)

        layout.add_widget(Label(size_hint=(None, 0.1)))

        # Botones para cargar imágenes
        fila_3 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, spacing=10, pos_hint={'x': 0.15})
        btn_img_cocina = Button(text="Seleccionar imagen de la cocina", background_color = (0, 0, 0, 1), size_hint=(0.35, 1))
        btn_img_cocina.bind(on_release=lambda *args: self.obtener_img(0))
        label_img_cocina = Label(text=self.imagenes[0], halign="left", valign="middle", font_size=18, size_hint=(0.5, 1))
        label_img_cocina.bind(size=label_img_cocina.setter('text_size'))
        fila_3.add_widget(btn_img_cocina)
        fila_3.add_widget(label_img_cocina)
        layout.add_widget(fila_3)

        fila_4 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, spacing=10, pos_hint={'x': 0.15})
        btn_img_habitacion = Button(text="Seleccionar imagen de la habitación principal", background_color = (0, 0, 0, 1), size_hint=(0.35, 1))
        btn_img_habitacion.bind(on_release=lambda *args: self.obtener_img(1))
        label_img_habitacion = Label(text=self.imagenes[1], halign="left", valign="middle", size_hint=(0.5, 1))
        label_img_habitacion.bind(size=label_img_habitacion.setter('text_size'))
        fila_4.add_widget(btn_img_habitacion)
        fila_4.add_widget(label_img_habitacion)
        layout.add_widget(fila_4)

        fila_5 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, spacing=10, pos_hint={'x': 0.15})
        btn_img_bano = Button(text="Seleccionar imagen del baño", background_color = (0, 0, 0, 1), size_hint=(0.35, 1))
        btn_img_bano.bind(on_release=lambda *args: self.obtener_img(2))
        label_img_bano = Label(text=self.imagenes[2], halign="left", valign="middle", size_hint=(0.5, 1))
        label_img_bano.bind(size=label_img_bano.setter('text_size'))
        fila_5.add_widget(btn_img_bano)
        fila_5.add_widget(label_img_bano)
        layout.add_widget(fila_5)

        fila_6 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, spacing=10, pos_hint={'x': 0.15})
        btn_img_frontal = Button(text="Seleccionar imagen frontal del inmueble", background_color = (0, 0, 0, 1), size_hint=(0.35, 1))
        btn_img_frontal.bind(on_release=lambda *args: self.obtener_img(3))
        label_img_frontal = Label(text=self.imagenes[3], halign="left", valign="middle", size_hint=(0.5, 1))
        label_img_frontal.bind(size=label_img_frontal.setter('text_size'))
        fila_6.add_widget(btn_img_frontal)
        fila_6.add_widget(label_img_frontal)
        layout.add_widget(fila_6)

        # Botones para generar avalúo y limpiar
        btn_generar = Button(text="Generar Avalúo", background_color=(0, 1, 0, 1), size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        btn_generar.bind(on_release=self.validar_campos)
        layout.add_widget(btn_generar)

        btn_limpiar = Button(text="Limpiar", background_color=(1, 0, 0, 1), size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        btn_limpiar.bind(on_release=lambda *args: self.limpiar_form(1))
        layout.add_widget(btn_limpiar)

        layout.add_widget(Label(size_hint=(None, 0.1)))

        self.edicion = False

        self.root.add_widget(layout)

    def pantalla_inicio(self, *args):
        self.crear_interfaz_inicio()
    
    def obtener_img(self, tipo_imagen):
        # Abrir el cuadro de diálogo para seleccionar una imagen
        ruta_img = filechooser.open_file(
            title="Selecciona una imagen",
            filters=[("Archivos de imagen", "*.png;*.jpg;*.jpeg")]
        )

        # Validar si el usuario seleccionó una imagen
        if ruta_img:
            ruta_img = ruta_img[0]

            # Actualizar las variables de entrada
            self.num_habitaciones = self.num_habitaciones_in.text
            self.num_banos = self.num_banos_in.text
            self.metros_cuadrados = self.metros_cuadrados_in.text
            self.codigo_postal = self.codigo_postal_in.text

            # Guardar la imagen en el diccionario
            self.imagenes[tipo_imagen] = ruta_img
            print(self.imagenes)

            # Continuar con el proceso
            self.empezar_avaluo()

    def resultado_avaluo(self, *args):
        self.root.clear_widgets()
        layout = ColoredBoxLayout(orientation='vertical')

        layout.add_widget(Label(size_hint=(1, 0.1)))

        layout_encabezado = ColoredBoxLayout(orientation='vertical', size_hint=(1, 0.7), spacing=10)

        btn_inicio = Button(text="Inicio", background_color=(0, 0.5, 0.5, 1), size_hint=(None, None), size=(150, 40), pos_hint={'x': 0.01, 'y': 0.01})
        btn_inicio.bind(on_release=self.pantalla_inicio)
        layout_encabezado.add_widget(btn_inicio)

        if(self.resultado != 0.0):
            resultado = Label(text="Avalúo generado correctamente", font_size=28, halign="center", size_hint=(0.5, None), pos_hint={'x': 0.25})
            with resultado.canvas.before:
                Color(0, 1, 0, 1)
                rect_fondo = Rectangle(size=resultado.size, pos=resultado.pos)

            resultado.bind(size=lambda instance, value: setattr(rect_fondo, 'size', instance.size),
                        pos=lambda instance, value: setattr(rect_fondo, 'pos', instance.pos))

            layout_encabezado.add_widget(resultado)

            layout_encabezado.add_widget(Label(size_hint=(1, 0.4)))

            info = Label(text="El precio estimado del inmueble es de:\n $ " + str(self.resultado) + " USD", font_size=28, halign="center")
            layout_encabezado.add_widget(info)
            layout_encabezado.add_widget(Label(size_hint=(1, 0.1)))

        else:
            resultado = Label(text="Error al generar el avalúo", font_size=28, halign="center", size_hint=(0.5, None), pos_hint={'x': 0.25})
            with resultado.canvas.before:
                Color(1, 0, 0, 1)
                rect_fondo = Rectangle(size=resultado.size, pos=resultado.pos)

            resultado.bind(size=lambda instance, value: setattr(rect_fondo, 'size', instance.size),
                        pos=lambda instance, value: setattr(rect_fondo, 'pos', instance.pos))

            layout_encabezado.add_widget(resultado) 
            
            layout_encabezado.add_widget(Label(size_hint=(1, 0.4)))

            info = Label(text="", font_size=28, halign="center")
            layout_encabezado.add_widget(info)
            layout_encabezado.add_widget(Label(size_hint=(1, 0.1)))

        layout.add_widget(layout_encabezado)

        linea_blanca = Label(size_hint=(1, 0.05), size=(0, 10))
        with linea_blanca.canvas.before:
            Color(1, 1, 1, 1)
            rect_fondo_linea = Rectangle(size=linea_blanca.size, pos=linea_blanca.pos)

        linea_blanca.bind(size=lambda instance, value: setattr(rect_fondo_linea, 'size', instance.size),
                          pos=lambda instance, value: setattr(rect_fondo_linea, 'pos', instance.pos))

        #layout_encabezado.add_widget(linea_blanca)
        layout.add_widget(linea_blanca)

        layout_encabezado.add_widget(Label())

        layout_2 = ColoredBoxLayout(orientation='vertical', size_hint=(1, 0.2))
        layout.add_widget(layout_2)

        if(self.resultado != 0.0):
            layout_reporte = ColoredBoxLayout(orientation='vertical', size_hint=(1, 0.3))

            instrucciones_1 = Label(text="Oprima el botón de descargar para obtener el\ndocumento con la información del avalúo.", font_size=28, halign="center")
            layout_reporte.add_widget(instrucciones_1)

            btn_descargar = Button(text="Descargar", background_color=(0, 0, 1, 1), size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
            btn_descargar.bind(on_release=self.reporte_pdf)
            layout_reporte.add_widget(btn_descargar)

            layout.add_widget(layout_reporte)

            layout.add_widget(Label(size_hint=(None, 0.1)))

        layout_editar = ColoredBoxLayout(orientation='vertical', size_hint=(1, 0.3))

        instrucciones_2 = Label(text="Oprima el botón de editar para regresar a la pantalla\nanterior y modificar la información ingresada.", size_hint=(1, 0.7), font_size=28, halign="center", valign="bottom")
        layout_editar.add_widget(instrucciones_2)

        btn_editar = Button(text="Editar", background_color=(1, 1, 0, 1), size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        btn_editar.bind(on_release=self.validar_campos)
        layout_editar.add_widget(btn_editar)

        layout.add_widget(layout_editar)

        """ layout_mitad = ColoredBoxLayout(orientation='vertical', size_hint=(1, 0.5))
        layout.add_widget(layout_mitad) """

        layout_3 = ColoredBoxLayout(orientation='vertical', size_hint=(1, 0.5))
        layout.add_widget(layout_3)

        self.edicion = True

        self.root.add_widget(layout)

    def reporte_pdf(self, *args):
        mapdf.generar_reporte_avaluo(self.imagenes[0], self.imagenes[1], self.imagenes[2], self.imagenes[3], 
                                    [self.num_habitaciones, self.num_banos, self.metros_cuadrados, self.codigo_postal], self.resultado)
        
        # Crear el contenido del popup
        popup_content = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Agregar un Label con el mensaje
        label = Label(text="Reporte de Avalúo generado correctamente.",
                    size_hint=(1, None), height=50,
                    halign='center', valign='middle')
        popup_content.add_widget(label)

        # Crear el Popup
        popup = Popup(title="Notificación",
                    content=popup_content,
                    size_hint=(None, None),
                    size=(500, 150),
                    auto_dismiss=True) 

        popup.open()

    def limpiar_form(self, accion):
        self.imagenes = ["Ninguna imagen subida"] * 4
        self.num_habitaciones = ''
        self.num_banos = ''
        self.metros_cuadrados = ''
        self.codigo_postal = ''
        self.resultado = ''

        if(accion == 1):
            self.empezar_avaluo() 

    def validar_campos(self, *args):
        try:
            self.num_habitaciones = self.num_habitaciones_in.text
            self.num_banos = self.num_banos_in.text
            self.metros_cuadrados = self.metros_cuadrados_in.text
            self.codigo_postal = self.codigo_postal_in.text

            if(self.edicion):
                self.empezar_avaluo()
            else:
                # Conexión con modelo y estimación
                #self.resultado = "100"
                try:
                    self.resultado = nom.normalizar_inmueble(self.imagenes[0], self.imagenes[1], self.imagenes[2], self.imagenes[3], 
                                    [self.num_habitaciones, self.num_banos, self.metros_cuadrados, self.codigo_postal])
                    self.resultado_avaluo()
                except:
                    self.resultado = 0.0
                    self.resultado_avaluo()
        except ValueError as e:
            print(f"Error de validación: {e}")

if __name__ == "__main__":
    AvaluoApp().run()